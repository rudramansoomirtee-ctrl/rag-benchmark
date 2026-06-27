"""IR metrics + deterministic answer-correctness scoring.

For retrieval: precision@k and recall@k against ground-truth chunk IDs.
For answers: `contains_match` (normalized containment) is the PRIMARY metric — it
matches the MultiHop-RAG paper (Tang & Yang 2024), whose gold answers are short
factoids (yes/no, entity, before/after) scored by presence in the response.
`exact_match` is a stricter SECONDARY variant; the CRAG LLM-as-judge lives in judge.py.
"""
import re
import string
from collections import Counter


def _article_id(chunk_id: str) -> str:
    """Map a passage ID `<url>#p<n>` back to its parent article URL.

    Lets IR metrics work uniformly across article-granular and passage-granular
    chunking: a retrieved passage counts as a hit if its parent URL is in the
    URL-keyed gold list (MultiHop-RAG's native gold format). Article-level IDs
    (no `#`) are returned unchanged, so legacy data still scores correctly.
    """
    return chunk_id.split("#", 1)[0]


def precision_at_k(retrieved: list[str], relevant: list[str], k: int = 5) -> float:
    """Fraction of top-k retrieved whose parent article is in the relevant set."""
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    relevant_set = set(relevant)
    return sum(1 for r in top if _article_id(r) in relevant_set) / k


def recall_at_k(retrieved: list[str], relevant: list[str], k: int = 5) -> float:
    """Fraction of relevant articles covered by ≥1 retrieved passage in top-k."""
    if not relevant:
        return 0.0
    retrieved_articles = {_article_id(r) for r in retrieved[:k]}
    return sum(1 for r in relevant if r in retrieved_articles) / len(relevant)


def reciprocal_rank_at_k(retrieved: list[str], relevant: list[str], k: int = 10) -> float:
    """1 / (rank of the first retrieved passage whose parent is relevant) within top-k.
    Mean → MRR@k."""
    relevant_set = set(relevant)
    for i, r in enumerate(retrieved[:k], start=1):
        if _article_id(r) in relevant_set:
            return 1.0 / i
    return 0.0


def average_precision_at_k(retrieved: list[str], relevant: list[str], k: int = 10) -> float:
    """Average precision over the relevant articles hit in top-k; mean → MAP@k.

    Under passage-granular retrieval against URL-keyed gold, only the FIRST
    passage from each relevant article counts toward hits — later passages from
    the same article would otherwise let hits/precision-at-k accumulate past
    1.0 (a bug observed on the n=41 sanity run that produced MAP@10 = 2.14).
    Article-level retrieval (no `#` in chunk_ids) is unchanged: each article
    appears at most once anyway.
    """
    if not relevant:
        return 0.0
    relevant_set = set(relevant)
    seen_articles: set[str] = set()
    hits = 0
    score = 0.0
    for i, r in enumerate(retrieved[:k], start=1):
        aid = _article_id(r)
        if aid in relevant_set and aid not in seen_articles:
            seen_articles.add(aid)
            hits += 1
            score += hits / i
    return score / min(len(relevant_set), k)


def hit_at_k(retrieved: list[str], relevant: list[str], k: int = 10) -> float:
    """Hit rate: 1.0 if at least one passage whose parent URL is relevant is in
    top-k. Mean → Hits@k."""
    relevant_set = set(relevant)
    return 1.0 if any(_article_id(r) in relevant_set for r in retrieved[:k]) else 0.0


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return s


def exact_match(predicted: str, gold: str) -> bool:
    """Normalized full-string equality. Stricter SECONDARY metric (the paper uses containment)."""
    if not predicted or not gold:
        return False
    return _normalize(predicted) == _normalize(gold)


# MultiHop-RAG marks null queries (corpus contains no answer) with gold
# "Insufficient information.". Systems trained on instruction-following refuse
# correctly but rarely use that exact phrase — they say "The provided context
# does not contain the answer.", "no information", etc. Without this equivalence
# table, correct refusals score as wrong and null-type accuracy reads 0% even
# when every system is doing the right thing.
_REFUSAL_GOLD_NORMS = {"insufficient information"}
_REFUSAL_PATTERNS = (
    "does not contain",
    "cannot be answered",
    "no information",
    "insufficient",
    "cannot determine",
    "unable to answer",
    "no answer",
    "not enough information",
    "not provided",
)


# Markers used by CoT-style answer prompts (System F-tuned and any future ones).
# When present, only the text AFTER the LAST marker is scored — prevents
# instruction-echo from leaking gold-shaped substrings into the prediction.
# Caught in dissertation work after qid=2252 was scored True because the CoT
# instruction "must contain the literal entity/yes-no/number" embedded "yes"
# as a substring, which Nova Lite echoed back into its response.
_ANSWER_MARKERS = (
    "final answer:",
    "final answer is:",
)


def _post_marker(predicted: str) -> str:
    """If the prediction contains a 'Final answer:' marker, return only what
    follows the LAST occurrence. Otherwise return the prediction unchanged."""
    lower = predicted.lower()
    best = -1
    best_len = 0
    for marker in _ANSWER_MARKERS:
        idx = lower.rfind(marker)
        if idx > best:
            best = idx
            best_len = len(marker)
    if best >= 0:
        return predicted[best + best_len:]
    return predicted


# Entity suffixes that gold answers often carry but predictions usually drop —
# "Everton Football Club" vs "Everton", "Apple Inc." vs "Apple", etc. Stripping
# the suffix from the normalized gold lets a bare-entity prediction still match.
_ENTITY_SUFFIXES = (
    " football club",
    " fc",
    " corporation",
    " corp",
    " inc",
    " incorporated",
    " company",
    " co",
    " ltd",
    " limited",
    " plc",
    " gmbh",
    " sa",
    " ag",
)


def _strip_entity_suffix(norm: str) -> str:
    """Strip a trailing known entity suffix from a normalized string."""
    for suf in _ENTITY_SUFFIXES:
        if norm.endswith(suf):
            return norm[: -len(suf)].strip()
    return norm


def _contains_phrase(haystack: str, needle: str) -> bool:
    """Whole-word containment: `needle` must appear bounded by word boundaries.

    Plain substring containment false-positives on short gold: gold 'No' matches
    inside 'not'/'cannot'/'does not contain', so a *refusal* scored correct on a
    yes/no question (qid 514/841). Word boundaries fix this (and the symmetric
    'AI' inside 'trained' case). Both args are pre-normalized (lowercase,
    punctuation-stripped, single-spaced), so \\b is well-defined over [a-z0-9 ]."""
    if not needle:
        return False
    return re.search(r"\b" + re.escape(needle) + r"\b", haystack) is not None


def contains_match(predicted: str, gold: str) -> bool:
    """Normalized containment: does the gold answer appear inside the prediction?

    PRIMARY correctness metric — the MultiHop-RAG paper (Tang & Yang 2024) scores
    a short factoid gold (yes/no, entity, before/after) by presence in the response.

    Containment is whole-word (word-boundary, not naive substring — see
    `_contains_phrase`), so short gold like 'No' is not matched inside 'not'.
    Three further layers of robustness:
      (1) Post-marker extraction — if the prediction has a 'Final answer:' marker,
          we score only what FOLLOWS the last marker. Stops CoT prompts from
          poisoning scoring by echoing gold-shaped tokens in their instructions
          (the qid=2252 'yes-no/number' instruction-echo false-positive).
      (2) Refusal equivalence — gold 'Insufficient information.' matches any
          standard refusal phrasing in the prediction.
      (3) Entity-suffix stripping — gold 'Everton Football Club' matches a
          prediction containing only 'Everton' (suffixes 'football club', 'inc',
          'corp', 'ltd', etc. are dropped before the final containment check).
    """
    if not predicted or not gold:
        return False
    p_text = _post_marker(predicted)
    p, g = _normalize(p_text), _normalize(gold)
    if g in _REFUSAL_GOLD_NORMS:
        return any(pat in p for pat in _REFUSAL_PATTERNS)
    if _contains_phrase(p, g):
        return True
    g_stripped = _strip_entity_suffix(g)
    if g_stripped and g_stripped != g and _contains_phrase(p, g_stripped):
        return True
    return False


def answer_match(predicted: str, golds: list[str]) -> bool:
    """Correct if the prediction matches the gold OR any of its aliases.

    For datasets that ship multiple acceptable surface forms (MuSiQue's
    `answer_aliases`). Beyond `contains_match` (gold ⊆ prediction, with the
    refusal/entity-suffix layers), this also credits the *reverse* whole-word
    containment — the committed answer ⊆ gold — so a correct-but-terser answer
    still scores: 'Final answer: 140' vs gold '140 mi', '25,000' vs 'nearly
    25,000', '92%' vs '92 percent', 'Paraguay' vs "Alfredo Stroessner's
    Paraguay". Both directions are word-boundary matched, so genuinely different
    answers (e.g. 'Epic Records' vs 'MGM Records') match in neither direction and
    stay wrong. The committed span is taken post-'Final answer:' marker.
    """
    p = _normalize(_post_marker(predicted or ""))
    for g in golds:
        if not g:
            continue
        if contains_match(predicted, g):
            return True
        gn = _normalize(g)
        if len(p) >= 2 and gn and _contains_phrase(gn, p):
            return True
    return False


# SQuAD-style F1 drops articles before token overlap, matching the standard
# QA-F1 convention (Rajpurkar et al. 2016) so the metric is comparable to the
# answer-F1 reported by Ammann et al. (2025) rather than a bespoke variant.
_ARTICLES = {"a", "an", "the"}


def _f1_tokens(s: str) -> list[str]:
    """Normalized, article-stripped token list for token-overlap F1."""
    return [t for t in _normalize(s).split() if t not in _ARTICLES]


def token_f1(predicted: str, gold: str) -> float:
    """SQuAD-style token-overlap F1 between prediction and gold (0..1).

    Complementary SECONDARY metric to `contains_match`: where containment is a
    binary "is the gold factoid present", token F1 is a graded lexical-overlap
    score, and the two are *meant* to diverge (that divergence is the metric
    audit, RQ4/O6). Two layers are shared with the primary metric so the columns
    stay comparable per question type:
      (1) Post-marker extraction — score only text after the last 'Final answer:'
          so CoT answers (System F-tuned) are scored on the answer, not the chain.
      (2) Refusal equivalence — gold 'Insufficient information.' scores 1.0 against
          any standard refusal phrasing, so null-type F1 is meaningful rather than
          uniformly ~0. Entity-suffix stripping is intentionally NOT applied —
          token overlap already partially credits 'Everton' vs 'Everton FC'.
    """
    if not predicted or not gold:
        return 0.0
    p_text = _post_marker(predicted)
    if _normalize(gold) in _REFUSAL_GOLD_NORMS:
        p_norm = _normalize(p_text)
        return 1.0 if any(pat in p_norm for pat in _REFUSAL_PATTERNS) else 0.0
    pred_toks = _f1_tokens(p_text)
    gold_toks = _f1_tokens(gold)
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)

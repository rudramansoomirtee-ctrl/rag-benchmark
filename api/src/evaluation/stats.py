"""Pure-Python statistical helpers shared by the analysis notebook and the API.

These were originally defined inline in `notebooks/analysis.py` cells. Hoisting them
here gives a single source of truth so the web dashboard, `compute-metrics`, and the
dissertation figures can never drift. Deliberately **stdlib-only** (no numpy) so the
`api` service can import them; `bootstrap_ci` uses a seeded `random.Random` for
determinism, so repeated calls — and the notebook — yield identical CIs.
"""
import math
import random
import statistics

from src.evaluation.metrics import _article_id


def _aslist(x) -> list:
    """JSONB columns arrive as Python lists, but coerce defensively."""
    if x is None:
        return []
    if isinstance(x, str):
        import json
        try:
            return json.loads(x)
        except Exception:
            return []
    return list(x)


def percentile(values, q: float) -> float:
    """Linear-interpolated percentile (matches numpy's default 'linear' method)."""
    v = sorted(float(x) for x in values)
    if not v:
        return float("nan")
    if len(v) == 1:
        return v[0]
    rank = (q / 100.0) * (len(v) - 1)
    lo_i = math.floor(rank)
    hi_i = math.ceil(rank)
    if lo_i == hi_i:
        return v[int(rank)]
    frac = rank - lo_i
    return v[lo_i] * (1 - frac) + v[hi_i] * frac


def bootstrap_ci(values, n_boot: int = 2000, seed: int = 0, lo: float = 2.5, hi: float = 97.5):
    """Percentile bootstrap CI for the mean of a 0/1 (or numeric) sequence."""
    v = [float(x) for x in values]
    n = len(v)
    if n == 0:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        total = 0.0
        for _ in range(n):
            total += v[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    return (percentile(means, lo), percentile(means, hi))


def kendall_tau_b(x, y) -> float:
    """Rank-correlation with tie handling. +1 identical order, -1 reversed."""
    x, y, n = list(x), list(y), len(x)
    C = D = Tx = Ty = 0
    for i in range(n):
        for j in range(i + 1, n):
            sx = (x[i] > x[j]) - (x[i] < x[j])
            sy = (y[i] > y[j]) - (y[i] < y[j])
            if sx == 0 and sy == 0:
                continue
            if sx == 0:
                Ty += 1
            elif sy == 0:
                Tx += 1
            elif sx == sy:
                C += 1
            else:
                D += 1
    denom = ((C + D + Tx) * (C + D + Ty)) ** 0.5
    return (C - D) / denom if denom else float("nan")


def pareto_frontier(points):
    """Non-dominated (label, accuracy, cost): no other point has higher-or-equal
    accuracy at lower-or-equal cost. Returned ordered by accuracy."""
    front = [
        (lbl, a, c) for lbl, a, c in points
        if not any(
            a2 >= a and c2 <= c and (a2 > a or c2 < c)
            for l2, a2, c2 in points if l2 != lbl
        )
    ]
    return sorted(front, key=lambda t: t[1])


def covered(retrieved, relevant):
    """True if >=1 gold article is present in the retrieved set; None when the query
    has no gold evidence (null-type) so it is excluded from the retrieval ceiling.

    Passage ids `<url>#p<i>` are mapped back to their parent article via
    `metrics._article_id`, matching how IR metrics score URL-keyed gold."""
    rel = set(_aslist(relevant))
    if not rel:
        return None
    return len({_article_id(c) for c in _aslist(retrieved)} & rel) > 0


def agreement_rate(a, b) -> float:
    """Fraction of positions where two binary sequences agree, over positions where
    BOTH are non-None. NaN if they never co-occur."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    if not pairs:
        return float("nan")
    return sum(1 for x, y in pairs if x == y) / len(pairs)


def stdev(values) -> float:
    """Sample standard deviation; 0.0 for a single value, NaN for empty."""
    v = [float(x) for x in values]
    if not v:
        return float("nan")
    if len(v) < 2:
        return 0.0
    return statistics.stdev(v)

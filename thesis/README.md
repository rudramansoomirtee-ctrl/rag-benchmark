# Thesis folder — complete write-up package

Everything needed to write and assemble the dissertation lives in this folder. Start with
**WRITING_GUIDE.md** (order, voice rules, ground rules), write chapters in the order
Ch3 → Ch4 → Ch2 → Ch1 → Ch5, abstract last.

| File / folder | What it is | Your action |
|---|---|---|
| `WRITING_GUIDE.md` | The playbook: writing order, resource map, voice checklist, verified citation base | read first |
| `front_matter.md` | Abstract draft, AI-usage declaration draft, preliminaries checklist, assembly order | rewrite abstract; adapt declaration to faculty form |
| `chapter1_introduction.md` | Ch1 draft — context, gaps, aim/objectives/RQs, contributions, scope | voice rewrite; fill the one `[CONFIRM]` (repository statement) |
| `chapter2_literature.md` | Ch2 draft — 4 orchestration families, retrieval substrate, evaluation practice, gaps | voice rewrite; one flagged claim to re-check (see its end-note) |
| `chapter3_methodology.md` | Ch3 draft — design, 8 systems, substrate, models, datasets, metrics, statistics (§3.8), reproducibility (§3.9) | voice rewrite |
| `chapter4_results.md` | Ch4 draft ("Results and Discussion") — both arms, CIs, Holm, behavioural analyses, cross-dataset synthesis | voice rewrite |
| `chapter5_conclusions.md` | Ch5 draft — RQ answers, contributions, recommendations, limitations, future work | voice rewrite |
| `references_draft.md` | 24 Harvard entries, all web-verified with source URLs and full author lists | strip status tags at assembly |
| `appendices/` | A cost survey · B ablations · C breakdowns+cost+behaviour · D statistics · E integrity audit · F reproducibility | reformat at Word conversion |
| `figures/` | Figures 3.1–3.7 (PNG + Mermaid source) and 4.1–4.4 (PNG + generating `make_figure_4_*.py` scripts) | insert at Word conversion |
| `musique_matrix_analysis.md` | The verified final-matrix analysis record (source of every number) — NOT a thesis chapter, NOT citable | reference only |

Word count: chapters total ≈ 12,700 words net of draft banners — inside the 12,000–14,000 band.

Status of every number: verified against the Postgres record by independent audit passes
(2026-07-12). Strip all `> **Status:**` banners and end-notes before any submission or share.

Related project records that stay at the repo root (their thesis-relevant content is already
copied into the appendices): `RELATED_WORK.md`, `DISSERTATION_AUDIT.md`.

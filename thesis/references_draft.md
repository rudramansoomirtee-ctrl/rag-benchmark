# References — working draft (Harvard style)

> **Status: skeleton, NOT submission-ready.** Built only from the verified citation base
> (WRITING_GUIDE §4). Tags: ✅ = core details primary-verified during this project;
> ⚠ = re-verify against the published PDF before print; `[verify: …]` = a specific field that must
> be confirmed from the primary source before the entry is usable. **Never** copy an entry from
> chap 2.docx or BG.pdf — both contain confirmed fabricated metadata (invented author tails and
> venues on Lewis 2020, Karpukhin 2020, Asai 2023/2024). Rebuild those from the primary source.
>
> House rule: no citation goes into a chapter unless its entry here is ✅ with no `[verify]` tags.

---

Ammann, N., Golde, J. and Akbik, A. (2025) 'Question decomposition for retrieval-augmented
generation', *Proceedings of the ACL 2025 Student Research Workshop*. arXiv:2507.00355. ✅
(headline numbers verified; the "lack of iterative retrieval" limitation wording is a paraphrase —
verify against the PDF before quoting verbatim)

Asai, A., Wu, Z., Wang, Y., Sil, A. and Hajishirzi, H. (2024) 'Self-RAG: learning to retrieve,
generate, and critique through self-reflection', *International Conference on Learning
Representations (ICLR)*. arXiv:2310.11511. ✅ web-verified https://arxiv.org/abs/2310.11511 ,
https://openreview.net/forum?id=hSyW5go0v8 (ICLR 2024 Oral; author list and venue confirmed —
matches the pre-existing entry, no correction needed. BG.pdf's version of this entry was fabricated;
this rebuilt entry is safe to use)

Chen, J., Xiao, S., Zhang, P., Luo, K., Lian, D. and Liu, Z. (2024) 'BGE M3-Embedding: multi-lingual,
multi-functionality, multi-granularity text embeddings through self-knowledge distillation',
arXiv:2402.03216. ✅ web-verified https://arxiv.org/abs/2402.03216 ,
https://huggingface.co/BAAI/bge-reranker-v2-m3 (NEW ENTRY — full 6-author list and title confirmed
from the arXiv abstract page; no conference/journal venue listed on arXiv, cite as an arXiv
preprint). **Scope note:** this paper covers the `bge-m3` embedding model (multi-lingual,
multi-vector/dense/sparse retrieval), not `bge-reranker-v2-m3` directly. However, it is the
citation BAAI's own model card gives as the *secondary* reference for `bge-reranker-v2-m3` (used
as this project's local reranker, `settings.reranker_model`) — the model card's *primary* reference
for the reranker is a different paper, Li, C., Liu, Z., Xiao, S. and Shao, Y. (2023) 'Making large
language models a better foundation for dense retrieval', arXiv:2312.15503 (not independently
verified here — check before citing as the reranker's primary source).

Chu, Z., Chen, J., Chen, Q., Wang, H., Zhu, K., Du, X., Yu, W., Liu, M. and Qin, B. (2024)
'BeamAggR: beam aggregation reasoning over multi-source knowledge for multi-hop question
answering', *Proceedings of the 62nd Annual Meeting of the Association for Computational
Linguistics (Volume 1: Long Papers)*, pp. 1229–1248. arXiv:2406.19820. ✅ web-verified
https://aclanthology.org/2024.acl-long.67/ (full 9-author list confirmed; ACL 2024 main
conference, not Findings) Table 1 values verified — cite table values only; the abstract's "+8.5%"
headline does not reconcile with the table.

Cormack, G.V., Clarke, C.L.A. and Büttcher, S. (2009) 'Reciprocal rank fusion outperforms
Condorcet and individual rank learning methods', *Proceedings of the 32nd International ACM SIGIR
Conference on Research and Development in Information Retrieval*, pp. 758–759. ✅ web-verified
https://cormack.uwaterloo.ca/cormacksigir09-rrf.pdf , https://dl.acm.org/doi/10.1145/1571941.1572114
(2-page short paper, SIGIR'09, 19–23 July 2009, Boston, MA. Verified from the primary PDF: exact
printed title capitalization is "Reciprocal Rank Fusion outperforms Condorcet and individual Rank
Learning Methods" — note lowercase "individual" and capitalized "Rank Learning Methods"; author
order and affiliations G.V. Cormack/C.L.A. Clarke (Univ. Waterloo), S. Büttcher (Google) confirmed
from the PDF header; the RRF formula in the paper is `RRFscore(d) = Σ 1/(k + r(d))` with k=60 fixed
during a pilot run and not altered afterward — cite this exact formula/constant if referencing RRF's
mechanics)

Es, S., James, J., Espinosa-Anke, L. and Schockaert, S. (2024) 'RAGAs: automated evaluation of
retrieval augmented generation', *Proceedings of the 18th Conference of the European Chapter of
the Association for Computational Linguistics: System Demonstrations*, pp. 150–158. arXiv:2309.15217.
✅ web-verified https://aclanthology.org/2024.eacl-demo.16/ (NEW ENTRY — full 4-author list, exact
title "RAGAs: Automated Evaluation of Retrieval Augmented Generation" and venue confirmed: EACL
2024 System Demonstrations track, St Julians, Malta, pp. 150–158 — not just an arXiv preprint)

Gutiérrez, B.J., Shu, Y., Gu, Y., Yasunaga, M. and Su, Y. (2024) 'HippoRAG: neurobiologically
inspired long-term memory for large language models', *Advances in Neural Information Processing
Systems (NeurIPS)*. arXiv:2405.14831. ✅ web-verified https://arxiv.org/abs/2405.14831 , PDF
Appendix G ("Cost and Efficiency Comparison", Table 17) (full 5-author list confirmed; NeurIPS
2024). **Cost claim CORRECTED — the draft's premise was wrong.** The paper does NOT report
"~$0.10 per 1,000 queries for offline indexing vs $1–3 for iterative retrieval." Table 17 ("Average
cost and efficiency measurements for online RETRIEVAL using GPT-3.5 Turbo on 1,000 queries") gives
API cost $0.1 for HippoRAG vs $1–3 for IRCoT — that figure is for **online retrieval**, not offline
indexing. Offline **indexing** (Table 18, per 10,000 passages) is actually *more* expensive for
HippoRAG than IRCoT/ColBERTv2: "offline indexing time and costs are higher for HippoRAG than
IRCoT — around 10 times slower and $15 more expensive for every 10,000 passages" (HippoRAG $15 vs
IRCoT/ColBERTv2 $0, GPT-3.5 Turbo-1106). Any thesis claim must say HippoRAG is cheap at *online
retrieval* time ($0.10 vs $1–3 per 1,000 queries), not at indexing time — indexing is where it costs
more.

Jeong, S., Baek, J., Cho, S., Hwang, S.J. and Park, J.C. (2024) 'Adaptive-RAG: learning to adapt
retrieval-augmented large language models through question complexity', *Proceedings of NAACL
2024*. arXiv:2403.14403. ✅ web-verified https://arxiv.org/abs/2403.14403 (full 5-author list
confirmed) (Table 8 containment numbers verified)

Jiang, Z., Xu, F.F., Gao, L., Sun, Z., Liu, Q., Dwivedi-Yu, J., Yang, Y., Callan, J. and Neubig, G.
(2023) 'Active retrieval augmented generation', *Proceedings of EMNLP 2023*. arXiv:2305.06983.
✅ web-verified https://arxiv.org/abs/2305.06983 (FLARE; full 9-author list confirmed)

Jin, B., Zeng, H., Yue, Z., Yoon, J., Arik, S., Wang, D., Zamani, H. and Han, J. (2025) 'Search-R1:
training LLMs to reason and leverage search engines with reinforcement learning', arXiv:2503.09516
— cite **v5** (submitted 5 Aug 2025; results drifted across arXiv versions). ✅ web-verified
https://arxiv.org/abs/2503.09516 (full 8-author list and title confirmed from the arXiv abstract
page) with version-drift note retained.

Jin, J., Zhu, Y., Dong, G., Zhang, Y., Yang, X., Zhang, C., Zhao, T., Yang, Z., Dou, Z. and Wen, J.-R.
(2025) 'FlashRAG: a modular toolkit for efficient retrieval-augmented generation research',
*Proceedings of the ACM Web Conference 2025 (WWW '25), Resource Track*. arXiv:2405.13576.
✅ web-verified https://arxiv.org/abs/2405.13576 (full 10-author list confirmed; peer-reviewed venue
now exists — accepted to WWW 2025 Resource Track, DOI 10.1145/3701716.3715313 — **update the
citation year to 2025 and drop the arXiv-only framing**)

Karpukhin, V., Oğuz, B., Min, S., Lewis, P., Wu, L., Edunov, S., Chen, D. and Yih, W. (2020)
'Dense passage retrieval for open-domain question answering', *Proceedings of the 2020 Conference
on Empirical Methods in Natural Language Processing (EMNLP)*, pp. 6769–6781. arXiv:2004.04906.
✅ web-verified https://aclanthology.org/2020.emnlp-main.550/ (author list, title and page range
confirmed against the ACL Anthology page exactly as drafted — the BG.pdf fabricated tail is
correctly absent here)

Lewis, P., Perez, E., Piktus, A., Petroni, F., Karpukhin, V., Goyal, N., Küttler, H., Lewis, M.,
Yih, W., Rocktäschel, T., Riedel, S. and Kiela, D. (2020) 'Retrieval-augmented generation for
knowledge-intensive NLP tasks', *Advances in Neural Information Processing Systems 33 (NeurIPS
2020)*, pp. 9459–9474. arXiv:2005.11401. ✅ web-verified
https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5-Abstract.html
(author list confirmed against the NeurIPS proceedings page exactly as drafted; proceedings form
adds volume 33 and pages 9459–9474 — the BG.pdf fabricated tail is correctly absent here)

Liu, N.F., Lin, K., Hewitt, J., Paranjape, A., Bevilacqua, M., Petroni, F. and Liang, P. (2024)
'Lost in the middle: how language models use long contexts', *Transactions of the Association for
Computational Linguistics*, 12, pp. 157–173. arXiv:2307.03172. ✅ web-verified
https://aclanthology.org/2024.tacl-1.9/ (full 7-author list confirmed; citable published version is
TACL 2024, vol. 12, not the 2023 arXiv-only form — **use year 2024 and this venue**, not "Proceedings
of EMNLP 2023" or an arXiv-only citation; cite for the U-shaped position effect under *forced*
positions — this study's position finding is observational, frame accordingly)

Nogueira, R. and Cho, K. (2019) 'Passage re-ranking with BERT', arXiv:1901.04085. ✅ web-verified
https://arxiv.org/abs/1901.04085 (NEW ENTRY — title, 2-author list and year confirmed from the
arXiv abstract page; no peer-reviewed venue found — arXiv preprint only (cs.IR), final revision
April 2020. Cited as the origin of cross-encoder passage reranking)

Press, O., Zhang, M., Min, S., Schmidt, L., Smith, N.A. and Lewis, M. (2023) 'Measuring and
narrowing the compositionality gap in language models', *Findings of the Association for
Computational Linguistics: EMNLP 2023*. arXiv:2210.03350. ✅ web-verified
https://arxiv.org/abs/2210.03350 (full 6-author list confirmed; the arXiv page itself states "To
appear at Findings of EMNLP 2023" — cite that venue, not a bare arXiv preprint) (2-hop subset
numbers from Tables 1/14 verified)

Shao, Z., Gong, Y., Shen, Y., Huang, M., Duan, N. and Chen, W. (2023) 'Enhancing retrieval-augmented
large language models with iterative retrieval-generation synergy', *Findings of the Association
for Computational Linguistics: EMNLP 2023*, pp. 9248–9274. arXiv:2305.15294. ✅ web-verified
https://aclanthology.org/2023.findings-emnlp.620/ (Iter-RetGen; full 6-author list and venue
confirmed — Findings of EMNLP 2023, Singapore)

Song, H., Jiang, J., Min, Y., Chen, J., Chen, Z., Zhao, W.X., Fang, L. and Wen, J.-R. (2025)
'R1-Searcher: incentivizing the search capability in LLMs via reinforcement learning',
arXiv:2503.05592. ✅ web-verified https://arxiv.org/abs/2503.05592 (full 8-author list and title
confirmed)

Tang, Y. and Yang, Y. (2024) 'MultiHop-RAG: benchmarking retrieval-augmented generation for
multi-hop queries', *Conference on Language Modeling (COLM)*. arXiv:2401.15391. ✅ (dataset
construction and containment scoring verified; Table-6 GPT-4 split ⚠)

Thakur, N., Reimers, N., Rücklé, A., Srivastava, A. and Gurevych, I. (2021) 'BEIR: a heterogeneous
benchmark for zero-shot evaluation of information retrieval models', *NeurIPS Datasets and
Benchmarks Track*. arXiv:2104.08663. ✅ (abstract quote verified)

Trivedi, H., Balasubramanian, N., Khot, T. and Sabharwal, A. (2022) 'MuSiQue: multihop questions
via single-hop question composition', *Transactions of the Association for Computational
Linguistics*, 10, pp. 539–554. arXiv:2108.00573. ✅ web-verified
https://aclanthology.org/2022.tacl-1.31/ (page range confirmed: TACL vol. 10, pp. 539–554)
(distractor-construction quote verified)

Trivedi, H., Balasubramanian, N., Khot, T. and Sabharwal, A. (2023) 'Interleaving retrieval with
chain-of-thought reasoning for knowledge-intensive multi-step questions', *Proceedings of ACL
2023*. arXiv:2212.10509. ✅ (Table 4 F1 verified)

Xiao, S., Liu, Z., Zhang, P., Muennighoff, N., Lian, D. and Nie, J.-Y. (2024) 'C-Pack: packed
resources for general Chinese embeddings', *Proceedings of the 47th International ACM SIGIR
Conference on Research and Development in Information Retrieval (SIGIR '24)*, pp. 641–649.
arXiv:2309.07597. ✅ web-verified https://dl.acm.org/doi/10.1145/3626772.3657878 ,
https://arxiv.org/abs/2309.07597 (NEW ENTRY — full 6-author list confirmed; exact title "C-Pack:
Packed Resources For General Chinese Embeddings"; published at SIGIR 2024, pp. 641–649, not just an
arXiv preprint — cite year 2024). **Scope note:** this is the general BGE-embedding-family resource
paper (BGE-base/large/small); it introduces the BGE lineage but this project's *embedding* model is
`BAAI/llm-embedder` (a different BAAI model, per `config.py:embedding_model`), whose primary
citation is Zhang et al. (2023) 'Retrieve anything to augment large language models' (see entry
below) — cite that, not C-Pack, for the embedder. See the Chen et al. (2024) BGE M3-Embedding entry
above for the model that covers this project's *reranker* (`bge-reranker-v2-m3`).

Yang, X., Sun, K., Xin, H., Sun, Y., Bhalla, N., Chen, X., Choudhary, S., Gui, R.D., Jiang, Z.W.,
Jiang, Z., Kong, L., Moran, B., Wang, J., Xu, Y.E., Yan, A., Yang, C., Yuan, E., Zha, H., Tang, N.,
Chen, L., Scheffer, N., Liu, Y., Shah, N., Wanga, R., Kumar, A., Yih, W. and Dong, X.L. (2024)
'CRAG -- comprehensive RAG benchmark', *Advances in Neural Information Processing Systems 37
(NeurIPS 2024), Datasets and Benchmarks Track*. arXiv:2406.04744. ✅ web-verified
https://arxiv.org/abs/2406.04744 ,
https://proceedings.neurips.cc/paper_files/paper/2024/hash/1435d2d0fca85a84d83ddcb754f58c29-Abstract-Datasets_and_Benchmarks_Track.html
**VENUE CORRECTED — this is NOT a KDD 2024 paper.** CRAG is a NeurIPS 2024 Datasets and Benchmarks
Track paper (27 authors, full list above); KDD Cup 2024 was a separate community *challenge/
competition* built on top of this benchmark, not the venue that published the paper itself. Exact
title is "CRAG -- Comprehensive RAG Benchmark" (with the double-dash from the arXiv title, no colon).
(cited only if the LLM-judge rubric is mentioned)

Yang, Z., Qi, P., Zhang, S., Bengio, Y., Cohen, W.W., Salakhutdinov, R. and Manning, C.D. (2018)
'HotpotQA: a dataset for diverse, explainable multi-hop question answering', *Proceedings of the
2018 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pp. 2369–2380.
arXiv:1809.09600. ✅ web-verified https://aclanthology.org/D18-1259/ (full 7-author list confirmed
against the ACL Anthology page exactly as drafted)

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K. and Cao, Y. (2023) 'ReAct:
synergizing reasoning and acting in language models', *International Conference on Learning
Representations (ICLR)*. arXiv:2210.03629. ✅ web-verified https://arxiv.org/abs/2210.03629
(full 7-author list confirmed; v3 on arXiv is noted as "the ICLR camera ready version" — ICLR 2023
confirmed)

Zhang, P., Xiao, S., Liu, Z., Dou, Z. and Nie, J.-Y. (2023) 'Retrieve anything to augment large
language models', arXiv:2310.07554. ✅ web-verified https://arxiv.org/abs/2310.07554 ,
https://arxiv.org/abs/2310.07554v1 (NEW ENTRY — full 5-author list and year confirmed; this is the
primary citation for the `BAAI/llm-embedder` model this project uses as its embedder
(`config.py:embedding_model`). No peer-reviewed venue found — arXiv preprint only. **Title-drift
warning:** the v1 (Oct 2023) title is 'Retrieve Anything To Augment Large Language Models', but a
later arXiv revision (Jan 2026) retitled the paper 'A Multi-Task Embedder For Retrieval Augmented
LLMs' with the same 5 authors — cite the v1 title/year as above, or pin the arXiv version explicitly
if quoting from a specific revision)

Zhou, D., Schärli, N., Hou, L., Wei, J., Scales, N., Wang, X., Schuurmans, D., Cui, C., Bousquet, O.,
Le, Q. and Chi, E. (2023) 'Least-to-most prompting enables complex reasoning in large language
models', *International Conference on Learning Representations (ICLR)*. arXiv:2205.10625.
✅ web-verified https://arxiv.org/abs/2205.10625 (full 11-author list confirmed; ICLR 2023
confirmed) (sequential-beats-parallel specifics not quote-verified)

# Week 1 retrospective

**Period:** October 28–November 3, 2026

**Goal:** Establish measurable retrieval performance for Chat-with-Your-Repo.

**Completed:** Built an isolated benchmark covering eight backend files and
32 chunks. Prepared 25 questions: 23 answerable questions scored for retrieval
and two out-of-scope questions excluded from those metrics. The original
30-question target has not yet been reached.

Compared semantic search, BM25, hybrid RRF, token windows and weighted fusion.
Semantic search performed best overall:
Precision@5 0.252, Recall@5 0.891, MRR@5 0.812,
and rank-1 hit rate 0.696.

All 10 Week 1 tests passed. The validator reproduced all 12 recorded summary
metrics. The baseline evaluation package is preserved in commit 5d28da6.

**Latency:** Warm local retrieval, including question embedding and Chroma
search, measured p50 4.71 ms and p95 9.66 ms. This used 23 questions repeated
five times, producing 115 observations after three warm-up requests.
Model startup, LLM generation and network/request transport were excluded.

**What did not work:** BM25, hybrid RRF, token windows and weighted fusion
did not improve overall results over semantic search. Token windows recovered
some missing evidence but caused regressions elsewhere.

**Representation problem:** Eleven of the original 32 chunks exceeded the
embedding configuration's 256-token limit. The chunk starting at
Backend/ask.py:11 contained 958 tokens, including code following parse_args().
Truncation is a credible contributor to retrieval failures, not a proven sole
cause. The complete source text remained stored.

**Decision:** Retain semantic search. Continue token-aware, structure-preserving
chunking as an isolated experiment. Retrieval improvement and a new deployment
have not been established.

**Learning:** Count the complete embedding input, including context and special
tokens. Preserve exact source locations. Use source evidence rather than old
chunk IDs when comparing different chunk boundaries. Retrieval recall is not
LLM answer accuracy.

**Carry into Week 2:** Finish text-file splitting and validate the latest
splitter across the frozen eight-file corpus. Complete metadata and relevance
mappings, add fresh evaluation questions, build a separate experimental index,
and compare quality, latency and cost. Consider reranking only with a clear
hypothesis. Keep time for QA Copilot architecture, interview practice and
applications.

**Limitations:** Results cover one small backend corpus and repeatedly reused
development questions. Held-out and multi-repository validation, answer
correctness, citation fidelity and abstention remain unmeasured. A three-question
cost pilot exists; a complete cost comparison remains outstanding.

**Other weekly goals:** Independent coding, the recorded project walkthrough,
LinkedIn updates and application submissions are not verified by this technical
retrospective.

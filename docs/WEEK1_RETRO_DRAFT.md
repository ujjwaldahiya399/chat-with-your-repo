# Week 1 retrospective — draft to confirm

**Goal:** establish measurable retrieval performance for Chat-with-Your-Repo.

**What worked:** Repaired broken Python virtual environment; verified ChromaDB retrieval; built isolated 32-chunk dataset, verified 23 answerable questions and compared semantic, BM25, hybrid, token windows, and two-representation fusion. Semantic-only performed best overall (P@5 0.252, R@5 0.891, MRR@5 0.812, Top1 0.696).

**What did not work:** Hybrid and weighted fusion did not beat baseline. Simply splitting every chunk into token-bounded windows improved certain questions but reduced aggregate recall/ranking. The fixed benchmark was small and not held out; do not tune repeatedly on it.

**Root cause discovered:** The model accepts 256 tokens, but 11 of 32 chunks exceed that length; `Backend/ask.py:11` has 958 tokens. This likely hides later semantic content from the embedding while full text remains stored. Window-based retrieval recovered R19/R23; other cases regressed.

**Decision:** Keep semantic-only in the application; treat token-aware logical splitting and a held-out multi-repository evaluation as next-phase work. No production changes or deployment claimed.

**Week 1 remaining before sign-off:** Record real p50/p95 latency, verify two QA Copilot query labels and run consistent three-repository baseline if maintaining original multi-repo DoD, confirm design doc and local Git state, and perform one project explanation practice.

**Interview practice (speak 60–90 seconds):** What is the retrieval pipeline? Why can MRR go down even when recall is unchanged? Why is 11/32 overlength chunks a retrieval bug? Why keep the simpler semantic search after measuring more complex methods?

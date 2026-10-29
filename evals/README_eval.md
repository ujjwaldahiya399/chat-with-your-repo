# Chat-with-Your-Repo: 32-chunk retrieval benchmark

Prepared from the uploaded `chunk_manifest.md` for the current isolated evaluation index.

## Contents

- `golden_set_rag.json` — 25 carefully scoped questions: 23 answerable, 2 deliberately outside the indexed source. Chunk IDs are exact matches to the supplied manifest. Multiple relevant chunks are labelled where the answer spans functions/files. Each question includes the rationale for its relevance label.
- `run_rag_benchmark.py` — compares semantic retrieval, in-memory BM25, and unweighted reciprocal rank fusion (RRF) at top 5. Produces per-question rankings, scores, and an evaluation report.

## Install into your existing macOS project

1. Copy **both files** into the existing `evals/` directory, next to `build_eval_index.py`.
2. From the **project root**, activate the working environment if needed:

   ```bash
   source Backend/.venv/bin/activate
   python -m pip install rank-bm25
   ```

3. Make sure the separate database created earlier is still at `evals/chroma_benchmark/` and its collection is `rag_backend_eval` (32 chunks). No database rebuilding is necessary.
4. Run:

   ```bash
   python evals/run_rag_benchmark.py
   ```

5. The script prints three-way comparison metrics and saves:

   - `evals/rag_benchmark_results.json` — full rankings, question-level metrics, dataset and index fingerprint.
   - `evals/rag_benchmark_report.md` — summary and queries requiring attention.

## What is and is not being measured

- **Scored:** 23 answerable questions, with Precision@5, Recall@5, MRR@5 and Top-1 hit rate. Results are **retrieval metrics**, not LLM answer-quality scores.
- **Not scored:** 2 out-of-scope questions. Nearest-neighbor retrievers always return some chunks; retrieving a chunk does *not* demonstrate an ability to say "I don't know". Test answer abstention separately using a generated-answer evaluator.
- **Annotation limitations:** Gold labels are manually grounded in the supplied manifest, not independently reviewed. Historical `Backend/NOTES.md` describes some outdated behaviors; current Python code is prioritized where the question asks about current behavior. A larger corpus and adjudicated labels are needed for stronger claims.
- **Consistency:** The collection must contain the expected 32 chunks. The script reports a SHA-256 fingerprint of ordered chunk IDs and content to track whether later runs are evaluating the same snapshot.
- **Safety:** The runner is read-only against `Backend/chroma_db` (it never touches it) and does not mutate the isolated ChromaDB database. It does not call Groq or use a Groq API key. Only local result files are written.

## Design choices

- BM25 uses lowercase lexical tokens and preserves underscores in identifiers, matching your earlier pilot.
- Both semantic and BM25 contribute up to 15 candidate chunks to **unweighted RRF** (`RRF_CONSTANT=60`), and all methods are evaluated at `K=5`.
- RRF combines *rank positions*, not incompatible raw Chroma vector distances and BM25 keyword scores.
- `Precision@5` divides by five for all scored queries. One-relevant-chunk queries cannot score above 0.2 under these labels; **Recall@5, MRR@5 and Top-1 accuracy** are often more informative for this pilot.
- Do not tune fusion weights to these 23 scored questions and report that as an unbiased generalization test. Preserve a held-out set for further validation.

## Suggested next iteration

After reviewing the output, add the current FastAPI `main.py` and frontend `.ts/.tsx` sources to a newly versioned index, use a bigger repository, then create a new benchmark including routing, authentication, API behavior, and real multi-file tasks.

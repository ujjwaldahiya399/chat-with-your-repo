# Chat-with-Your-Repo — Retrieval Evaluation Design

## Objective
Evaluate whether semantic retrieval finds relevant source-code evidence for repository questions. Compare retrieval methods using fixed relevance labels and an isolated ChromaDB benchmark index.

## Evaluation Dataset
- Primary benchmark: 25 questions about Chat-with-Your-Repo.
- Answerable questions: 23.
- Out-of-scope questions: 2, excluded from retrieval relevance metrics.
- Primary corpus: 32 source chunks from eight backend files.
- Extended question tracker: 30 questions across three repositories.
- The extended tracker is not a unified three-repository retrieval benchmark.

## Relevance Labels
Relevant chunk IDs are manually assigned and verified against the evaluation manifest. A question may require multiple chunks. Out-of-scope questions have no relevant labels.

The extended tracker additionally includes source-path labels for QA Copilot questions. These are not interchangeable with indexed chunk IDs.

## Retrieval Methods
1. Semantic search using all-MiniLM-L6-v2 and ChromaDB.
2. BM25 lexical retrieval.
3. Reciprocal Rank Fusion of semantic and BM25 retrieval.
4. Token-bounded embedding windows.
5. Weighted fusion of original chunks and windows.

## Metrics
- Precision@5: Fraction of the first five retrieved chunks that are relevant.
- Recall@5 and Recall@10: Fraction of labelled relevant chunks retrieved within the first K results.
- MRR@5: Reciprocal rank of the first relevant result, averaged across answerable questions.
- Top-1 accuracy: Fraction of questions with a relevant result at Rank 1.
- p50/p95 latency: Median and 95th-percentile warm local retrieval time.

## Results

| Method | Precision@5 | Recall@5 | MRR@5 | Top-1 |
|---|---:|---:|---:|---:|
| Semantic | 0.252 | 0.891 | 0.812 | 0.696 |
| BM25 | 0.191 | 0.696 | 0.686 | 0.609 |
| Hybrid RRF | 0.217 | 0.761 | 0.755 | 0.652 |
| Token windows | 0.252 | 0.880 | 0.797 | 0.652 |
| Weighted RRF | 0.252 | 0.891 | 0.809 | 0.696 |

Semantic Recall@10: 0.957.

Warm local retrieval latency, 115 measurements:
- p50: 4.71 ms
- p95: 9.66 ms

These timings exclude LLM generation, network requests and frontend rendering.

## Key Findings
- Semantic retrieval is the strongest overall method on this dataset.
- Eleven of 32 indexed chunks exceed the embedding model's 256-token limit.
- Token windows improve some citation-related queries but reduce overall benchmark performance.
- Weighted fusion does not outperform the original semantic baseline.

## Limitations
- Small benchmark derived primarily from a single repository.
- Labels and questions may be influenced by familiarity with the source code.
- Extended three-repository tracker has not been evaluated as one retrieval benchmark.
- Out-of-scope refusal accuracy and final answer correctness were not measured.
- Warm local latency does not represent end-to-end production latency.

## Decision
Retain semantic retrieval as the measured baseline.

Investigate token-aware chunking and validate improvements on held-out questions before modifying production retrieval.

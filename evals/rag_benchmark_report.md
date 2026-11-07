# Chat-with-Your-Repo — Retrieval Benchmark

## Scope and methodology

- Corpus: 32 backend chunks in `rag_backend_eval`.
- Questions: 23 answerable and 2 deliberately out of scope.
- Model: `all-MiniLM-L6-v2`. Evaluation depth: top-5; candidate depth: 15.
- Gold labels are provisional chunk-level relevance judgments from the uploaded manifest, not independently adjudicated annotations.
- The RRF configuration is unweighted with constant 60. BM25 uses a simple lowercase, identifier-preserving tokenizer.
- Queries marked out of scope are **excluded from retrieval metrics**. The current retrievers cannot determine when to abstain; LLM answer abstention requires a separate experiment.
- Index fingerprint (SHA-256): `07ea544807b53b9c384eb2f0a9ad59cdebee060aeead0b1943956a8a5bb55609`.

## Scores on answerable questions

| Method | Precision@5 | Recall@5 | MRR@5 | Top-1 accuracy |
|---|---:|---:|---:|---:|
| Semantic | 0.252 | 0.891 | 0.812 | 0.696 |
| BM25 | 0.191 | 0.696 | 0.686 | 0.609 |
| Hybrid_RRF | 0.217 | 0.761 | 0.755 | 0.652 |

Precision@5 uses five as its denominator for every question. Most questions have only one or two labelled relevant chunks, so low values can be a consequence of the annotation design rather than poor retrieval.

## Questions needing attention

| ID | Question | Relevant chunks | Semantic RR | BM25 RR | Hybrid RR |
|---|---|---|---:|---:|---:|
| R01 | What embedding model, ChromaDB directory, and collection name are configured by the backend? | Backend/embed.py:1 | 0.25 | 0.00 | 0.00 |
| R05 | How are non-Python documents chunked, and what is the default number of lines per chunk? | Backend/ingest.py:84, Backend/ingest.py:68 | 1.00 | 0.00 | 0.33 |
| R06 | How is a Python file with no top-level functions or classes represented in the index? | Backend/ingest.py:13 | 0.50 | 1.00 | 1.00 |
| R07 | How does get_chunks respond when the requested file does not exist? | Backend/ingest.py:84 | 1.00 | 0.25 | 1.00 |
| R08 | Which folders and file extensions does repository ingestion skip or accept? | Backend/embed_repo.py:1, Backend/embed_repo.py:17 | 1.00 | 0.20 | 0.50 |
| R11 | How is the temporary cloned repository removed after embedding finishes or raises an error? | Backend/embed_repo.py:103 | 0.50 | 0.00 | 0.00 |
| R12 | Trace the code path from a GitHub URL to embedded searchable chunks stored in ChromaDB. | Backend/embed_repo.py:103, Backend/embed_repo.py:80, Backend/embed_repo.py:17, Backend/embed.py:32 | 0.50 | 1.00 | 1.00 |
| R16 | How is a GitHub blob URL with a source line number built from retrieved chunk metadata? | Backend/embed.py:86 | 1.00 | 0.33 | 1.00 |
| R17 | How does the ask.py CLI filter retrieval to one repository when --repo-url is supplied? | Backend/ask.py:11 | 1.00 | 0.50 | 1.00 |
| R18 | How does ask.py encode the user's question and obtain its top-k matching ChromaDB chunks? | Backend/ask.py:11 | 0.33 | 1.00 | 0.50 |
| R19 | How are retrieved chunks given source labels and then sent to Groq as a streamed response? | Backend/ask.py:11, Backend/query.py:22 | 0.33 | 1.00 | 0.50 |
| R21 | How does the backend obtain GROQ_API_KEY, and what happens if it is missing? | Backend/query.py:55 | 1.00 | 0.50 | 1.00 |
| R22 | How are Groq API exceptions handled during streaming in the command-line application? | Backend/query.py:76, Backend/query.py:22 | 1.00 | 0.00 | 0.20 |
| R23 | How are the final source listings constructed from retrieval metadata instead of from the language model's citations? | Backend/ask.py:11, Backend/embed.py:86 | 0.25 | 1.00 | 0.33 |

## Out-of-scope questions (diagnostic only)

- **R24**: Which React component contains the search form submit handler in the frontend?
  - Why out of scope: No frontend component is indexed in this backend-only manifest.
  - Semantic top result: `Backend/test_decorator.py:20`. This is a nearest neighbour, **not proof of an answer**.
- **R25**: What exact HTTP request timeout in seconds is configured for the FastAPI /ask endpoint?
  - Why out of scope: No current FastAPI main.py implementation or timeout configuration is indexed.
  - Semantic top result: `Backend/NOTES.md:151`. This is a nearest neighbour, **not proof of an answer**.

## Interpretation notes

- Do not treat a perfect score as general AI answer accuracy: this only tests retrieval.
- Historical `Backend/NOTES.md` may mention superseded behavior. Current executable code is authoritative for current-implementation questions.
- `main.py`, frontend files and any unindexed extensions are outside this corpus.
- Verify partial relevance and annotation completeness before publishing this benchmark as portfolio-level evidence.
- Compare changes against a held-out benchmark instead of tuning RRF to these same questions.

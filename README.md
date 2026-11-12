# Chat with Your Repo

A RAG-powered tool that lets you ask natural-language questions about
a codebase and get grounded, cited answers — pointing to the exact
file and line the answer came from.




## Recorded Demo

https://github.com/user-attachments/assets/d50acb6d-7524-405c-8769-e92f8e65d33e


## What it does

Paste a GitHub repo URL. The app clones the repo, splits source files into chunks using function and class boundaries for Python, then embeds and indexes those chunks for retrieval.

Ask a question in plain English to get a streamed answer with inline citations and links to the GitHub files and lines the app retrieved.
## Stack

**Backend:** Python, FastAPI, ChromaDB (vector storage),
sentence-transformers (`all-MiniLM-L6-v2` embeddings), Groq API
(LLaMA 3.3 70B) for generation.

**Frontend:** React, TypeScript, Vite. Server-Sent Events for
token-by-token streaming, markdown rendering with syntax-highlighted
code blocks.

## How it works

1. **Ingestion** — `/embed_repo` clones a GitHub repo, walks its file
   tree, and chunks each file: for Python, chunks are split at
   function/class definitions using the AST, not fixed-size splits.
2. **Retrieval** — each chunk is embedded and stored in ChromaDB with
   metadata (relative file path, start line, repo URL, branch). A
   question is embedded the same way and matched against stored chunks
   via similarity search.
3. **Generation** — retrieved chunks are labeled and passed to Groq's
   LLaMA 3.3 70B, which is instructed to answer only from the provided
   chunks and to cite which source(s) it used.
4. **Citations** — sources returned alongside the answer are built
   directly from retrieval metadata, not parsed from the model's own
   text — so a citation can't silently point to the wrong file or line
   even if the model's inline reference is imprecise.

## Design decisions worth knowing about

- **Chunking is AST-aware.** Python files are split at function/class
  boundaries so a chunk is a coherent unit of code, not an arbitrary
  slice.
- **Citations are deterministic, not model-generated.** The sources
  list returned to the user is built from what was actually retrieved,
  independent of what the model claims — a deliberate safeguard
  against the model's inline `(Source N)` reference hallucinating a
  file/line even when retrieval itself was correct.
- **Repo-scoped retrieval.** Since one deployment can index multiple
  repos, queries can optionally be scoped to a single `repo_url` to
  avoid cross-repo retrieval noise.
- **Streaming end to end.** Both the Groq API call and the FastAPI
  endpoint stream token-by-token via SSE, rendered incrementally in
  the React frontend rather than waiting for a full response.

## Status

- ✅ Backend: chunking, retrieval, citations, streaming — verified end
  to end against multiple real repos (single-file and multi-file,
  including subfolder paths)
- ✅ Frontend: chat UI, live streaming, clickable GitHub citations,
  markdown rendering with syntax-highlighted code blocks
- ✅ Deployment — backend live on Google Cloud Run, frontend live on
  Vercel, fully connected and verified end to end (repo embedding,
  streaming answers, citations all tested against the live URLs, not
  just locally)
- ⚠️ **Known limitation (Cloud Run):** the container filesystem is
  ephemeral, so `chroma_db/` is wiped on every redeploy or cold start.
  Any repo you've embedded will need to be re-embedded after that.
  Accepted for this portfolio demo — not something a persistent
  volume/managed vector DB has been added to solve yet.

## Running locally

**Backend:**
```
cd Backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --port 8000
```

**Frontend:**
```
cd Frontend
npm install
npm run dev
```

Requires a `GROQ_API_KEY` in `Backend/.env`.

<!-- week2-evaluation-start -->

## Week 2 retrieval evaluation

### Original chunk-ID benchmark

The original benchmark used 32 chunks from eight backend files. The metrics below average the 23 answerable questions. Two additional questions were outside the indexed corpus.

| Retrieval method | Precision@5 | Recall@5 | MRR@5 |
|---|---:|---:|---:|
| Semantic | 0.252 | 0.891 | 0.812 |
| BM25 | 0.191 | 0.696 | 0.686 |
| Semantic + BM25 RRF | 0.217 | 0.761 | 0.755 |

### Source evidence on the experimental index

A separate frozen index uses token-aware structural chunks and records their source revision and line ranges. All 70 required evidence units are present across the index. A unit counts when the retrieved passages together cover its complete labeled source span. All methods below return up to five chunks for each of the 23 development questions.

| Retrieval method | Evidence units covered | Questions with all evidence | Decision |
|---|---:|---:|---|
| Semantic top five | 31/70 (44.3%) | 8/23 | Comparison baseline |
| BM25 top five | 32/70 (45.7%) | 7/23 | Keep experimental |
| RRF of semantic and BM25 top-five lists | 32/70 (45.7%) | 6/23 | Keep experimental |
| Cross-encoder reranking of semantic top 20 | 41/70 (58.6%) | 9/23 | Keep experimental |

Cross-encoder prediction added a median of 49 ms in the development run, excluding model loading and answer generation. One question requires at least nine chunks to cover all its evidence under this representation.

In a separate six-question pilot using the same `openai/gpt-oss-120b` generation settings for both methods, semantic retrieval produced two fully correct and complete answers out of four answerable cases. Reranking produced one. Both methods gave the expected refusal on both questions outside the corpus. On the branch-deletion question, the reranked answer incorrectly inferred that the repository URL scopes deletion to a branch. Keep the application retrieval baseline and retain the reranker as an experiment.

The original benchmark counts matching chunk IDs. The newer benchmark counts complete source spans and uses a different index. The six-case answer pilot uses one generation per method per question on a familiar development corpus. See the [answer-quality review](docs/WEEK2_ANSWER_QUALITY_REVIEW.md) for the scoring, timing scope, and source evidence.

<!-- week2-evaluation-end -->
## Week 5 observability and regression evidence

The `/ask` workflow is traced in Langfuse as a root `chat-repo-request`
observation with separate `retrieval` and `generation` steps. The trace
records repository scope, retrieved source previews, model, latency, output,
and technical failure category.

### Diagnosing a retrieval failure

The same question was run twice:

> How does the ask endpoint scope retrieval to a specific repository?

| Run | Repository scope | Retrieval result | Answer |
|---|---|---|---|
| Unscoped | `repo_url: null` | Mixed results from unrelated repositories and outdated notes | Incorrectly claimed that no repository filter existed |
| Scoped | `chat-with-your-repo` | `Backend/main.py` ranked first | Correctly explained the optional Chroma `where` filter |

The unscoped trace took `1.00s`: `0.15s` for retrieval and `0.85s` for
generation. The scoped trace took `0.96s`: `0.03s` for retrieval and `0.93s`
for generation. These are single local observations, not benchmark results.

![Unscoped retrieval trace](docs/images/langfuse-before-unscoped.png)

![Scoped retrieval trace](docs/images/langfuse-after-scoped.png)

The trace showed that the original failure began in retrieval. The model did
not receive the relevant `Backend/main.py` evidence, so generation relied on
irrelevant context. Once retrieval was scoped correctly, the model produced
the correct answer with source citations.

### Repository URL regression check

Repository URLs are canonicalized before creating the Chroma filter, so both
of these forms resolve to the same indexed repository:

```text
https://github.com/ujjwaldahiya399/chat-with-your-repo
https://github.com/ujjwaldahiya399/chat-with-your-repo.git


The first is the normal browser URL. The second is commonly used as a Git clone URL.
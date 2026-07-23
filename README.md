# Chat with Your Repo

A RAG-powered tool that lets you ask natural-language questions about
a codebase and get grounded, cited answers — pointing to the exact
file and line the answer came from.

## Demo


https://github.com/user-attachments/assets/d50acb6d-7524-405c-8769-e92f8e65d33e


## What it does

Paste a GitHub repo URL, and the app clones it, chunks it along
function/class boundaries (AST-aware, not blind character splits),
embeds each chunk, and indexes it for retrieval. Ask a question in
plain English and get a streamed, cited answer back — every claim in
the response is backed by a clickable link to the real file and line
on GitHub.

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
- 🚧 Deployment — not yet started

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

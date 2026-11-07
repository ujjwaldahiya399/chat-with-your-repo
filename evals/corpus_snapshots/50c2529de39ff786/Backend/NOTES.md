# NOTES

## Session Log — 2026-07-16

### Built

- `ingest.py` — AST-based chunking: splits a Python file into one chunk per
  top-level function/class, plus a leading module-level chunk. Falls back to
  fixed-size line chunking for non-Python files or syntax errors.
- `query.py` — Groq querying with grounding: sends a chunk + question to
  `llama-3.3-70b-versatile` with a system prompt restricting answers to the
  given chunk.
- `ask.py` — wires `ingest.py` and `query.py` together using a "concatenate
  everything" strategy: all chunks are joined into one blob and sent as a
  single query, rather than retrieved selectively.
- `embed.py` — local embeddings: chunks a file via `get_chunks`, embeds each
  chunk with `sentence-transformers` (`all-MiniLM-L6-v2`), and stores it in a
  persistent Chroma collection (`code_chunks`) with `file_path`/`start_line`
  metadata.

### Bugs found and fixed

**`ingest.py` — gap between definitions silently dropped content**
- Bug: a chunk for one definition stopped at that definition's own body,
  leaving the lines between it and the next definition (blank lines,
  comments) unassigned to any chunk.
- Found: via code review.
- Fix: extend each chunk's end line to the next definition's start line
  instead of the current definition's own end.
- Verified: re-chunked a real repo file and confirmed no lines were dropped
  between consecutive definitions.

**`ingest.py` — decorators excluded from their function's chunk**
- Bug: a decorated function's chunk started at `def`/`class`, leaving its
  decorator lines in the previous chunk instead of with the function they
  apply to.
- Found: via code review.
- Fix: added `_def_start_line`, a decorator-aware helper that returns the
  first decorator's line (if any) instead of the `def`/`class` line.
- Verified: constructed a test file with stacked decorators
  (`test_decorator.py`) and confirmed each decorator stayed grouped with its
  function/class in the resulting chunk.

**`ask.py` — empty-chunk guard didn't fire for empty `.py` files**
- Bug: `chunk_python_file` returns one chunk containing an empty string for
  an empty file (not an empty list), so a truthiness/length check on the
  chunk list didn't catch the empty-content case.
- Found: by deliberately testing an empty `.py` file.
- Fix: check whether the combined chunk content is empty/whitespace after
  stripping, not whether the chunk list itself is empty.
- Verified: forced an invalid API key and confirmed no Groq request was sent
  for the empty file (guard exits before `query()` is called).

**`embed.py` — chunk IDs went stale when a file's line numbers shifted**
- Bug: IDs are `file_path:start_line`. Editing a file (e.g. adding a line at
  the top) shifts every later definition's `start_line`, producing new IDs.
  Since `upsert` only touches the IDs it's given, the old IDs for that file
  were never removed, leaving stale duplicate chunks in the collection.
- Found: by deliberately editing a test file and re-embedding it.
- Verified: computed the expected chunk count by hand before re-running,
  then confirmed the actual collection count matched the stale-duplicate
  math (old + new chunk IDs coexisting) before the fix, and collapsed to
  just the current chunks after adding `collection.delete(where={"file_path":
  ...})` before the upsert.

**`embed.py` — same file produced different IDs depending on invocation directory**
- Bug: `file_path` was used raw (as typed on the command line), so the same
  physical file got different IDs depending on the caller's working
  directory (e.g. `test_decorator.py` vs. `backend/test_decorator.py`).
- Found: by running `embed.py` on the same file from two different working
  directories and comparing the resulting IDs.
- Fix: resolve `file_path` to its absolute, canonical form
  (`Path(file_path).resolve()`) immediately, before it's used for chunking,
  ID construction, the delete filter, or stored metadata.
- Verified: re-ran from both directories and confirmed the collection's
  total chunk count stayed constant rather than growing.

### Process note

Bugs were found by predicting the expected output before running the code
and comparing that prediction against the actual result, not by trusting
diff summaries or assuming a change worked.

## Phase 1 — Interview & Architecture Notes

### Why RAG instead of fine-tuning or full-file concatenation

`ask.py`'s original strategy ("Option A: send everything") joined every chunk
of a file into one blob and sent the whole thing as context on every call.
That works for a single small file but doesn't scale: context windows are
finite, cost/latency grow with input size, and the strategy breaks entirely
at repo scale (Phase 3) where the full content of every file can't fit in
one prompt regardless of window size. Fine-tuning was rejected too — it
bakes a snapshot of the code into model weights, going stale the moment the
repo changes, and requires retraining to pick up edits. RAG solves this by
separating storage from querying: chunks are embedded and indexed once, and
each question only pulls in the handful of chunks actually relevant to it,
so cost and latency stay roughly constant regardless of repo size.

### Embedding + retrieval pipeline, end to end

Question → embedded with the same `sentence-transformers` model
(`all-MiniLM-L6-v2`) used to embed the stored chunks → Chroma similarity
search against `code_chunks` → top-k nearest chunks → chunks joined and
passed into the same grounded Groq prompt `query.py` already used for
Option A. Using the *same* model for both chunks and questions matters
because embeddings from different models live in unrelated vector spaces —
distances between them wouldn't mean anything. Chroma isn't doing a
brute-force scan for this: it computes distance (cosine or L2, depending on
collection config) and uses HNSW (Hierarchical Navigable Small World), an
approximate-nearest-neighbor graph index, to avoid comparing the query
against every stored vector.

### Repo-wide vs. file-scoped retrieval

`collection.query()` in `ask.py`/`main.py` has no `where` filter on
`file_path`, so retrieval is repo-wide, not scoped to the file passed in.
This was a deliberate choice: the actual product goal (Phase 3+) is "chat
with your whole repo," not "chat with one file I already pointed you at" —
scoping to a single file now would mean ripping the filter back out later.
The tradeoff: a question aimed at one file can retrieve a more semantically
similar chunk from a completely different file, because Chroma only knows
about vector distance, not the caller's intent to stay within one file.

### TOP_K as a tunable parameter, not a fixed constant

`TOP_K` was made a CLI flag (`--top-k`, default 5) and an API field
(`top_k`, default 5) instead of staying a hardcoded constant, because the
right retrieval depth is a real tradeoff, not a fixed fact. Observed
directly: asking the same question against `ingest.py` with `top_k=3`
returned a shorter, surface-level answer, while `top_k=5` surfaced a detail
about `GroqError` exception handling that wasn't mentioned at all in the
`k=3` answer. That's concrete evidence of the recall/precision tradeoff —
fewer chunks means tighter, more confident answers but a higher chance of
missing relevant context; more chunks means better recall but more noise
(and more tokens) for the model to sort through.

### Why `ask.py` re-embeds the file on every call

Every `ask.py` (and `main.py` `/ask`) call re-runs `embed_file()` on the
target file before querying, rather than assuming the store is already
up to date. This is a deliberate freshness guarantee: it's the same
reasoning behind the Day 1 stale-ID fix (`collection.delete()` before
`upsert()`) — a store that's silently out of date is worse than a slower
one that's always correct. The known cost is latency: every `/ask` call
pays for a full re-chunk + re-embed of the file even if nothing changed
since the last call. Acceptable for Phase 1's single-file scale; worth
revisiting once retrieval is repo-wide over many files (Phase 3) where
re-embedding on every question stops being cheap.

### `SystemExit` handling in `main.py`

`get_chunks()`, `embed_file()`, and `safe_query()` were all written as CLI
scripts — on failure they print an error and call `sys.exit()`, which is
correct for a one-shot script but would kill the entire FastAPI process if
left uncaught in a request handler. `main.py` wraps each call in
`try/except SystemExit`, converting it into an `HTTPException` (400 for
missing-file/empty-content, 500 for a Groq failure) instead of letting it
propagate. This wasn't assumed safe just from reading the code — it was
verified with a real request: `POST /embed` with a nonexistent file
returned a clean `400` with the collection process still alive and serving
`/docs` afterward, confirming the process didn't go down.

### Known, deliberate tradeoffs in `main.py`

- `get_chunks()` runs twice per `/ask` request — once directly (to validate
  the file exists and return a clean 400 otherwise) and once again inside
  `embed_file()` (which re-chunks internally). This duplicates file parsing
  work but was chosen over modifying `embed.py` to return a chunk count,
  since the task constraint was to wrap the existing scripts, not change
  them.
- `safe_query()`'s `SystemExit` swallows the real Groq error — it only
  prints the exception before exiting, so by the time `main.py` catches the
  `SystemExit`, the original error text is gone. The `/ask` 500 response can
  say a Groq request failed but can't say why. Both are flagged here as
  accepted-for-now tradeoffs of wrapping unmodified CLI scripts, not bugs.

### Why `lifespan` instead of `@app.on_event`

`@app.on_event("startup")` is deprecated in current FastAPI; `lifespan` is
the supported replacement for run-once-at-startup logic. The
`SentenceTransformer` model, the Chroma client/collection, and the Groq API
key are all loaded once in `lifespan` and stashed on `app.state` rather than
being recreated per request — loading the transformer model alone takes
noticeable time, so doing it on every request would make every endpoint
call pay that cost instead of paying it once at process start.

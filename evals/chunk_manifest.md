# RAG Backend Evaluation — Chunk Manifest

Total indexed chunks: 32


============================================================
CHUNK ID: Backend/NOTES.md:1
FILE: Backend/NOTES.md
START LINE: 1
============================================================

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


============================================================
CHUNK ID: Backend/NOTES.md:101
FILE: Backend/NOTES.md
START LINE: 101
============================================================


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



============================================================
CHUNK ID: Backend/NOTES.md:151
FILE: Backend/NOTES.md
START LINE: 151
============================================================

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


============================================================
CHUNK ID: Backend/NOTES.md:51
FILE: Backend/NOTES.md
START LINE: 51
============================================================

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


============================================================
CHUNK ID: Backend/ask.py:1
FILE: Backend/ask.py
START LINE: 1
============================================================

import sys
import argparse

import chromadb
from sentence_transformers import SentenceTransformer

from embed import build_github_url, CHROMA_DB_PATH, COLLECTION_NAME, MODEL_NAME
from query import load_api_key, safe_query_stream




============================================================
CHUNK ID: Backend/ask.py:11
FILE: Backend/ask.py
START LINE: 11
============================================================

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("question")
    parser.add_argument("--top-k", type=int, default=5)
    # Optional — same $and-free single-condition `where` scoping as main.py's
    # /ask; omitted, this searches the whole collection (old local-file
    # chunks with no repo_url included) same as before this change.
    parser.add_argument("--repo-url", default=None)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # No more re-embedding here: ask.py used to take a single file_path and
    # re-embed it via embed_file() on every call, which made sense when
    # there was only ever one local file in play. Now that whole repos are
    # pre-embedded via embed_repo_from_url() (and single files via
    # embed_file() directly), there's no single file to re-embed — ask.py
    # is now a pure query against whatever's already in the collection,
    # matching main.py's /ask.
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    question_embedding = model.encode([args.question]).tolist()
    where = {"repo_url": args.repo_url} if args.repo_url else None
    # Capture the full query result (not just ["documents"][0]) so metadata
    # (file_path/start_line) can be read from the same call instead of a second query.
    query_result = collection.query(
        query_embeddings=question_embedding, n_results=args.top_k, where=where
    )
    retrieved_chunks = query_result["documents"][0]
    retrieved_metadatas = query_result["metadatas"][0]

    if not retrieved_chunks:
        print("Error: no chunks found in the collection", file=sys.stderr)
        sys.exit(1)

    # Prefix each chunk with a "[Source N: file:line]" header (1-indexed, matching
    # retrieval order) so the model can cite "(Source N)" and we can print a
    # source list afterward that lines up with what was actually retrieved.
    labeled_chunks = [
        f"[Source {i}: {meta['file_path']}:{meta['start_line']}]\n{content}"
        for i, (content, meta) in enumerate(zip(retrieved_chunks, retrieved_metadatas), start=1)
    ]
    combined_chunk = "\n\n".join(labeled_chunks)

    api_key = load_api_key()

    # Print tokens as they arrive instead of waiting for the full answer —
    # end=""/flush=True makes each token appear in the terminal immediately.
    for token in safe_query_stream(combined_chunk, args.question, api_key):
        print(token, end="", flush=True)
    print()

    # Sources are NOT streamed — retrieval already finished before the model
    # said a word, and this list is built directly from retrieved_metadatas,
    # not parsed from the model's answer, since an inline "(Source N)"
    # citation could hallucinate a file/line even when retrieval was correct.
    print("\nSources:")
    for i, meta in enumerate(retrieved_metadatas, start=1):
        line = f"  [{i}] {meta['file_path']}:{meta['start_line']}"
        github_url = build_github_url(meta)
        if github_url:
            line += f"  ({github_url})"
        print(line)


============================================================
CHUNK ID: Backend/embed.py:1
FILE: Backend/embed.py
START LINE: 1
============================================================

import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from ingest import get_chunks

MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DB_PATH = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "code_chunks"




============================================================
CHUNK ID: Backend/embed.py:14
FILE: Backend/embed.py
START LINE: 14
============================================================

def _real_case_path(path):
    """Resolve to the on-disk casing of every path component.

    macOS/Windows filesystems are case-insensitive but case-preserving, so
    Path.resolve() alone doesn't fix typed-in casing: "backend/x.py" and
    "Backend/x.py" resolve to two different strings for the same file,
    which produces two different chunk IDs (ids are f"{file_path}:{line}")
    for what should be one entry. Walking the real directory listing and
    swapping in the actual entry name per component makes the same file
    always map to the same file_path, regardless of how it was typed.
    """
    resolved = Path(path).resolve()
    real = Path(resolved.anchor)
    for part in resolved.parts[1:]:
        real = real / next((e.name for e in real.iterdir() if e.name.lower() == part.lower()), part)
    return real




============================================================
CHUNK ID: Backend/embed.py:32
FILE: Backend/embed.py
START LINE: 32
============================================================

def embed_file(file_path, model=None, *, repo_root=None, repo_url=None, branch=None):
    """Chunk a file, embed each chunk, and upsert it into the Chroma collection.

    repo_root/repo_url/branch are optional and only passed by repo-clone
    ingestion (embed_repo_from_url). When given, the stored file_path is
    relative to repo_root instead of absolute — a relative path survives
    the repo being re-cloned into a different temp dir next time, and it's
    what a GitHub URL needs (github.com/owner/repo/blob/branch/<relative
    path>). repo_url/branch are attached to every chunk's own metadata
    rather than kept as a module-level constant, because one Chroma
    collection can hold chunks from many repos at once — a single global
    value would mislabel every repo but the last one embedded.
    """
    if repo_root is not None:
        stored_path = str(Path(file_path).resolve().relative_to(Path(repo_root).resolve()))
    else:
        file_path = str(_real_case_path(file_path))
        stored_path = file_path

    chunks = get_chunks(file_path)
    combined_chunk = "\n".join(content for _, content in chunks)

    if not combined_chunk.strip():
        print(f"Error: no content to embed in {stored_path}", file=sys.stderr)
        sys.exit(1)

    model = model or SentenceTransformer(MODEL_NAME)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    # repo_url/branch prefix keeps IDs unique across repos that happen to
    # share a relative path (e.g. every repo has a README.md).
    id_prefix = f"{repo_url}@{branch}:" if repo_url else ""
    ids = [f"{id_prefix}{stored_path}:{start_line}" for start_line, _ in chunks]
    documents = [content for _, content in chunks]
    metadata_base = {"file_path": stored_path}
    if repo_url:
        metadata_base["repo_url"] = repo_url
        metadata_base["branch"] = branch
    metadatas = [{**metadata_base, "start_line": start_line} for start_line, _ in chunks]
    embeddings = model.encode(documents).tolist()

    # Scope the dedup delete by repo too, or two repos sharing a relative
    # path (e.g. both have a README.md) would delete each other's chunks.
    delete_where = {"file_path": stored_path}
    if repo_url:
        delete_where = {"$and": [{"file_path": stored_path}, {"repo_url": repo_url}]}
    collection.delete(where=delete_where)
    collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)

    print(f"Embedded and stored {len(chunks)} chunks from {stored_path}.")
    print(f"Collection '{COLLECTION_NAME}' now has {collection.count()} chunks total.")




============================================================
CHUNK ID: Backend/embed.py:86
FILE: Backend/embed.py
START LINE: 86
============================================================

def build_github_url(meta):
    """Build a clickable GitHub URL from chunk metadata, or None if this
    chunk wasn't embedded from a repo clone (repo_url/branch absent — e.g.
    a plain local file via /embed, which has no browsable remote to link to).
    """
    if not meta.get("repo_url"):
        return None
    return f"{meta['repo_url']}/blob/{meta['branch']}/{meta['file_path']}#L{meta['start_line']}"


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python embed.py <file_path>", file=sys.stderr)
        sys.exit(1)

    embed_file(sys.argv[1])


============================================================
CHUNK ID: Backend/embed_repo.py:1
FILE: Backend/embed_repo.py
START LINE: 1
============================================================

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import chromadb

from embed import embed_file, CHROMA_DB_PATH, COLLECTION_NAME

SKIP_DIRS = {".git", "venv", "__pycache__", "node_modules", "chroma_db"}
ALLOWED_EXTENSIONS = {".py", ".txt", ".md"}




============================================================
CHUNK ID: Backend/embed_repo.py:103
FILE: Backend/embed_repo.py
START LINE: 103
============================================================

def embed_repo_from_url(github_url, branch="main"):
    """Clone a GitHub repo and embed every file in it, tagging each chunk
    with repo_url/branch so retrieval can build a real GitHub citation link.
    """
    dest, repo_url = clone_repo(github_url, branch=branch)
    try:
        return embed_repo(str(dest), repo_root=dest, repo_url=repo_url, branch=branch)
    finally:
        shutil.rmtree(dest, ignore_errors=True)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python embed_repo.py <folder_path>", file=sys.stderr)
        sys.exit(1)

    folder_arg = sys.argv[1]
    folder = Path(folder_arg)

    if not folder.exists():
        print(f"Error: Folder not found: {folder_arg}", file=sys.stderr)
        sys.exit(1)

    if not folder.is_dir():
        print(f"Error: Not a directory: {folder_arg}", file=sys.stderr)
        sys.exit(1)

    embed_repo(folder_arg)


============================================================
CHUNK ID: Backend/embed_repo.py:17
FILE: Backend/embed_repo.py
START LINE: 17
============================================================

def embed_repo(folder_path, *, repo_root=None, repo_url=None, branch=None):
    """Walk a folder and embed every relevant file in it, skipping the rest.

    repo_root/repo_url/branch are passed straight through to embed_file() —
    see its docstring. Left unset (the plain local-folder CLI usage below),
    behavior is unchanged from before this file supported repo-URL ingestion.
    """
    dirs_skipped = 0
    files_walked = 0
    files_skipped_extension = 0
    files_embedded = 0
    files_failed = 0

    for dirpath, dirnames, filenames in os.walk(folder_path, topdown=True):
        keep = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        dirs_skipped += len(dirnames) - len(keep)
        dirnames[:] = keep

        for filename in filenames:
            files_walked += 1
            file_path = os.path.join(dirpath, filename)

            if Path(filename).suffix not in ALLOWED_EXTENSIONS:
                files_skipped_extension += 1
                continue

            try:
                embed_file(file_path, repo_root=repo_root, repo_url=repo_url, branch=branch)
                files_embedded += 1
            except SystemExit:
                print(f"Warning: skipped {file_path} (no content to embed)", file=sys.stderr)
                files_failed += 1
            except Exception as e:
                print(f"Warning: failed to embed {file_path}: {e}", file=sys.stderr)
                files_failed += 1

    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    print()
    print(f"Files walked: {files_walked}")
    print(f"Skipped: {dirs_skipped} directories, {files_skipped_extension} files (extension)")
    print(f"Embedded: {files_embedded}")
    print(f"Failed: {files_failed}")
    print(f"Collection '{COLLECTION_NAME}' total chunk count: {collection.count()}")

    return files_embedded, collection.count()




============================================================
CHUNK ID: Backend/embed_repo.py:66
FILE: Backend/embed_repo.py
START LINE: 66
============================================================

def parse_github_url(url):
    """Extract (owner, repo, canonical_https_url) from a github.com HTTPS URL.

    Only the https://github.com/<owner>/<repo>[.git][/] form is supported —
    that's the form the citation links need anyway (a browsable URL), so
    there's no reason to also accept ssh git@github.com:... URLs here.
    """
    match = re.match(r"https://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", url.strip())
    if not match:
        raise ValueError(f"Not a github.com HTTPS URL: {url}")
    owner, repo = match.group(1), match.group(2)
    return owner, repo, f"https://github.com/{owner}/{repo}"




============================================================
CHUNK ID: Backend/embed_repo.py:80
FILE: Backend/embed_repo.py
START LINE: 80
============================================================

def clone_repo(url, branch="main"):
    """Shallow-clone a GitHub repo into a fresh temp dir.

    Shells out to the system `git` binary via subprocess rather than using
    GitPython: git is already a hard requirement for this project (it's a
    git repo), so this adds zero new dependencies, where GitPython would be
    a new package for something `git clone` already does in one line.
    --depth 1 skips full history since only one commit's file tree is
    needed for embedding.
    """
    owner, repo, repo_url = parse_github_url(url)
    dest = Path(tempfile.mkdtemp(prefix=f"{repo}-"))
    result = subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", branch, url, str(dest)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        shutil.rmtree(dest, ignore_errors=True)
        raise RuntimeError(f"git clone failed: {result.stderr.strip()}")
    return dest, repo_url




============================================================
CHUNK ID: Backend/ingest.py:1
FILE: Backend/ingest.py
START LINE: 1
============================================================

import sys
import ast
from pathlib import Path




============================================================
CHUNK ID: Backend/ingest.py:13
FILE: Backend/ingest.py
START LINE: 13
============================================================

def chunk_python_file(file_path):
    """Split a Python file into chunks by function/class boundary."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.splitlines(keepends=True)

    try:
        tree = ast.parse(content, filename=file_path)
    except SyntaxError as e:
        print(f"Error parsing {file_path}: {e}", file=sys.stderr)
        return chunk_fixed_size(file_path, lines)

    chunks = []
    definitions = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions.append(node)

    if not definitions:
        # No functions or classes, return the whole file as one chunk
        return [(1, ''.join(lines))]

    # Sort by line number (should already be sorted)
    definitions.sort(key=lambda n: n.lineno)

    # Add module-level code before first definition
    first_def_line = _def_start_line(definitions[0])
    if first_def_line > 1:
        chunk_lines = lines[0:first_def_line - 1]
        chunk_content = ''.join(chunk_lines)
        chunks.append((1, chunk_content))

    # Create chunks for each definition, including gap to next definition
    for i, definition in enumerate(definitions):
        # Check if definition has decorators; start at first decorator if present
        start_line_1indexed = _def_start_line(definition)
        start_line = start_line_1indexed - 1  # Convert to 0-indexed

        # Determine end line: extend to next definition or end of file
        if i < len(definitions) - 1:
            # Include everything up to (but not including) the next definition
            next_def_line = _def_start_line(definitions[i + 1])
            end_line = next_def_line - 1
        else:
            # Last definition: include to end of file
            end_line = len(lines)

        chunk_lines = lines[start_line:end_line]
        chunk_content = ''.join(chunk_lines)
        chunks.append((start_line_1indexed, chunk_content))

    return chunks




============================================================
CHUNK ID: Backend/ingest.py:6
FILE: Backend/ingest.py
START LINE: 6
============================================================

def _def_start_line(definition):
    """Return the 1-indexed line where a definition (including decorators) starts."""
    if definition.decorator_list:
        return definition.decorator_list[0].lineno
    return definition.lineno




============================================================
CHUNK ID: Backend/ingest.py:68
FILE: Backend/ingest.py
START LINE: 68
============================================================

def chunk_fixed_size(file_path, lines=None, chunk_size=50):
    """Split a file into fixed-size chunks."""
    if lines is None:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

    chunks = []
    for i in range(0, len(lines), chunk_size):
        start_line = i + 1  # Line numbers are 1-indexed for display
        chunk_lines = lines[i:i + chunk_size]
        chunk_content = ''.join(chunk_lines)
        chunks.append((start_line, chunk_content))

    return chunks




============================================================
CHUNK ID: Backend/ingest.py:84
FILE: Backend/ingest.py
START LINE: 84
============================================================

def get_chunks(file_path):
    """Validate the file exists and split it into chunks."""
    path = Path(file_path)

    if not path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    if path.suffix == '.py':
        return chunk_python_file(file_path)
    return chunk_fixed_size(file_path)




============================================================
CHUNK ID: Backend/ingest.py:97
FILE: Backend/ingest.py
START LINE: 97
============================================================

def ingest_file(file_path):
    """Ingest a file and split into chunks."""
    chunks = get_chunks(file_path)

    # Print chunks
    for start_line, content in chunks:
        print(f"=== Starting at line {start_line} ===")
        print(content, end='')
        print()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python ingest.py <file_path>", file=sys.stderr)
        sys.exit(1)

    ingest_file(sys.argv[1])


============================================================
CHUNK ID: Backend/query.py:1
FILE: Backend/query.py
START LINE: 1
============================================================

import os
import sys

from dotenv import load_dotenv
from groq import Groq, GroqError

# The chunk passed to this prompt is made of one or more labeled sources, each
# starting with a header like "[Source N: file_path:start_line]" (see ask.py /
# main.py). The model is told to cite that same N inline so answers stay
# traceable back to a specific chunk, while the "I don't know" fallback is kept
# verbatim so ungrounded questions still get a clean, exact refusal.
SYSTEM_PROMPT = (
    "Answer the question using only the information in the provided chunk of "
    "text. The chunk is made up of one or more sources, each starting with a "
    "header like \"[Source N: file_path:start_line]\". When your answer uses "
    "information from a source, cite it inline like \"(Source N)\", using the "
    "same N from that source's header. If the chunk does not contain the "
    "answer, respond with exactly: I don't know."
)




============================================================
CHUNK ID: Backend/query.py:22
FILE: Backend/query.py
START LINE: 22
============================================================

def query_stream(chunk, question, api_key):
    """Like query(), but yields the answer token by token as Groq streams it back.

    stream=True makes create() return an iterator of chunk events instead of
    one response; each event's delta.content is the next bit of text (or None
    on events that carry no new text, e.g. the final one), so we skip those.
    """
    client = Groq(api_key=api_key)

    stream = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Chunk:\n{chunk}\n\nQuestion:\n{question}"},
        ],
        stream=True,
    )

    for event in stream:
        token = event.choices[0].delta.content
        if token:
            yield token




============================================================
CHUNK ID: Backend/query.py:46
FILE: Backend/query.py
START LINE: 46
============================================================

def query(chunk, question, api_key):
    # Same external signature/behavior as before (still returns the full string) —
    # safe to rebuild on top of query_stream since the only difference is that the
    # underlying Groq request is now made in streaming mode instead of one-shot;
    # the final joined text is identical, and this way there's one place (above)
    # that builds the Groq request instead of two.
    return "".join(query_stream(chunk, question, api_key))




============================================================
CHUNK ID: Backend/query.py:55
FILE: Backend/query.py
START LINE: 55
============================================================

def load_api_key():
    """Load GROQ_API_KEY from .env, or print an error and exit."""
    load_dotenv()
    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        print("Error: GROQ_API_KEY is missing or empty. Set it in your .env file.", file=sys.stderr)
        sys.exit(1)

    return api_key




============================================================
CHUNK ID: Backend/query.py:67
FILE: Backend/query.py
START LINE: 67
============================================================

def safe_query(chunk, question, api_key):
    """Call query(), printing a clean error and exiting if the Groq request fails."""
    try:
        return query(chunk, question, api_key)
    except GroqError as e:
        print(f"Error: request to Groq failed: {e}", file=sys.stderr)
        sys.exit(1)




============================================================
CHUNK ID: Backend/query.py:76
FILE: Backend/query.py
START LINE: 76
============================================================

def safe_query_stream(chunk, question, api_key):
    """Like safe_query, but for query_stream: yields tokens, printing a clean
    error and exiting if the Groq request fails. CLI-only (see ask.py) — a
    long-running server must not let sys.exit reach it, so main.py calls
    query_stream directly instead of this."""
    try:
        yield from query_stream(chunk, question, api_key)
    except GroqError as e:
        print(f"\nError: request to Groq failed: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python query.py <chunk> <question>", file=sys.stderr)
        sys.exit(1)

    api_key = load_api_key()
    chunk_arg, question_arg = sys.argv[1], sys.argv[2]

    print(safe_query(chunk_arg, question_arg, api_key))


============================================================
CHUNK ID: Backend/requirements.txt:1
FILE: Backend/requirements.txt
START LINE: 1
============================================================

--extra-index-url https://download.pytorch.org/whl/cpu
torch==2.13.0
groq==1.5.0
python-dotenv==1.2.2
sentence-transformers==5.6.0
chromadb==1.5.9
fastapi==0.139.2
uvicorn==0.51.0
pydantic==2.13.4

============================================================
CHUNK ID: Backend/test_decorator.py:1
FILE: Backend/test_decorator.py
START LINE: 1
============================================================

# another new line
# a new comment line
import functools


CONSTANT = 42




============================================================
CHUNK ID: Backend/test_decorator.py:14
FILE: Backend/test_decorator.py
START LINE: 14
============================================================

@app.route("/login")
@require_auth
def login():
    return "ok"




============================================================
CHUNK ID: Backend/test_decorator.py:20
FILE: Backend/test_decorator.py
START LINE: 20
============================================================

class Handler:
    pass


============================================================
CHUNK ID: Backend/test_decorator.py:9
FILE: Backend/test_decorator.py
START LINE: 9
============================================================

@functools.lru_cache
def cached_func():
    return 1

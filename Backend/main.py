import json
import sys
from contextlib import asynccontextmanager

import chromadb
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from groq import GroqError
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from ingest import get_chunks
from embed import embed_file, build_github_url, CHROMA_DB_PATH, COLLECTION_NAME, MODEL_NAME
from embed_repo import embed_repo_from_url
from query import load_api_key, query_stream


class EmbedRequest(BaseModel):
    file_path: str


class EmbedResponse(BaseModel):
    chunks_embedded: int
    total_chunks_in_collection: int


class EmbedRepoRequest(BaseModel):
    github_url: str
    branch: str = "main"


class EmbedRepoResponse(BaseModel):
    files_embedded: int
    total_chunks_in_collection: int


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    # Optional — scopes retrieval to one repo's chunks via a `where` filter
    # instead of searching the whole collection. Chunks embedded via the
    # old local-file /embed path have no repo_url in their metadata at
    # all, so omitting this still searches everything, old and new chunks
    # alike — it's additive scoping, not a required tag on every chunk.
    repo_url: str | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = SentenceTransformer(MODEL_NAME)
    app.state.client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    app.state.collection = app.state.client.get_or_create_collection(COLLECTION_NAME)
    app.state.api_key = load_api_key()
    yield


app = FastAPI(lifespan=lifespan)

# ponytail: wide open for local dev, lock to the deployed frontend origin before shipping
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/embed", response_model=EmbedResponse)
def embed(body: EmbedRequest, request: Request):
    try:
        chunks = get_chunks(body.file_path)
    except SystemExit:
        raise HTTPException(status_code=400, detail=f"File not found: {body.file_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read file: {e}")

    try:
        embed_file(body.file_path, model=request.app.state.model)
    except SystemExit:
        raise HTTPException(status_code=400, detail=f"No content to embed in {body.file_path}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding failed: {e}")

    total = request.app.state.collection.count()
    return EmbedResponse(chunks_embedded=len(chunks), total_chunks_in_collection=total)


@app.post("/embed_repo", response_model=EmbedRepoResponse)
def embed_repo_endpoint(body: EmbedRepoRequest):
    # ponytail: embed_repo() (unchanged) doesn't thread a model through its
    # per-file embed_file() calls, so each file reloads SentenceTransformer
    # from scratch — same pre-existing behavior as the local-folder CLI path,
    # not something introduced here. Worth fixing if repo embedding is slow
    # enough to matter; out of scope for this metadata/URL-plumbing change.
    try:
        files_embedded, total = embed_repo_from_url(body.github_url, branch=body.branch)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        # clone_repo() raises this on a failed `git clone` (bad URL, bad
        # branch, private repo, etc.) — a client input problem, so 400, not
        # 500. The raw message embeds git's stderr, which includes the
        # server-side temp clone dir path — real for our own debugging,
        # but not something to hand back to whoever's calling this API.
        print(f"embed_repo failed for {body.github_url}@{body.branch}: {e}", file=sys.stderr)
        raise HTTPException(
            status_code=400,
            detail="Repository not found or not accessible. Check the URL and branch name.",
        )
    return EmbedRepoResponse(files_embedded=files_embedded, total_chunks_in_collection=total)


@app.post("/ask")
def ask(body: AskRequest, request: Request):
    # No response_model here — a StreamingResponse's body is built by the
    # generator below, not a single Pydantic object FastAPI can validate.
    #
    # No re-embedding here anymore. /ask used to take a single file_path,
    # validate it with get_chunks(), and re-embed it via embed_file() on
    # every call — that made sense when there was only ever one local file
    # in play. Now that whole repos are pre-embedded via /embed_repo (and
    # single files via /embed), there's no single file_path to re-embed,
    # and re-cloning+re-embedding a whole repo per question would be
    # wildly wasteful. /ask is now read-only against the collection —
    # writes happen exclusively through /embed and /embed_repo.
    question_embedding = request.app.state.model.encode([body.question]).tolist()

    # Only build a `where` filter when repo_url was given. Unlike embed.py's
    # dedup delete — which always combines file_path + repo_url and needs
    # $and for that — there's only ever one possible condition here, so a
    # plain single-key dict is the right-sized version of that same
    # pattern; wrapping one clause in $and would just be noise.
    where = {"repo_url": body.repo_url} if body.repo_url else None

    # Capture the full query result so metadata (file_path/start_line) is
    # available alongside the chunk text, same as ask.py.
    query_result = request.app.state.collection.query(
        query_embeddings=question_embedding, n_results=body.top_k, where=where
    )
    retrieved_chunks = query_result["documents"][0]
    retrieved_metadatas = query_result["metadatas"][0]

    if not retrieved_chunks:
        raise HTTPException(status_code=400, detail="No chunks found in the collection")

    # Label each chunk with "[Source N: file:line]" — query.py's SYSTEM_PROMPT
    # now always describes this format, so /ask must build it the same way
    # ask.py does, or the model would be told to expect headers that aren't there.
    labeled_chunks = [
        f"[Source {i}: {meta['file_path']}:{meta['start_line']}]\n{content}"
        for i, (content, meta) in enumerate(zip(retrieved_chunks, retrieved_metadatas), start=1)
    ]
    combined_chunk = "\n\n".join(labeled_chunks)
    api_key = request.app.state.api_key

    def event_stream():
        # SSE format: each event is "event: <name>\ndata: <json>\n\n" — the name
        # lets a client (EventSource / fetch+ReadableStream) tell token events
        # apart from the one final sources event without parsing the text.
        #
        # query_stream is used directly here, NOT safe_query_stream/safe_query:
        # those call sys.exit() on a Groq failure, which is fine for the ask.py
        # CLI but would try to kill the whole server process if it ever ran
        # inside a request handler. A live server has to turn the failure into
        # an SSE event instead, so the GroqError is caught right here.
        try:
            for token in query_stream(combined_chunk, body.question, api_key):
                yield f"event: token\ndata: {json.dumps(token)}\n\n"
        except GroqError as e:
            yield f"event: error\ndata: {json.dumps(str(e))}\n\n"
            return

        # Sources are sent once, after the token stream ends — not interleaved
        # with tokens — since retrieval already finished before the model
        # started answering, and this list is ground truth from
        # retrieved_metadatas, not parsed from the model's own citations.
        sources = [
            {
                "file_path": meta["file_path"],
                "start_line": meta["start_line"],
                # None for chunks embedded via plain /embed (no repo_url on
                # the metadata) — not every source has a browsable remote.
                "github_url": build_github_url(meta),
            }
            for meta in retrieved_metadatas
        ]
        yield f"event: sources\ndata: {json.dumps(sources)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

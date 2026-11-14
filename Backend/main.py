import json
import os
import re
import sys
from contextlib import asynccontextmanager
from langfuse import get_client
import chromadb
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from groq import GroqError
from pydantic import BaseModel
from embedding_model import EmbeddingModel

from ingest import get_chunks
from embed import embed_file, build_github_url, CHROMA_DB_PATH, COLLECTION_NAME
from embed_repo import embed_repo_from_url, parse_github_url
from demo_seed import seed_collection
from query import load_api_key, query_stream, MODEL_ID


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
    app.state.model = EmbeddingModel()
    app.state.client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    app.state.collection = app.state.client.get_or_create_collection(
        COLLECTION_NAME
    )
    seed_collection(app.state.collection)
    app.state.api_key = load_api_key()
    app.state.langfuse = get_client()

    yield

    app.state.langfuse.flush()

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.environ.get("FRONTEND_URL", "http://localhost:5173")],
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
def embed_repo_endpoint(body: EmbedRepoRequest, request: Request):
    try:
        files_embedded, total = embed_repo_from_url(
            body.github_url,
            branch=body.branch,
            model=request.app.state.model,
        )
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
    normalized_repo_url = None

    if body.repo_url:
        try:
            _, _, normalized_repo_url = parse_github_url(body.repo_url)
        except ValueError as error:
            raise HTTPException(status_code=400, detail=str(error)) from error
    langfuse = request.app.state.langfuse

    root = langfuse.start_observation(
        name="chat-repo-request",
        as_type="chain",
        input={
            "question": body.question,
            "top_k": body.top_k,
            "requested_repo_url": body.repo_url,
            "normalized_repo_url": normalized_repo_url,
        },
        metadata={"failure_category": "none"},
    )

    retrieval = root.start_observation(
        name="retrieval",
        as_type="retriever",
        input={
            "question": body.question,
            "top_k": body.top_k,
            "requested_repo_url": body.repo_url,
            "normalized_repo_url": normalized_repo_url,
        },
    )

    try:
        question_embedding = request.app.state.model.encode(
            [body.question]
        ).tolist()

        where = (
            {"repo_url": normalized_repo_url}
            if normalized_repo_url
            else None
        )


        query_result = request.app.state.collection.query(
            query_embeddings=question_embedding,
            n_results=body.top_k,
            where=where,
        )

        retrieved_chunks = query_result["documents"][0]
        retrieved_metadatas = query_result["metadatas"][0]

        retrieval.update(
            output={
                "source_count": len(retrieved_chunks),
                "sources": [
                    {
                        "source_number": index,
                        "file_path": metadata["file_path"],
                        "start_line": metadata["start_line"],
                        "content_preview": content[:1000],
                    }
                    for index, (content, metadata) in enumerate(
                        zip(retrieved_chunks, retrieved_metadatas),
                        start=1,
                    )
                ],
            }
        )

    except Exception as error:
        retrieval.update(
            level="ERROR",
            status_message=str(error),
        )
        retrieval.end()

        root.update(
            level="ERROR",
            status_message=str(error),
            output={"failure_category": "retrieval_error"},
            metadata={"failure_category": "retrieval_error"},
        )
        root.end()
        langfuse.flush()
        raise

    retrieval.end()

    if not retrieved_chunks:
        root.update(
            level="WARNING",
            status_message="No chunks found in the collection.",
            output={"failure_category": "retrieval_empty"},
            metadata={"failure_category": "retrieval_empty"},
        )
        root.end()
        langfuse.flush()

        raise HTTPException(
            status_code=400,
            detail="No chunks found in the collection",
        )

    labeled_chunks = [
        (
            f"[Source {index}: "
            f"{metadata['file_path']}:{metadata['start_line']}]\n"
            f"{content}"
        )
        for index, (content, metadata) in enumerate(
            zip(retrieved_chunks, retrieved_metadatas),
            start=1,
        )
    ]
    combined_chunk = "\n\n".join(labeled_chunks)
    api_key = request.app.state.api_key

    sources = [
        {
            "file_path": metadata["file_path"],
            "start_line": metadata["start_line"],
            "github_url": build_github_url(metadata),
        }
        for metadata in retrieved_metadatas
    ]

    def event_stream():
        generation = root.start_observation(
            name="generation",
            as_type="generation",
            model=MODEL_ID,
            model_parameters={"stream": True},
            input={
                "question": body.question,
                "context": combined_chunk,
            },
        )

        answer_parts = []

        try:
            for token in query_stream(
                combined_chunk,
                body.question,
                api_key,
            ):
                answer_parts.append(token)
                yield f"event: token\ndata: {json.dumps(token)}\n\n"

            answer = "".join(answer_parts)
            cited_numbers = {
                int(number)
                for number in re.findall(r"\bSource\s+(\d+)\b", answer)
            }
            invalid_citations = sorted(
                cited_numbers - set(range(1, len(sources) + 1))
            )

            if invalid_citations:
                message = "The answer cited a source that was not retrieved."
                generation.update(
                    level="ERROR",
                    status_message=message,
                    output={
                        "answer": answer,
                        "invalid_citations": invalid_citations,
                    },
                )
                root.update(
                    level="ERROR",
                    status_message=message,
                    output={"failure_category": "invalid_citation"},
                    metadata={"failure_category": "invalid_citation"},
                )
                yield (
                    "event: error\n"
                    f"data: {json.dumps(message)}\n\n"
                )
                return

            generation.update(output=answer)
            root.update(
                output={
                    "answer": answer,
                    "sources": sources,
                    "failure_category": "none",
                },
                metadata={"failure_category": "none"},
            )

            yield (
                "event: sources\n"
                f"data: {json.dumps(sources)}\n\n"
            )

        except GroqError as error:
            generation.update(
                level="ERROR",
                status_message=str(error),
                output={"error": str(error)},
            )
            root.update(
                level="ERROR",
                status_message=str(error),
                output={"failure_category": "provider_error"},
                metadata={"failure_category": "provider_error"},
            )

            yield (
                "event: error\n"
                f"data: {json.dumps(str(error))}\n\n"
            )

        except GeneratorExit:
            generation.update(
                level="WARNING",
                status_message=(
                    "Client disconnected before streaming completed."
                ),
                output={"partial_answer": "".join(answer_parts)},
            )
            root.update(
                level="WARNING",
                status_message="Client disconnected.",
                output={"failure_category": "client_disconnect"},
                metadata={"failure_category": "client_disconnect"},
            )
            raise

        except Exception as error:
            generation.update(
                level="ERROR",
                status_message=str(error),
                output={
                    "partial_answer": "".join(answer_parts),
                    "error": str(error),
                },
            )
            root.update(
                level="ERROR",
                status_message=str(error),
                output={"failure_category": "stream_error"},
                metadata={"failure_category": "stream_error"},
            )
            raise

        finally:
            generation.end()
            root.end()
            langfuse.flush()

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
    )



if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))

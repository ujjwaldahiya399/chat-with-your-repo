import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

from ingest import get_chunks

MODEL_NAME = "all-MiniLM-L6-v2"
CHROMA_DB_PATH = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "code_chunks"


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

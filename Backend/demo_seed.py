import gzip
import json
from pathlib import Path

from embedding_model import EmbeddingModel
from ingest import get_chunks

REPO_URL = "https://github.com/ujjwaldahiya399/chat-with-your-repo"
BRANCH = "main"
REPO_ROOT = Path(__file__).resolve().parent.parent
SEED_PATH = Path(__file__).with_name("demo_seed.json.gz")

SOURCE_FILES = (
    "Backend/ask.py",
    "Backend/embed.py",
    "Backend/embed_repo.py",
    "Backend/embedding_model.py",
    "Backend/ingest.py",
    "Backend/main.py",
    "Backend/query.py",
    "Backend/tests/test_ai_behaviors.py",
    "Backend/tests/test_embedding_model_reuse.py",
    "Backend/tests/test_repo_url_normalization.py",
)


def build_seed():
    ids = []
    documents = []
    metadatas = []

    for relative_path in SOURCE_FILES:
        file_path = REPO_ROOT / relative_path
        for start_line, content in get_chunks(str(file_path)):
            ids.append(f"{REPO_URL}@{BRANCH}:{relative_path}:{start_line}")
            documents.append(content)
            metadatas.append({
                "file_path": relative_path,
                "start_line": start_line,
                "repo_url": REPO_URL,
                "branch": BRANCH,
            })

    embeddings = EmbeddingModel().encode(documents).tolist()
    with gzip.open(SEED_PATH, "wt", encoding="utf-8") as output:
        json.dump({
            "ids": ids,
            "documents": documents,
            "metadatas": metadatas,
            "embeddings": embeddings,
        }, output)

    return len(ids)


def seed_collection(collection):
    with gzip.open(SEED_PATH, "rt", encoding="utf-8") as source:
        records = json.load(source)

    existing_ids = set(collection.get(ids=records["ids"])["ids"])
    missing = [
        index for index, chunk_id in enumerate(records["ids"])
        if chunk_id not in existing_ids
    ]
    if not missing:
        return 0

    collection.upsert(
        ids=[records["ids"][index] for index in missing],
        documents=[records["documents"][index] for index in missing],
        metadatas=[records["metadatas"][index] for index in missing],
        embeddings=[records["embeddings"][index] for index in missing],
    )
    return len(missing)


if __name__ == "__main__":
    print(f"Created seed with {build_seed()} chunks at {SEED_PATH}")

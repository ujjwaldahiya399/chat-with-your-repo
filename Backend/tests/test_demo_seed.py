import gzip
import json

import chromadb

from demo_seed import (
    BRANCH, REPO_ROOT, REPO_URL, SEED_PATH, SOURCE_FILES, seed_collection,
)
from ingest import get_chunks


def load_seed():
    with gzip.open(SEED_PATH, "rt", encoding="utf-8") as source:
        return json.load(source)


def test_seed_matches_current_source_files():
    records = load_seed()
    expected = [
        (path, start_line, content)
        for path in SOURCE_FILES
        for start_line, content in get_chunks(str(REPO_ROOT / path))
    ]

    assert records["ids"] == [
        f"{REPO_URL}@{BRANCH}:{path}:{line}"
        for path, line, _ in expected
    ]
    assert records["documents"] == [content for _, _, content in expected]
    assert records["metadatas"] == [
        {
            "file_path": path,
            "start_line": line,
            "repo_url": REPO_URL,
            "branch": BRANCH,
        }
        for path, line, _ in expected
    ]
    assert len(records["embeddings"]) == len(expected)
    assert all(len(vector) == 384 for vector in records["embeddings"])


def test_seed_fills_partial_collection_and_is_idempotent(tmp_path):
    records = load_seed()
    client = chromadb.PersistentClient(path=str(tmp_path / "db"))
    collection = client.get_or_create_collection("code_chunks")

    collection.upsert(
        ids=records["ids"][:1],
        documents=records["documents"][:1],
        metadatas=records["metadatas"][:1],
        embeddings=records["embeddings"][:1],
    )

    assert seed_collection(collection) == len(records["ids"]) - 1
    assert collection.count() == len(records["ids"])
    assert seed_collection(collection) == 0

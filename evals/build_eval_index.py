
import sys
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Backend"))

from embed import get_chunks, MODEL_NAME, CHROMA_DB_PATH

# Only index these known backend files for this experiment.
SOURCE_FILES = [
    "Backend/NOTES.md",
    "Backend/ask.py",
    "Backend/embed.py",
    "Backend/embed_repo.py",
    "Backend/ingest.py",
    "Backend/query.py",
    "Backend/requirements.txt",
    "Backend/test_decorator.py",
]

EVAL_DB_PATH = ROOT / "evals" / "chroma_benchmark"
EVAL_COLLECTION = "rag_backend_eval"


def build_index():
    # Safety: never write to the production database.
    if EVAL_DB_PATH.resolve() == Path(CHROMA_DB_PATH).resolve():
        raise RuntimeError(
            "Evaluation database cannot be the production database"
        )

    ids = []
    documents = []
    metadatas = []

    print("\n=== PREPARING EVALUATION DATASET ===")

    for relative_path in SOURCE_FILES:
        source_path = ROOT / relative_path

        if not source_path.is_file():
            raise FileNotFoundError(source_path)

        # Reuse the actual application's chunking logic.
        chunks = get_chunks(str(source_path))

        count = 0

        for start_line, content in chunks:
            if not content.strip():
                continue

            # Relative IDs remain stable when the project moves.
            chunk_id = f"{relative_path}:{start_line}"

            ids.append(chunk_id)
            documents.append(content)
            metadatas.append({
                "file_path": relative_path,
                "start_line": start_line,
                "project": "chat-with-your-repo",
            })

            count += 1

        print(f"{relative_path}: {count} chunks")

    if not documents:
        raise RuntimeError("No valid chunks were produced")

    if len(ids) != len(set(ids)):
        raise RuntimeError("Duplicate chunk IDs detected")

    print(f"\nTotal chunks prepared: {len(documents)}")

    # Generate embeddings before modifying the evaluation DB.
    print("\nGenerating embeddings...")
    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(
        documents,
        show_progress_bar=True
    ).tolist()

    # Separate database from Backend/chroma_db.
    EVAL_DB_PATH.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(
        path=str(EVAL_DB_PATH)
    )
    collection = client.get_or_create_collection(
        EVAL_COLLECTION
    )

    # Refresh only the dedicated evaluation collection.
    existing_ids = collection.get()["ids"]

    if existing_ids:
        collection.delete(ids=existing_ids)

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )

    print("\n=== EVALUATION INDEX CREATED ===")
    print(f"Database: {EVAL_DB_PATH}")
    print(f"Collection: {EVAL_COLLECTION}")
    print(f"Files indexed: {len(SOURCE_FILES)}")
    print(f"Chunks stored: {collection.count()}")


if __name__ == "__main__":
    build_index()

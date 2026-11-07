"""Build or verify a separate index for the frozen chunking-v2 experiment.

Install beside experimental_chunk_metadata.py and experimental_chunker_v2.py
in evals/. Existing production and baseline databases are never opened. A new
experiment directory is used for each corpus/representation/configuration.
Interrupted builds are left intact and are never silently overwritten.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path
import platform

import experimental_chunk_metadata as metadata_builder
import experimental_chunker_v2 as chunker


MODEL_NAME = "all-MiniLM-L6-v2"
COLLECTION_NAME = "rag_chunking_v2"


def digest(value):
    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def model_identity(model):
    """Record actual loaded weights, tokenizer vocabulary/options and layout."""
    import torch

    weights = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        tensor = tensor.detach().cpu().contiguous()
        header = [name, str(tensor.dtype), list(tensor.shape)]
        weights.update(json.dumps(header).encode("utf-8") + b"\0")
        weights.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    tokenizer = model.tokenizer
    token_options = {
        "class": type(tokenizer).__name__,
        "vocab_sha256": digest(tokenizer.get_vocab()),
        "special_tokens": {k: str(v) for k, v in tokenizer.special_tokens_map.items()},
        "padding_side": tokenizer.padding_side,
        "truncation_side": tokenizer.truncation_side,
    }
    basic = getattr(tokenizer, "basic_tokenizer", None)
    if basic is not None:
        token_options["basic"] = {
            key: getattr(basic, key, None) for key in
            ("do_lower_case", "strip_accents", "tokenize_chinese_chars", "do_split_on_punc")
        }
        token_options["basic"]["never_split"] = sorted(basic.never_split)
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        token_options["backend_sha256"] = hashlib.sha256(
            backend.to_str().encode("utf-8")
        ).hexdigest()
    config = model[0].auto_model.config
    return {
        "model_name": MODEL_NAME,
        "resolved_model_revision": getattr(config, "_commit_hash", None),
        "weights_sha256": weights.hexdigest(),
        "layout": str(model),
        "tokenizer": token_options,
        "model_max_seq_length": int(model.max_seq_length),
        "embedding_dimension": int(model.get_sentence_embedding_dimension()),
        "device": str(model.device),
    }


def experiment_directory(root, experiment_id, snapshot):
    root = Path(root).resolve()
    evals = (root / "evals").resolve()
    destination = (evals / "chroma_chunking_v2" / experiment_id[:16]).resolve()
    if not destination.is_relative_to(evals) or destination == evals:
        raise ValueError("Experimental database must stay inside evals/")
    protected = [(root / "Backend" / "chroma_db").resolve(),
                 (evals / "chroma_benchmark").resolve(), Path(snapshot).resolve()]
    for path in protected:
        if destination.is_relative_to(path) or path.is_relative_to(destination):
            raise ValueError(f"Experimental path overlaps a protected directory: {path}")
    return destination


def vectors_digest(ids, embeddings, dimension):
    import numpy as np

    array = np.asarray(embeddings, dtype="<f4")
    if array.shape != (len(ids), dimension) or not np.isfinite(array).all():
        raise ValueError("Invalid embedding dimensions or non-finite embedding values")
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate vector IDs")
    result = hashlib.sha256()
    for index in sorted(range(len(ids)), key=lambda i: ids[i]):
        result.update(ids[index].encode("utf-8") + b"\0")
        result.update(array[index].tobytes())
    return result.hexdigest()


def verify_collection(collection, records, dimension, expected_vectors):
    expected = {record["chunk_id"]: record for record in records}
    saved = collection.get(include=["documents", "metadatas", "embeddings"])
    ids = saved["ids"]
    if collection.count() != len(records) or len(ids) != len(records) or set(ids) != set(expected):
        raise ValueError("Stored chunk IDs or count disagree with the prepared records")
    if len(saved["documents"]) != len(ids) or len(saved["metadatas"]) != len(ids):
        raise ValueError("Stored documents or metadata are incomplete")
    for index, chunk_id in enumerate(ids):
        record = expected[chunk_id]
        if (saved["documents"][index] != record["source_text"]
                or saved["metadatas"][index] != record["metadata"]):
            raise ValueError(f"Stored source text or metadata disagrees: {chunk_id}")
    fingerprint = vectors_digest(ids, saved["embeddings"], dimension)
    if fingerprint != expected_vectors:
        raise ValueError("Stored embeddings disagree with the recorded vector fingerprint")


def build_index(snapshot, repo_id, root, budget=220):
    manifest, sources = chunker.load_source_snapshot(snapshot)
    repo_id = metadata_builder.canonical_repo_id(repo_id)
    # Validate source identity before loading the model or opening any database.
    metadata_builder._verified_files(manifest, sources)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(MODEL_NAME)
    if not isinstance(budget, int) or isinstance(budget, bool) or not 0 < budget <= model.max_seq_length:
        raise ValueError(f"budget must be between 1 and {model.max_seq_length}")
    records = metadata_builder.build_chunk_records(
        repo_id, manifest, sources, model.tokenizer, budget
    )
    if not records:
        raise ValueError("No indexable chunks were prepared")
    counts = dict(sorted(Counter(r["metadata"]["file_path"] for r in records).items()))
    configuration = {
        "schema_version": 1,
        "repo_id": repo_id,
        "snapshot_id": manifest["snapshot_id"],
        "collection": COLLECTION_NAME,
        "distance": "l2",
        "normalize_embeddings_argument": False,
        "encode_precision": "float32",
        "embedding_input_format": "File: {file_path}\\nScope: {symbol}\\nSource:\\n{source_text}",
        "stored_document": "exact source_text without the generated header",
        "token_budget": budget,
        "model": model_identity(model),
        "code_sha256": {
            "splitter": file_digest(chunker.__file__),
            "metadata_builder": file_digest(metadata_builder.__file__),
            "index_builder": file_digest(__file__),
        },
        "packages": {name: version(name) for name in
                     ("sentence-transformers", "transformers", "torch", "chromadb", "numpy")},
        "python": platform.python_version(),
        "records_sha256": digest(records),
        "chunk_count": len(records),
        "file_counts": counts,
    }
    experiment_id = digest(configuration)
    destination = experiment_directory(root, experiment_id, snapshot)
    saved_manifest = destination / "experiment.json"
    print(f"Snapshot ID: {manifest['snapshot_id']}")
    print(f"Prepared: {len(records)} chunks across {len(sources)} files")
    print(f"Largest final input: {max(r['metadata']['token_count'] for r in records)} tokens")
    print(f"Experiment ID: {experiment_id}")
    import chromadb

    if destination.exists():
        if not saved_manifest.is_file():
            raise ValueError(f"Existing experiment is incomplete; left untouched: {destination}")
        saved = json.loads(saved_manifest.read_text(encoding="utf-8"))
        if (saved.get("status") != "complete" or saved.get("experiment_id") != experiment_id
                or saved.get("configuration") != configuration):
            raise ValueError("Existing experiment manifest does not match; left untouched")
        expected_fingerprint = digest({
            "experiment_id": experiment_id, "vectors_sha256": saved["vectors_sha256"]
        })
        if expected_fingerprint != saved.get("index_fingerprint"):
            raise ValueError("Existing experiment fingerprint is inconsistent")
        client = chromadb.PersistentClient(path=str(destination))
        collection = client.get_collection(COLLECTION_NAME, embedding_function=None)
        if (collection.metadata or {}).get("experiment_id") != experiment_id:
            raise ValueError("Collection experiment identity does not match")
        verify_collection(collection, records, configuration["model"]["embedding_dimension"],
                          saved["vectors_sha256"])
        print("Existing completed experiment verified; no records rewritten.")
    else:
        print("Embedding the validated file-and-scope inputs...")
        embeddings = model.encode(
            [record["embedding_input"] for record in records],
            normalize_embeddings=False, precision="float32",
            convert_to_numpy=True, show_progress_bar=True,
        )
        vector_hash = vectors_digest([r["chunk_id"] for r in records], embeddings,
                                     configuration["model"]["embedding_dimension"])
        # Reserve a fresh directory only after all embeddings were generated.
        destination.mkdir(parents=True, exist_ok=False)
        client = chromadb.PersistentClient(path=str(destination))
        kwargs = {
            "name": COLLECTION_NAME, "embedding_function": None,
            "metadata": {"experiment_id": experiment_id, "snapshot_id": manifest["snapshot_id"]},
        }
        if "configuration" in inspect.signature(client.create_collection).parameters:
            kwargs["configuration"] = {"hnsw": {"space": "l2"}}
        else:
            kwargs["metadata"]["hnsw:space"] = "l2"
        collection = client.create_collection(**kwargs)
        collection.add(
            ids=[record["chunk_id"] for record in records],
            documents=[record["source_text"] for record in records],
            metadatas=[record["metadata"] for record in records],
            embeddings=embeddings.tolist(),
        )
        verify_collection(collection, records, configuration["model"]["embedding_dimension"], vector_hash)
        saved = {
            "status": "complete", "experiment_id": experiment_id,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "configuration": configuration, "vectors_sha256": vector_hash,
            "index_fingerprint": digest({"experiment_id": experiment_id, "vectors_sha256": vector_hash}),
            "comparison_scope": "New chunk boundaries plus file/scope headers; not a chunking-only ablation.",
        }
        with saved_manifest.open("x", encoding="utf-8") as stream:
            json.dump(saved, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        print("New experimental index created and read-back verified.")

    print(f"Database: {destination}")
    print(f"Collection: {COLLECTION_NAME}")
    print(f"Chunks stored: {collection.count()}")
    print(f"Manifest: {saved_manifest}")
    print(f"Index fingerprint: {saved['index_fingerprint']}")
    print("Production and historical baseline database paths were not opened.")
    print("Index construction is verified; retrieval quality has not yet been evaluated.")
    return saved_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--budget", type=int, default=220)
    args = parser.parse_args()
    try:
        build_index(args.snapshot, args.repo_id, Path(__file__).resolve().parents[1], args.budget)
    except Exception as error:
        parser.exit(1, f"Experimental index not completed: {error}\n")


if __name__ == "__main__":
    main()

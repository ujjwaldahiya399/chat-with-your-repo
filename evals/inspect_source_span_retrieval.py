"""Inspect one development question against the separate chunking-v2 index.

Uses the verified indexing preparation before checking model identity. Searches
the whole experimental collection, without filtering to the labelled file or
expanding parents. Does not modify labels or database records. An optional
--output writes a new JSON report and refuses to overwrite an existing file.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import build_chunking_v2_index as index_builder
import experimental_chunk_metadata as metadata_builder
import experimental_chunker_v2 as chunker


def load_question(path, question_id, snapshot, sources):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("purpose") != "development":
        raise ValueError("This runner requires schema-1 development labels")
    questions = data["questions"]
    ids = [q["id"] for q in questions]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate question IDs")
    matches = [q for q in questions if q["id"] == question_id]
    if len(matches) != 1:
        raise ValueError(f"Question not found: {question_id}")
    question = matches[0]
    if question.get("answerable") is not True or not question.get("question", "").strip():
        raise ValueError("Select an answerable question with nonempty text")
    evidence = question.get("required_evidence", [])
    if not evidence or len({u["id"] for u in evidence}) != len(evidence):
        raise ValueError("Required evidence must have unique IDs and cannot be empty")
    files = {item["path"]: item for item in snapshot["files"]}
    commit = data.get("source_commit")
    if not commit or commit != snapshot["git_base_commit"]:
        raise ValueError("Label source commit disagrees with the frozen snapshot")
    for unit in evidence:
        path = unit["file_path"]
        if path not in files or files[path]["origin"] != "git" or files[path]["source_commit"] != commit:
            raise ValueError(f"Evidence is not from the labelled Git revision: {path}")
        start, end = unit["start_line"], unit["end_line"]
        if (type(start) is not int or type(end) is not int
                or not 1 <= start <= end <= len(sources[path].splitlines())):
            raise ValueError(f"Invalid evidence line range: {unit['id']}")
    return question, commit


def evidence_coverage(units, retrieved, commit, repo_id, snapshot_id):
    results = []
    for unit in units:
        required = set(range(unit["start_line"], unit["end_line"] + 1))
        missing = required.copy()
        for meta in retrieved:
            if (meta["file_path"] == unit["file_path"]
                    and meta.get("source_commit") == commit
                    and meta["repo_id"] == repo_id and meta["snapshot_id"] == snapshot_id):
                missing.difference_update(range(meta["start_line"], meta["end_line"] + 1))
        results.append({
            **unit, "required_lines": len(required),
            "covered_lines": len(required) - len(missing),
            "complete": not missing, "missing_lines": sorted(missing),
        })
    return results


def inspect_question(index_dir, snapshot_dir, labels_path, question_id, top_k=5):
    if type(top_k) is not int or top_k <= 0:
        raise ValueError("top-k must be a positive integer")
    index_dir = Path(index_dir).resolve()
    run = json.loads((index_dir / "experiment.json").read_text(encoding="utf-8"))
    config = run["configuration"]
    experiment_id = index_builder.digest(config)
    if run.get("status") != "complete" or run.get("experiment_id") != experiment_id:
        raise ValueError("Experimental index manifest is incomplete or inconsistent")
    if index_builder.digest({"experiment_id": experiment_id, "vectors_sha256": run["vectors_sha256"]}) != run.get("index_fingerprint"):
        raise ValueError("Index fingerprint is inconsistent")
    snapshot, sources = chunker.load_source_snapshot(snapshot_dir)
    if snapshot["snapshot_id"] != config["snapshot_id"]:
        raise ValueError("Selected snapshot differs from the indexed corpus")
    question, commit = load_question(labels_path, question_id, snapshot, sources)
    for name, module in [("splitter", chunker), ("metadata_builder", metadata_builder),
                         ("index_builder", index_builder)]:
        if index_builder.file_digest(module.__file__) != config["code_sha256"][name]:
            raise ValueError(f"The {name} code differs from this experiment's recorded version")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(config["model"]["model_name"])
    records = metadata_builder.build_chunk_records(
        config["repo_id"], snapshot, sources, model.tokenizer, config["token_budget"]
    )
    if index_builder.digest(records) != config["records_sha256"]:
        raise ValueError("Prepared records differ from the indexed records")
    # Fingerprint after the SAME tokenizer preparation used during indexing.
    if index_builder.model_identity(model) != config["model"]:
        raise ValueError("Prepared query model differs from the indexed model")
    if chunker.count_tokens(question["question"], model.tokenizer) > model.max_seq_length:
        raise ValueError("Question exceeds the model input limit; refusing silent truncation")

    import chromadb
    client = chromadb.PersistentClient(path=str(index_dir))
    collection = client.get_collection(config["collection"], embedding_function=None)
    if (collection.metadata or {}).get("experiment_id") != experiment_id:
        raise ValueError("Collection experiment identity differs")
    index_builder.verify_collection(
        collection, records, config["model"]["embedding_dimension"], run["vectors_sha256"]
    )
    result = collection.query(
        query_embeddings=model.encode(
            [question["question"]],
            normalize_embeddings=config["normalize_embeddings_argument"],
            precision=config["encode_precision"],
        ).tolist(),
        n_results=min(top_k, len(records)), include=["metadatas"],
    )
    ids, metas = result["ids"][0], result["metadatas"][0]
    expected = {record["chunk_id"]: record["metadata"] for record in records}
    if len(ids) != len(metas) or len(set(ids)) != len(ids):
        raise ValueError("Query returned inconsistent or duplicate records")
    if any(key not in expected or meta != expected[key] for key, meta in zip(ids, metas)):
        raise ValueError("Query result differs from verified source metadata")
    coverage = evidence_coverage(
        question["required_evidence"], metas, commit, config["repo_id"], config["snapshot_id"]
    )
    complete = sum(unit["complete"] for unit in coverage)
    return {
        "schema_version": 1, "purpose": "development",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "question_id": question_id, "question": question["question"],
        "repo_id": config["repo_id"], "snapshot_id": config["snapshot_id"],
        "source_commit": commit, "experiment_id": experiment_id,
        "index_fingerprint": run["index_fingerprint"],
        "labels_sha256": index_builder.file_digest(labels_path),
        "inspector_sha256": index_builder.file_digest(__file__),
        "top_k": top_k, "retrieved_count": len(ids),
        "retrieved": [{"rank": rank, "chunk_id": key, "metadata": meta}
                      for rank, (key, meta) in enumerate(zip(ids, metas), start=1)],
        "evidence": coverage, "complete_evidence_units": complete,
        "total_evidence_units": len(coverage),
        "evidence_unit_coverage_at_k": complete / len(coverage),
        "parent_expansion": False,
        "scope": "One development question; not an overall quality or baseline comparison.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("evals/golden_set_source_spans.dev.json"))
    parser.add_argument("--question-id", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, help="Optional new JSON report; existing files are preserved")
    args = parser.parse_args()
    try:
        if args.output and args.output.exists():
            raise ValueError(f"Output already exists; left untouched: {args.output}")
        report = inspect_question(args.index, args.snapshot, args.labels, args.question_id, args.top_k)
        if args.output:
            with args.output.open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, ensure_ascii=False)
                stream.write("\n")
    except Exception as error:
        parser.exit(1, f"Development retrieval check not completed: {error}\n")
    print(f"\nQuestion {report['question_id']}: {report['question']}\n")
    for item in report["retrieved"]:
        meta = item["metadata"]
        print(f"{item['rank']}. {meta['file_path']}:{meta['start_line']}-{meta['end_line']}")
    print("\nRequired evidence:")
    for unit in report["evidence"]:
        status = "COVERED" if unit["complete"] else "INCOMPLETE"
        print(f"{unit['id']}: {unit['covered_lines']}/{unit['required_lines']} required lines — {status}")
    print(f"\n{report['question_id']} evidence coverage@{report['top_k']}: "
          f"{report['complete_evidence_units']}/{report['total_evidence_units']} units")
    print("Development diagnostic only; no baseline comparison yet.")
    if args.output:
        print(f"Saved report: {args.output}")


if __name__ == "__main__":
    main()

"""Find the exact minimum chunks needed to cover each development question.

Place beside inspect_source_span_retrieval.py in evals/ and run from the project
root. Uses labels as an evaluation oracle only, never as a retrieval strategy.
Reads indexed metadata; does not load an embedding model or update records.
A new JSON report records a minimum-size witness for each question. This checks
the capacity of a chunk-count limit, not ranking quality or a token budget.
"""

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import tempfile


def minimum_cover(required, candidates):
    """Return a smallest tuple of candidate keys covering all required atoms.

    Exact set cover with memoization. Intended for the small evidence subsets
    in this project, rather than large arbitrary overlapping set-cover inputs.
    None means required evidence is missing from the candidate collection.
    """
    required = frozenset(required)
    if not required:
        return ()
    unique = {}
    for key in sorted(candidates):
        covered = frozenset(candidates[key]) & required
        if covered:
            unique.setdefault(covered, key)
    if frozenset().union(*unique) != required:
        return None
    # A strict subset cannot improve the minimum number of chunks required.
    rows = sorted((key, covered) for covered, key in unique.items()
                  if not any(covered < other for other in unique))
    choices = {atom: [i for i, (_, covered) in enumerate(rows) if atom in covered]
               for atom in required}

    @lru_cache(None)
    def solve(remaining):
        if not remaining:
            return ()
        atom = min(remaining, key=lambda x: (len(choices[x]), repr(x)))
        ordered = sorted(choices[atom],
                         key=lambda i: (-len(rows[i][1] & remaining), rows[i][0]))
        best = None
        for i in ordered:
            key, covered = rows[i]
            candidate = (key,) + solve(remaining - covered)
            if best is None or len(candidate) < len(best):
                best = candidate
            if len(best) == 1:
                break
        return best

    return solve(required)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("evals/golden_set_source_spans.dev.json"))
    parser.add_argument("--snapshot", type=Path, default=Path("evals/corpus_snapshots/50c2529de39ff786"))
    args = parser.parse_args()

    from inspect_source_span_retrieval import load_question, evidence_coverage
    from experimental_chunker_v2 import load_source_snapshot
    from build_chunking_v2_index import digest
    import chromadb

    label_bytes = args.labels.read_bytes()
    labels = json.loads(label_bytes)
    questions = sorted(labels["questions"], key=lambda q: q["id"])
    if not questions:
        raise ValueError("No development questions found")
    first_id = questions[0]["id"]
    reference = json.loads((args.reports_dir / (first_id + ".json")).read_text())
    if (reference["labels_sha256"] != hashlib.sha256(label_bytes).hexdigest()
            or reference["source_commit"] != labels["source_commit"]
            or reference["top_k"] != 5 or reference.get("purpose") != "development"):
        raise ValueError("Saved evaluation does not match these development labels")
    snapshot, sources = load_source_snapshot(args.snapshot)
    if snapshot["snapshot_id"] != reference["snapshot_id"]:
        raise ValueError("Selected source snapshot differs from the evaluated index")
    for question in questions:
        load_question(args.labels, question["id"], snapshot, sources)

    matches = []
    for path in Path("evals/chroma_chunking_v2").glob("*/experiment.json"):
        manifest = json.loads(path.read_text())
        if manifest.get("index_fingerprint") == reference["index_fingerprint"]:
            matches.append((path.parent, manifest))
    if len(matches) != 1:
        raise ValueError(f"Expected one matching index; found {len(matches)}")
    index_dir, manifest = matches[0]
    config = manifest["configuration"]
    if (manifest.get("status") != "complete"
            or digest(config) != reference["experiment_id"]
            or manifest["experiment_id"] != reference["experiment_id"]
            or digest({"experiment_id": manifest["experiment_id"],
                       "vectors_sha256": manifest["vectors_sha256"]}) != reference["index_fingerprint"]
            or any(config[key] != reference[key] for key in ["repo_id", "snapshot_id"])):
        raise ValueError("Experimental index identity is inconsistent")
    client = chromadb.PersistentClient(path=str(index_dir))
    collection = client.get_collection(config["collection"], embedding_function=None)
    if (collection.metadata or {}).get("experiment_id") != reference["experiment_id"]:
        raise ValueError("Collection identity disagrees with the evaluated index")
    saved = collection.get(include=["metadatas"])
    if (len(saved["ids"]) != config["chunk_count"]
            or len(set(saved["ids"])) != len(saved["ids"])
            or len(saved["metadatas"]) != len(saved["ids"])):
        raise ValueError("Index metadata is incomplete")
    indexed = {key: meta for key, meta in zip(saved["ids"], saved["metadatas"])
               if meta.get("source_commit") == labels["source_commit"]
               and meta.get("repo_id") == config["repo_id"]
               and meta.get("snapshot_id") == config["snapshot_id"]}

    rows = []
    for q in questions:
        required = {(unit["file_path"], line) for unit in q["required_evidence"]
                    for line in range(unit["start_line"], unit["end_line"] + 1)}
        candidates = {key: {atom for atom in required
                            if atom[0] == meta["file_path"]
                            and meta["start_line"] <= atom[1] <= meta["end_line"]}
                      for key, meta in indexed.items()}
        witness = minimum_cover(required, candidates)
        count = None if witness is None else len(witness)
        if witness is not None:
            check = evidence_coverage(q["required_evidence"], [indexed[key] for key in witness],
                                      labels["source_commit"], config["repo_id"], config["snapshot_id"])
            if not all(unit["complete"] for unit in check):
                raise ValueError("Minimum-cover witness failed the existing coverage scorer")
        status = "evidence missing from index" if count is None else (
            "can fit within 5" if count <= 5 else "cannot fit within 5")
        print(f"{q['id']}: minimum chunks = {count}; {status}", flush=True)
        rows.append({"question_id": q["id"], "question": q["question"],
                     "minimum_chunks": count, "can_fit_at_5": count is not None and count <= 5,
                     "witness": [{"chunk_id": key, "metadata": indexed[key]}
                                 for key in witness or ()]})
    feasible = sum(row["can_fit_at_5"] for row in rows)
    blocked = [row["question_id"] for row in rows
               if row["minimum_chunks"] is not None and row["minimum_chunks"] > 5]
    missing = [row["question_id"] for row in rows if row["minimum_chunks"] is None]
    print(f"\nQuestions whose full evidence can fit within 5 chunks: {feasible}/{len(rows)}")
    print("Require more than 5 chunks:", ", ".join(blocked) or "(none)")
    print("Missing indexed evidence:", ", ".join(missing) or "(none)")
    result = {"purpose": "development", "top_k": 5,
              "index_fingerprint": reference["index_fingerprint"],
              "labels_sha256": hashlib.sha256(label_bytes).hexdigest(),
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "scope": "Exact evidence-capacity oracle using source-range metadata; not a retriever, answer-quality measure, or token-budget comparison.",
              "fully_coverable_questions_at_5": feasible, "questions": rows}
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix="dev_capacity_",
                                     suffix=".json", dir="evals", delete=False) as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
        print("Saved report:", stream.name)


if __name__ == "__main__":
    main()

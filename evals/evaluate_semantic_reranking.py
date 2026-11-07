"""Test semantic top-20 candidates followed by a cross-encoder's top five.

Place in evals/ beside the existing experimental helpers. Both models load once.
The first run may download the public reranker. Source code is scored locally.
No chunks are re-embedded and no database records are changed. A new JSON report
compares saved semantic top-five, fresh semantic top-five from the candidate
query, and reranked top-five. Coverage of all 20 candidates is a diagnostic,
not a result at the five-chunk budget. Labels are used only for scoring.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import math
from pathlib import Path
import tempfile
from time import perf_counter


RERANKER = "cross-encoder/ms-marco-MiniLM-L6-v2"
DEPTH = 20
FINAL_K = 5
PAIR_LIMIT = 512


def rank_scores(keys, scores, k=FINAL_K):
    values = [float(value) for value in scores]
    if len(values) != len(keys) or not all(math.isfinite(v) for v in values):
        raise ValueError("Reranker returned inconsistent or non-finite scores")
    if len(set(keys)) != len(keys):
        raise ValueError("Candidate IDs are not unique")
    # Equal cross-encoder scores retain the semantic order.
    return sorted(range(len(keys)), key=lambda i: (-values[i], i))[:k]


def weight_digest(model):
    import torch
    state = model.state_dict() if hasattr(model, "state_dict") else model.model.state_dict()
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(json.dumps([name, str(tensor.dtype), list(tensor.shape)]).encode() + b"\0")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--capacity-report", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("evals/golden_set_source_spans.dev.json"))
    parser.add_argument("--snapshot", type=Path, default=Path("evals/corpus_snapshots/50c2529de39ff786"))
    args = parser.parse_args()

    import build_chunking_v2_index as builder
    import experimental_chunk_metadata as metadata_builder
    import experimental_chunker_v2 as chunker
    from inspect_source_span_retrieval import load_question, evidence_coverage

    label_bytes = args.labels.read_bytes()
    labels = json.loads(label_bytes)
    questions = sorted(labels["questions"], key=lambda q: q["id"])
    if not questions:
        raise ValueError("No development questions")
    label_hash = hashlib.sha256(label_bytes).hexdigest()
    old_reports = {q["id"]: json.loads((args.reports_dir / (q["id"] + ".json")).read_text())
                   for q in questions}
    reference = old_reports[questions[0]["id"]]
    for q in questions:
        old = old_reports[q["id"]]
        if (old["question_id"] != q["id"] or old["question"] != q["question"]
                or old["labels_sha256"] != label_hash or old["top_k"] != FINAL_K
                or old.get("purpose") != "development" or old.get("parent_expansion") is not False
                or old["source_commit"] != labels["source_commit"]
                or any(old[key] != reference[key] for key in
                       ["index_fingerprint", "experiment_id", "repo_id", "snapshot_id"])):
            raise ValueError("Saved reports do not match these labels and experiment")
    capacity = json.loads(args.capacity_report.read_text())
    if (capacity["index_fingerprint"] != reference["index_fingerprint"]
            or capacity["labels_sha256"] != label_hash):
        raise ValueError("Capacity report refers to different labels or index")
    if {q["question_id"] for q in capacity["questions"]} != {q["id"] for q in questions}:
        raise ValueError("Capacity report question set differs")
    feasible = {q["question_id"] for q in capacity["questions"] if q["can_fit_at_5"]}

    matches = []
    for path in Path("evals/chroma_chunking_v2").glob("*/experiment.json"):
        run = json.loads(path.read_text())
        if run.get("index_fingerprint") == reference["index_fingerprint"]:
            matches.append((path.parent, run))
    if len(matches) != 1:
        raise ValueError(f"Expected one matching index; found {len(matches)}")
    index_dir, run = matches[0]
    config = run["configuration"]
    if (run.get("status") != "complete" or builder.digest(config) != reference["experiment_id"]
            or run["experiment_id"] != reference["experiment_id"]
            or builder.digest({"experiment_id": run["experiment_id"],
                               "vectors_sha256": run["vectors_sha256"]}) != reference["index_fingerprint"]
            or any(config[key] != reference[key] for key in ["repo_id", "snapshot_id"])):
        raise ValueError("Experimental index identity is inconsistent")
    snapshot, sources = chunker.load_source_snapshot(args.snapshot)
    if snapshot["snapshot_id"] != config["snapshot_id"]:
        raise ValueError("Selected snapshot differs from the indexed corpus")
    for q in questions:
        load_question(args.labels, q["id"], snapshot, sources)
    for name, module in [("splitter", chunker), ("metadata_builder", metadata_builder),
                         ("index_builder", builder)]:
        if builder.file_digest(module.__file__) != config["code_sha256"][name]:
            raise ValueError("Experiment helper has changed: " + name)

    from sentence_transformers import SentenceTransformer, CrossEncoder
    import chromadb
    print("Loading and verifying the embedding model once...", flush=True)
    model = SentenceTransformer(config["model"]["model_name"])
    records = metadata_builder.build_chunk_records(
        config["repo_id"], snapshot, sources, model.tokenizer, config["token_budget"])
    if builder.digest(records) != config["records_sha256"] or builder.model_identity(model) != config["model"]:
        raise ValueError("Prepared records or query-model identity differ from the experiment")
    expected = {r["chunk_id"]: r for r in records}
    for q in questions:
        if chunker.count_tokens(q["question"], model.tokenizer) > model.max_seq_length:
            raise ValueError("Question would be truncated: " + q["id"])
    client = chromadb.PersistentClient(path=str(index_dir))
    collection = client.get_collection(config["collection"], embedding_function=None)
    if (collection.metadata or {}).get("experiment_id") != reference["experiment_id"]:
        raise ValueError("Collection experiment identity differs")
    builder.verify_collection(collection, records, config["model"]["embedding_dimension"], run["vectors_sha256"])

    # Gather candidates before reading any labels for scoring.
    candidates, search_seconds = {}, {}
    for q in questions:
        started = perf_counter()
        embedding = model.encode([q["question"]],
                                 normalize_embeddings=config["normalize_embeddings_argument"],
                                 precision=config["encode_precision"]).tolist()
        result = collection.query(query_embeddings=embedding, n_results=min(DEPTH, len(records)),
                                  include=["metadatas"])
        ids, metas = result["ids"][0], result["metadatas"][0]
        if (len(ids) != len(metas) or len(set(ids)) != len(ids) or not ids
                or any(key not in expected or meta != expected[key]["metadata"]
                       for key, meta in zip(ids, metas))):
            raise ValueError("Query results disagree with verified records")
        candidates[q["id"]] = ids
        search_seconds[q["id"]] = perf_counter() - started

    print("Loading the reranker once (first run may download its weights)...", flush=True)
    reranker = CrossEncoder(RERANKER, max_length=PAIR_LIMIT)
    reranker_identity = {"model_name": RERANKER, "weights_sha256": weight_digest(reranker),
                         "max_length": PAIR_LIMIT,
                         "tokenizer_class": type(reranker.tokenizer).__name__,
                         "vocabulary_sha256": builder.digest(reranker.tokenizer.get_vocab()),
                         "device": str(getattr(reranker, "device", "unknown"))}
    rows = []
    for q in questions:
        ids = candidates[q["id"]]
        pairs = [(q["question"], expected[key]["embedding_input"]) for key in ids]
        for query, document in pairs:
            length = len(reranker.tokenizer(query, document, add_special_tokens=True,
                                           truncation=False)["input_ids"])
            if length > PAIR_LIMIT:
                raise ValueError(f"Reranker input would truncate: {q['id']} has {length} tokens")
        started = perf_counter()
        scores = reranker.predict(pairs, batch_size=16, show_progress_bar=False)
        selected = rank_scores(ids, scores)
        elapsed = perf_counter() - started

        old = old_reports[q["id"]]
        old_ids = [item["chunk_id"] for item in old["retrieved"]]
        if (len(old_ids) != len(set(old_ids)) or len(old_ids) > FINAL_K
                or any(item["chunk_id"] not in expected
                       or item["metadata"] != expected[item["chunk_id"]]["metadata"]
                       for item in old["retrieved"])):
            raise ValueError("Saved semantic matches disagree with verified records")
        rankings = {"saved_semantic5": old_ids, "fresh_semantic5": ids[:FINAL_K],
                    "reranked5": [ids[i] for i in selected], "candidate_pool20": ids}
        covered, details = {}, {}
        for name, keys in rankings.items():
            details[name] = evidence_coverage(q["required_evidence"],
                [expected[key]["metadata"] for key in keys], labels["source_commit"],
                config["repo_id"], config["snapshot_id"])
            covered[name] = sum(unit["complete"] for unit in details[name])
        if covered["saved_semantic5"] != old["complete_evidence_units"]:
            raise ValueError("Saved semantic coverage does not reproduce")
        row = {"question_id": q["id"], "question": q["question"],
               "total": len(q["required_evidence"]), "covered": covered, "evidence": details,
               "rankings": rankings, "saved_order_matches_fresh": old_ids == ids[:FINAL_K],
               "search_seconds": search_seconds[q["id"]], "rerank_seconds": elapsed,
               "reranker_scores": {key: float(value) for key, value in zip(ids, scores)}}
        rows.append(row)
        print(f"{q['id']}: semantic5 {covered['fresh_semantic5']}/{row['total']} | "
              f"reranked5 {covered['reranked5']}/{row['total']} | "
              f"pool20 {covered['candidate_pool20']}/{row['total']}", flush=True)

    total = sum(row["total"] for row in rows)
    summary = {}
    print("\n=== DEVELOPMENT COMPARISON ===")
    for method in ["saved_semantic5", "fresh_semantic5", "reranked5", "candidate_pool20"]:
        covered = sum(row["covered"][method] for row in rows)
        full = sum(row["covered"][method] == row["total"] for row in rows)
        feasible_full = sum(row["question_id"] in feasible and row["covered"][method] == row["total"]
                            for row in rows)
        summary[method] = {"covered_units": covered, "total_units": total,
                           "fully_covered_questions": full, "feasible_questions_fully_covered": feasible_full}
        suffix = " (pool diagnostic; includes up to 20 chunks)" if method == "candidate_pool20" else ""
        print(f"{method}: {covered}/{total} ({covered / total:.1%}); "
              f"full {full}/{len(rows)}; feasible full {feasible_full}/{len(feasible)}{suffix}")
    changed = [row["question_id"] for row in rows if not row["saved_order_matches_fresh"]]
    print("Saved vs fresh semantic top-five order changed:", ", ".join(changed) or "(none)")
    for label, predicate in [("Improved", lambda a,b: b > a), ("Regressed", lambda a,b: b < a)]:
        matches = [row["question_id"] for row in rows
                   if predicate(row["covered"]["fresh_semantic5"], row["covered"]["reranked5"])]
        print(label + " vs fresh semantic5:", ", ".join(matches) or "(none)")
    result = {"purpose": "development", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "index_fingerprint": reference["index_fingerprint"], "labels_sha256": label_hash,
              "runner_sha256": builder.file_digest(__file__), "reranker": reranker_identity,
              "candidate_depth": DEPTH, "final_k": FINAL_K, "rerank_input": "existing file/scope embedding_input",
              "packages": {name: version(name) for name in ["sentence-transformers", "transformers", "torch", "chromadb"]},
              "scope": "Development evidence coverage only. Compare fresh_semantic5 with reranked5 on the same top-20 pool. Candidate-pool coverage uses a different chunk budget. R12 cannot achieve full evidence coverage in five current chunks. No generated-answer or historical-chunking-baseline claim.",
              "summary": summary, "questions": rows}
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix="dev_reranking_",
                                     suffix=".json", dir="evals", delete=False) as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
        print("Saved report:", stream.name)


if __name__ == "__main__":
    main()

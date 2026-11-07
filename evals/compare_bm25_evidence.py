"""Compare BM25 with saved semantic top-5 results on the same development set.

Place in evals/ and run from the project root with --reports-dir pointing to
the existing dev_batch directory. No embedding model, network requests, or
database record writes are used. A new JSON report is saved under evals/.
BM25 searches exact stored source text, with identifier-aware tokenization.
This is a development retrieval comparison, not generated-answer evaluation
or a comparison against the historical chunking baseline.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import tempfile


def tokens(text):
    text = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1 \2", text)
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    return re.findall(r"[a-z]+|[0-9]+", text.lower())


class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        if not documents:
            raise ValueError("Cannot rank an empty corpus")
        self.counts = [Counter(tokens(doc)) for doc in documents]
        self.lengths = [sum(count.values()) for count in self.counts]
        self.average = sum(self.lengths) / len(documents) or 1.0
        self.df = Counter(term for count in self.counts for term in count)
        self.k1, self.b = k1, b

    def rank(self, question, ids, k=5):
        scores = [0.0] * len(self.counts)
        for term in set(tokens(question)):
            frequency = self.df.get(term, 0)
            if not frequency:
                continue
            idf = math.log(1 + (len(scores) - frequency + 0.5) / (frequency + 0.5))
            for i, count in enumerate(self.counts):
                tf = count.get(term, 0)
                if tf:
                    norm = self.k1 * (1 - self.b + self.b * self.lengths[i] / self.average)
                    scores[i] += idf * tf * (self.k1 + 1) / (tf + norm)
        ordered = sorted(range(len(scores)), key=lambda i: (-scores[i], ids[i]))
        return [(i, scores[i]) for i in ordered if scores[i] > 0][:k]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path, required=True)
    parser.add_argument("--labels", type=Path, default=Path("evals/golden_set_source_spans.dev.json"))
    args = parser.parse_args()

    # These existing project helpers do not load an embedding model on import.
    from inspect_source_span_retrieval import evidence_coverage
    from build_chunking_v2_index import digest
    import chromadb

    label_bytes = args.labels.read_bytes()
    labels = json.loads(label_bytes)
    questions = sorted(labels["questions"], key=lambda q: q["id"])
    question_ids = [q["id"] for q in questions]
    if (labels.get("schema_version") != 1 or labels.get("purpose") != "development"
            or not questions or len(set(question_ids)) != len(question_ids)
            or any(q.get("answerable") is not True for q in questions)):
        raise ValueError("Expected unique answerable development questions")
    reports = {
        q["id"]: json.loads((args.reports_dir / (q["id"] + ".json")).read_text())
        for q in questions
    }
    reference = reports[questions[0]["id"]]
    for q in questions:
        old = reports[q["id"]]
        if (old["question_id"] != q["id"] or old["question"] != q["question"]
                or old["labels_sha256"] != hashlib.sha256(label_bytes).hexdigest()
                or old["source_commit"] != labels["source_commit"]
                or old["top_k"] != 5 or old.get("parent_expansion") is not False
                or old.get("purpose") != "development"):
            raise ValueError("Saved report does not match current labels: " + q["id"])
        for field in ["index_fingerprint", "experiment_id", "repo_id", "snapshot_id"]:
            if old[field] != reference[field]:
                raise ValueError("Saved reports refer to different experiments")

    matches = []
    for path in Path("evals/chroma_chunking_v2").glob("*/experiment.json"):
        manifest = json.loads(path.read_text())
        if manifest.get("index_fingerprint") == reference["index_fingerprint"]:
            matches.append((path.parent, manifest))
    if len(matches) != 1:
        raise ValueError(f"Expected one matching experimental index; found {len(matches)}")
    index_dir, manifest = matches[0]
    config = manifest["configuration"]
    if (manifest.get("status") != "complete"
            or digest(config) != reference["experiment_id"]
            or digest({"experiment_id": manifest["experiment_id"],
                       "vectors_sha256": manifest["vectors_sha256"]}) != reference["index_fingerprint"]
            or any(config[key] != reference[key] for key in ["repo_id", "snapshot_id"])):
        raise ValueError("Experimental index identity is inconsistent")

    client = chromadb.PersistentClient(path=str(index_dir))
    collection = client.get_collection(config["collection"], embedding_function=None)
    if (collection.metadata or {}).get("experiment_id") != reference["experiment_id"]:
        raise ValueError("Collection identity disagrees with saved reports")
    saved = collection.get(include=["documents", "metadatas"])
    ids, documents, metas = saved["ids"], saved["documents"], saved["metadatas"]
    if (len(ids) != config["chunk_count"] or len(set(ids)) != len(ids)
            or len(documents) != len(ids) or len(metas) != len(ids)
            or any(not isinstance(doc, str) for doc in documents)):
        raise ValueError("Collection content is incomplete")
    expected = dict(zip(ids, metas))
    ranker = BM25(documents)
    rows = []

    def coverage(question, metadata):
        return evidence_coverage(question["required_evidence"], metadata,
                                 labels["source_commit"], config["repo_id"], config["snapshot_id"])

    for q in questions:
        old = reports[q["id"]]
        retrieved = old["retrieved"]
        if (len(retrieved) != old["retrieved_count"] or len(retrieved) > 5
                or len({item["chunk_id"] for item in retrieved}) != len(retrieved)
                or any(expected.get(item["chunk_id"]) != item["metadata"] for item in retrieved)):
            raise ValueError("Saved semantic matches disagree with this index: " + q["id"])
        semantic_units = coverage(q, [item["metadata"] for item in retrieved])
        semantic = sum(unit["complete"] for unit in semantic_units)
        if semantic != old["complete_evidence_units"]:
            raise ValueError("Saved semantic coverage does not reproduce: " + q["id"])
        ranking = ranker.rank(q["question"], ids)
        bm25_units = coverage(q, [metas[i] for i, _ in ranking])
        lexical = sum(unit["complete"] for unit in bm25_units)
        total = len(q["required_evidence"])
        rows.append({"question_id": q["id"], "question": q["question"],
                     "total": total, "semantic_complete": semantic, "bm25_complete": lexical,
                     "bm25_evidence": bm25_units,
                     "bm25_retrieved": [{"chunk_id": ids[i], "score": score, "metadata": metas[i]}
                                        for i, score in ranking]})
        print(f"{q['id']}: semantic {semantic}/{total} | BM25 {lexical}/{total}", flush=True)

    total = sum(row["total"] for row in rows)
    summary = {"questions": len(rows), "total_evidence_units": total}
    for method in ["semantic", "bm25"]:
        covered = sum(row[method + "_complete"] for row in rows)
        full = sum(row[method + "_complete"] == row["total"] for row in rows)
        summary[method] = {"covered_units": covered, "coverage_at_5": covered / total,
                           "fully_covered_questions": full}
        print(f"\n{method}: {covered}/{total} ({covered / total:.1%}); "
              f"fully covered {full}/{len(rows)}")
    for name, predicate in [("improved", lambda a, b: b > a),
                            ("regressed", lambda a, b: b < a),
                            ("unchanged", lambda a, b: b == a)]:
        summary[name] = [row["question_id"] for row in rows
                         if predicate(row["semantic_complete"], row["bm25_complete"])]
        print(f"{name}: {', '.join(summary[name]) or '(none)'}")
    result = {"purpose": "development", "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "index_fingerprint": reference["index_fingerprint"],
              "labels_sha256": hashlib.sha256(label_bytes).hexdigest(),
              "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "semantic_reports": str(args.reports_dir), "top_k": 5,
              "bm25_configuration": {"k1": 1.5, "b": 0.75,
                  "text": "exact stored source text", "tokenizer": "split camel case and nonletters; retain numbers",
                  "query_term_frequency": "binary", "zero_score_matches": "excluded"},
              "scope": "Ranking comparison on existing development questions; same chunks and k, not equal context tokens. Semantic embeddings include file/scope headers; BM25 uses source text only. No historical-baseline or answer-quality claim.",
              "summary": summary, "questions": rows}
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", prefix="dev_bm25_",
                                     suffix=".json", dir="evals", delete=False) as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
        print("\nSaved comparison:", stream.name)


if __name__ == "__main__":
    main()

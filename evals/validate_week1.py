"""Read-only validation for the frozen Week 1 Chat-with-Your-Repo evaluation.

Usage from project root:
    python evals/validate_week1.py

Only standard-library dependencies are needed. This checks the user's existing
manifest, golden set and benchmark output; it never modifies ChromaDB.
"""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVALS = ROOT / "evals"


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def main():
    golden = read_json(EVALS / "golden_set_rag.json")
    snapshot = read_json(EVALS / "week1_metrics_snapshot.json")
    manifest_path = EVALS / "chunk_manifest.md"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing {manifest_path}; do not substitute historical records")
    manifest = manifest_path.read_text(encoding="utf-8")
    known_ids = set(re.findall(r"(?m)^CHUNK ID:\s*(\S+)\s*$", manifest))

    assert len(known_ids) == 32, f"Expected 32 unique indexed chunk IDs; found {len(known_ids)}"
    assert len(golden) == 25, f"Expected 25 questions; found {len(golden)}"
    assert len({q['id'] for q in golden}) == len(golden), "Duplicate question IDs"
    answerable = [q for q in golden if q['answerable']]
    unanswerable = [q for q in golden if not q['answerable']]
    assert len(answerable) == 23 and len(unanswerable) == 2
    for q in golden:
        labels = q['relevant_chunks']
        assert bool(labels) == q['answerable'], f"Bad answerability label: {q['id']}"
        assert len(labels) == len(set(labels)), f"Duplicate relevance label in {q['id']}"
        missing = set(labels) - known_ids
        assert not missing, f"{q['id']} references absent chunk IDs: {sorted(missing)}"
    print("PASS: 25 unique questions (23 answerable, 2 unanswerable)")
    print("PASS: Every relevant chunk ID is grounded in the 32-chunk manifest")

    results_path = EVALS / "rag_benchmark_results.json"
    if not results_path.is_file():
        raise FileNotFoundError(
            f"Missing {results_path}. Run python evals/run_rag_benchmark.py first."
        )
    results = read_json(results_path)
    assert results['corpus_chunks'] == 32, "Corpus size changed"
    assert results['answerable_questions'] == 23
    assert results['unanswerable_questions'] == 2
    actual_questions = results['questions']
    assert len(actual_questions) == 25
    for expected, actual in zip(golden, actual_questions):
        assert expected['id'] == actual['id'], "Dataset order or content changed"
        assert expected['question'] == actual['question'], f"Question changed: {expected['id']}"
        assert expected['relevant_chunks'] == actual['relevant_chunks'], (
            f"Relevance labels changed for {expected['id']}"
        )

    for method, recorded in snapshot['three_way_baseline'].items():
        reported = results['summary'][method]
        for name, expected_score in recorded.items():
            actual_score = reported[name]
            assert abs(actual_score - expected_score) <= 0.00051, (
                f"{method} {name}: current {actual_score:.6f} vs recorded {expected_score:.3f}. "
                "This may indicate an index or benchmark change; investigate before updating the report."
            )
    print("PASS: Local detailed results reproduce all 12 recorded summary metrics (rounding tolerance)")
    print(f"Index fingerprint recorded by local run: {results['index_sha256']}")
    print("Week 1 evaluation contract verified. No files or databases were modified.")


if __name__ == "__main__":
    main()

import json
from pathlib import Path

data = json.loads(
    Path("evals/golden_set_source_spans.dev.json").read_text()
)
question = data["questions"][0]
revision = data["source_commit"]

def evidence_covered(chunks, evidence):
    missing = set(range(
        evidence["start_line"], evidence["end_line"] + 1
    ))

    for chunk in chunks:
        same_source = (
            chunk["file_path"] == evidence["file_path"]
            and chunk["source_commit"] == revision
        )
        if same_source:
            missing.difference_update(range(
                chunk["start_line"], chunk["end_line"] + 1
            ))

    return not missing

def chunk(start, end, path="Backend/ingest.py", commit=revision):
    return {
        "file_path": path,
        "source_commit": commit,
        "start_line": start,
        "end_line": end,
    }

cases = [
    ("Whole function", [chunk(13, 67)], 3),
    ("Partial evidence", [chunk(28, 30), chunk(40, 44)], 2),
    ("Combined pieces", [
        chunk(28, 30), chunk(40, 44),
        chunk(47, 54), chunk(55, 63),
    ], 3),
    ("Duplicate evidence", [chunk(28, 30), chunk(28, 30)], 1),
    ("Wrong file", [chunk(1, 113, path="Backend/ask.py")], 0),
    ("Wrong revision", [chunk(1, 113, commit="different")], 0),
]

evidence = question["required_evidence"]
for name, chunks, expected in cases:
    covered = sum(evidence_covered(chunks, item) for item in evidence)
    assert covered == expected, (name, covered, expected)
    print(f"{name}: {covered}/{len(evidence)} evidence units covered")

print("All six evaluator checks passed.")
"""Build the frozen micrograd claim-faithfulness development cases."""

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_SNAPSHOT_ID = (
    "2d319a78dffd90aa6fa26bf69d8c92a258b3ed4179b6d2129e8fbde280c55541"
)

EXPECTED_COMMIT = (
    "7bc720e951fe422b8f8814aa5aa1b64121d26b4c"
)

EXPECTED_REPOSITORY = "https://github.com/karpathy/micrograd"


QUESTIONS = [
    {
        "id": "F01",
        "case_kind": "simple_summary",
        "answerable": True,
        "question": (
            "According to the repository, what does micrograd implement, "
            "what kind of computation graph and value granularity does it "
            "use, and what is its intended purpose?"
        ),
        "expected_claims": [
            "micrograd implements reverse-mode automatic differentiation.",
            "It operates over a dynamically built directed acyclic graph.",
            "The graph operates on individual scalar values rather than tensors.",
            "It includes a small neural-network library with a PyTorch-like API.",
            "The project is intended primarily for educational use.",
        ],
        "evidence": [
            {
                "file_path": "README.md",
                "start_line": 6,
                "end_line": 6,
            }
        ],
    },
    {
        "id": "F02",
        "case_kind": "code_behavior",
        "answerable": True,
        "question": (
            "When Value.backward() is called, where does traversal begin, "
            "how is the topological list constructed, what gradient is seeded, "
            "and in what direction are the stored backward functions executed?"
        ),
        "expected_claims": [
            "Traversal begins from the Value on which backward() is called.",
            "The recursive traversal visits each node's _prev values before appending the node.",
            "The starting output Value has its gradient set to 1.",
            "The backward functions execute in reversed topological order, from the output toward its input values.",
        ],
        "evidence": [
            {
                "file_path": "micrograd/engine.py",
                "start_line": 54,
                "end_line": 70,
            }
        ],
    },
    {
        "id": "F03",
        "case_kind": "shared_operand_gradient",
        "answerable": True,
        "question": (
            "For x = Value(3.0) and y = x + x, what should x.grad be after "
            "y.backward(), and which implementation details produce that result?"
        ),
        "expected_claims": [
            "x.grad should be 2.",
            "The output gradient is seeded to 1.",
            "Both operands of the addition refer to the same x object.",
            "The addition backward function accumulates the output gradient into both operand references using +=.",
        ],
        "evidence": [
            {
                "file_path": "micrograd/engine.py",
                "start_line": 13,
                "end_line": 20,
            },
            {
                "file_path": "micrograd/engine.py",
                "start_line": 54,
                "end_line": 70,
            },
        ],
    },
    {
        "id": "F04",
        "case_kind": "relu_edge_case",
        "answerable": True,
        "question": (
            "How does Value.relu() behave in both the forward and backward "
            "passes for negative, exactly zero, and positive inputs?"
        ),
        "expected_claims": [
            "A negative input produces forward output 0.",
            "An input of exactly zero produces forward output 0.",
            "A positive input is passed through unchanged.",
            "The backward contribution is zero for negative and exactly zero outputs.",
            "For a positive output, the incoming output gradient is accumulated into the input gradient.",
        ],
        "evidence": [
            {
                "file_path": "micrograd/engine.py",
                "start_line": 45,
                "end_line": 50,
            }
        ],
    },
    {
        "id": "F05",
        "case_kind": "operator_reuse",
        "answerable": True,
        "question": (
            "How are self / other and other / self implemented by Value, "
            "and which existing operations do those implementations reuse?"
        ),
        "expected_claims": [
            "self / other is implemented as self multiplied by other raised to -1.",
            "other / self is implemented as other multiplied by self raised to -1.",
            "Division therefore reuses multiplication and power operations.",
            "Value.__pow__ accepts only integer or floating-point exponents.",
        ],
        "evidence": [
            {
                "file_path": "micrograd/engine.py",
                "start_line": 35,
                "end_line": 41,
            },
            {
                "file_path": "micrograd/engine.py",
                "start_line": 87,
                "end_line": 91,
            },
        ],
    },
    {
        "id": "F06",
        "case_kind": "network_architecture",
        "answerable": True,
        "question": (
            "How does MLP construct and execute its layers, and why is its "
            "final layer linear rather than ReLU?"
        ),
        "expected_claims": [
            "MLP prepends the input size to the requested output sizes.",
            "It creates a Layer between every consecutive pair of sizes.",
            "All layers except the last are constructed with nonlinearity enabled.",
            "The last layer is constructed with nonlinearity disabled.",
            "The input is passed through the layers sequentially.",
        ],
        "evidence": [
            {
                "file_path": "micrograd/nn.py",
                "start_line": 45,
                "end_line": 54,
            }
        ],
    },
    {
        "id": "F07",
        "case_kind": "parameter_reasoning",
        "answerable": True,
        "question": (
            "How many trainable Value parameters does MLP(3, [4, 4, 1]) "
            "contain? Show the count for each layer and explain how the "
            "parameters are collected."
        ),
        "expected_claims": [
            "Each neuron contains one weight per input plus one bias.",
            "The first layer has 4 times (3 plus 1), or 16, parameters.",
            "The second layer has 4 times (4 plus 1), or 20, parameters.",
            "The final layer has 1 times (4 plus 1), or 5, parameters.",
            "The MLP contains 41 trainable Value parameters in total.",
            "Layer and MLP parameter methods flatten parameters from their contained neurons and layers.",
        ],
        "evidence": [
            {
                "file_path": "micrograd/nn.py",
                "start_line": 13,
                "end_line": 25,
            },
            {
                "file_path": "micrograd/nn.py",
                "start_line": 30,
                "end_line": 40,
            },
            {
                "file_path": "micrograd/nn.py",
                "start_line": 45,
                "end_line": 57,
            },
        ],
    },
    {
        "id": "F08",
        "case_kind": "multi_file_testing",
        "answerable": True,
        "question": (
            "How do the engine tests use PyTorch as a reference for both "
            "forward values and gradients, and how do the assertion strategies "
            "differ between test_sanity_check and test_more_ops?"
        ),
        "expected_claims": [
            "The tests recreate equivalent expressions using micrograd Value objects and PyTorch tensors.",
            "The PyTorch tensors use double precision and require gradients.",
            "Both tests compare forward results and backward gradients.",
            "test_sanity_check uses exact equality for its output and gradient comparisons.",
            "test_more_ops uses an absolute tolerance of 1e-6 for its output and two gradient comparisons.",
        ],
        "evidence": [
            {
                "file_path": "README.md",
                "start_line": 63,
                "end_line": 69,
            },
            {
                "file_path": "test/test_engine.py",
                "start_line": 4,
                "end_line": 26,
            },
            {
                "file_path": "test/test_engine.py",
                "start_line": 28,
                "end_line": 67,
            },
        ],
    },
    {
        "id": "U01",
        "case_kind": "unsupported_metric",
        "answerable": False,
        "question": (
            "What exact numerical classification accuracy did the demonstrated "
            "moon-dataset neural network achieve?"
        ),
        "expected_claims": [
            "Exact refusal: I don't know.",
            "The retrieval corpus does not report a numerical classification accuracy.",
            "The model should not infer an accuracy from the displayed decision-boundary image.",
        ],
        "evidence": [],
    },
    {
        "id": "U02",
        "case_kind": "unsupported_performance",
        "answerable": False,
        "question": (
            "What exact training time and hardware were used to train the "
            "demonstrated moon-dataset neural network?"
        ),
        "expected_claims": [
            "Exact refusal: I don't know.",
            "The retrieval corpus does not report training duration or hardware.",
        ],
        "evidence": [],
    },
]


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def evidence_with_hash(snapshot, evidence):
    path = snapshot / evidence["file_path"]

    if not path.is_file():
        raise FileNotFoundError(
            f"Evidence file does not exist: {evidence['file_path']}"
        )

    lines = path.read_text(encoding="utf-8").splitlines(
        keepends=True
    )
    start = evidence["start_line"]
    end = evidence["end_line"]

    if not 1 <= start <= end <= len(lines):
        raise ValueError(
            f"Invalid evidence range for {evidence['file_path']}: "
            f"{start}-{end}"
        )

    text = "".join(lines[start - 1:end])

    return {
        **evidence,
        "text_sha256": sha256(text.encode("utf-8")),
    }


def build_cases(snapshot):
    manifest = json.loads(
        (snapshot / "manifest.json").read_text(encoding="utf-8")
    )

    if manifest.get("snapshot_id") != EXPECTED_SNAPSHOT_ID:
        raise ValueError("Unexpected snapshot ID.")

    if manifest.get("git_base_commit") != EXPECTED_COMMIT:
        raise ValueError("Unexpected micrograd commit.")

    if manifest.get("repository_url") != EXPECTED_REPOSITORY:
        raise ValueError("Unexpected repository URL.")

    retrieval_files = sorted(
        record["path"]
        for record in manifest["files"]
        if record.get("retrieval_input") is True
    )

    built_questions = []

    for question in QUESTIONS:
        built_questions.append(
            {
                **question,
                "evidence": [
                    evidence_with_hash(snapshot, item)
                    for item in question["evidence"]
                ],
            }
        )

    if len(built_questions) != 10:
        raise ValueError("Expected exactly ten questions.")

    if sum(q["answerable"] for q in built_questions) != 8:
        raise ValueError(
            "Expected eight answerable and two unanswerable questions."
        )

    return {
        "schema_version": 1,
        "purpose": (
            "micrograd_claim_faithfulness_development_set_"
            "not_independent_holdout"
        ),
        "repository_url": EXPECTED_REPOSITORY,
        "source_commit": EXPECTED_COMMIT,
        "snapshot_id": EXPECTED_SNAPSHOT_ID,
        "retrieval_files": retrieval_files,
        "claim_labels": [
            "SUPPORTED",
            "CONTRADICTED",
            "NOT_IN_SOURCE",
        ],
        "questions": built_questions,
        "limitations": (
            "Ten hand-authored development questions over a known small "
            "repository. This is not an independent holdout or production "
            "accuracy estimate. Notebooks and images are excluded because "
            "the application does not index their file extensions."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path(
            "evals/corpus_snapshots/2d319a78dffd90aa"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "evals/micrograd_faithfulness_cases.json"
        ),
    )
    args = parser.parse_args()

    cases = build_cases(args.snapshot.resolve())
    serialized = (
        json.dumps(cases, indent=2, ensure_ascii=False) + "\n"
    )

    if args.output.exists():
        existing = args.output.read_text(encoding="utf-8")

        if existing != serialized:
            raise RuntimeError(
                "Existing cases file differs; refusing to overwrite."
            )

        print("Cases file reused and verified.")
    else:
        args.output.write_text(serialized, encoding="utf-8")
        print("Cases file created and verified.")

    print(f"Cases: {args.output}")
    print(f"Questions: {len(cases['questions'])}")
    print(
        "Answerable:",
        sum(q["answerable"] for q in cases["questions"]),
    )
    print(
        "Unanswerable:",
        sum(not q["answerable"] for q in cases["questions"]),
    )


if __name__ == "__main__":
    main()
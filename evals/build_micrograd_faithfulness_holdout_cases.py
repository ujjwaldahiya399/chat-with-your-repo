"""Build the pre-registered harder micrograd faithfulness holdout."""

import argparse
import json
from pathlib import Path

from build_micrograd_faithfulness_cases import (
    EXPECTED_COMMIT,
    EXPECTED_REPOSITORY,
    EXPECTED_SNAPSHOT_ID,
    evidence_with_hash,
)


QUESTIONS = [
    {
        "id": "H01",
        "case_kind": "package_and_test_requirements",
        "answerable": True,
        "question": (
            "According to setup.py and the README, what Python version is "
            "required, does the package declare runtime dependencies, how is "
            "micrograd installed, and why is PyTorch still mentioned?"
        ),
        "expected_claims": [
            "setup.py declares Python 3.6 or newer.",
            "setup.py does not declare an install_requires dependency list.",
            "The README instructs users to install micrograd with pip install micrograd.",
            "PyTorch is required for running the repository's unit tests.",
            "The tests use PyTorch as a reference for checking calculated gradients.",
        ],
        "evidence": [
            {
                "file_path": "setup.py",
                "start_line": 6,
                "end_line": 22,
            },
            {
                "file_path": "README.md",
                "start_line": 8,
                "end_line": 12,
            },
            {
                "file_path": "README.md",
                "start_line": 63,
                "end_line": 69,
            },
        ],
    },
    {
        "id": "H02",
        "case_kind": "cross_file_mixed_arithmetic",
        "answerable": True,
        "question": (
            "How does Neuron compute an activation when its input list contains "
            "ordinary Python numbers, and which Value operations keep the "
            "weighted sum differentiable?"
        ),
        "expected_claims": [
            "Neuron pairs weights and inputs with zip.",
            "Each paired weight and input are multiplied.",
            "Value.__mul__ converts a non-Value operand into a Value.",
            "Python sum starts the accumulated activation from the neuron's bias.",
            "Value.__add__ creates graph nodes and its backward closure accumulates gradients into both operands.",
            "The activation is passed through ReLU only when the neuron's nonlin flag is true.",
        ],
        "evidence": [
            {
                "file_path": "micrograd/nn.py",
                "start_line": 15,
                "end_line": 25,
            },
            {
                "file_path": "micrograd/engine.py",
                "start_line": 13,
                "end_line": 33,
            },
        ],
    },
    {
        "id": "H03",
        "case_kind": "silent_input_truncation",
        "answerable": True,
        "question": (
            "What happens if Neuron(3) is called with only two inputs or with "
            "four inputs? Explain the behavior from the implementation and "
            "state whether an input-length check is performed."
        ),
        "expected_claims": [
            "The implementation uses zip to pair weights and inputs.",
            "With two inputs, only two of the three weights participate.",
            "With four inputs, the fourth input is ignored.",
            "The bias is still included in the activation.",
            "Neuron.__call__ performs no explicit input-length validation.",
        ],
        "evidence": [
            {
                "file_path": "micrograd/nn.py",
                "start_line": 15,
                "end_line": 22,
            }
        ],
    },
    {
        "id": "H04",
        "case_kind": "layer_output_shape",
        "answerable": True,
        "question": (
            "What does Layer.__call__ return for one neuron versus multiple "
            "neurons, and how does that affect the intermediate and final "
            "values produced by MLP(2, [3, 1])?"
        ),
        "expected_claims": [
            "A one-neuron Layer returns its sole output directly rather than a list.",
            "A Layer with multiple neurons returns a list of outputs.",
            "The first layer of MLP(2, [3, 1]) produces a three-element list.",
            "The final one-neuron layer produces a single Value.",
            "MLP passes each layer's output directly into the next layer.",
        ],
        "evidence": [
            {
                "file_path": "micrograd/nn.py",
                "start_line": 30,
                "end_line": 54,
            }
        ],
    },
    {
        "id": "H05",
        "case_kind": "nested_parameter_reset",
        "answerable": True,
        "question": (
            "When zero_grad() is called on an MLP, how does it reach every "
            "weight and bias, and what value is assigned to each gradient?"
        ),
        "expected_claims": [
            "MLP inherits zero_grad from Module.",
            "Module.zero_grad calls self.parameters().",
            "MLP.parameters flattens parameters from every layer.",
            "Layer.parameters flattens parameters from every neuron.",
            "Neuron.parameters returns its weights followed by its bias.",
            "zero_grad assigns 0 to every returned parameter's grad field.",
        ],
        "evidence": [
            {
                "file_path": "micrograd/nn.py",
                "start_line": 4,
                "end_line": 11,
            },
            {
                "file_path": "micrograd/nn.py",
                "start_line": 24,
                "end_line": 25,
            },
            {
                "file_path": "micrograd/nn.py",
                "start_line": 39,
                "end_line": 40,
            },
            {
                "file_path": "micrograd/nn.py",
                "start_line": 56,
                "end_line": 57,
            },
        ],
    },
    {
        "id": "H06",
        "case_kind": "repeated_backward_accumulation",
        "answerable": True,
        "question": (
            "For x = Value(2.0) and y = x * 3, what is x.grad after calling "
            "y.backward() once and then a second time without clearing "
            "gradients? Explain why and identify the network helper that can "
            "clear parameter gradients."
        ),
        "expected_claims": [
            "After the first backward call, x.grad is 3.",
            "After the second backward call without clearing, x.grad is 6.",
            "backward sets the output y.grad to 1 on each call.",
            "backward does not clear the existing gradient on x.",
            "The multiplication closure accumulates 3 times the output gradient into x.grad.",
            "Module.zero_grad clears gradients for parameters returned by a network module.",
        ],
        "evidence": [
            {
                "file_path": "micrograd/engine.py",
                "start_line": 24,
                "end_line": 31,
            },
            {
                "file_path": "micrograd/engine.py",
                "start_line": 54,
                "end_line": 70,
            },
            {
                "file_path": "micrograd/nn.py",
                "start_line": 4,
                "end_line": 8,
            },
        ],
    },
    {
        "id": "H07",
        "case_kind": "power_operand_constraint",
        "answerable": True,
        "question": (
            "How does Value.__pow__ treat x ** 2 compared with "
            "x ** Value(2), which node is stored as a predecessor, and for "
            "which operand does its backward closure calculate a gradient?"
        ),
        "expected_claims": [
            "Value.__pow__ accepts only integer or floating-point exponents.",
            "x ** 2 is accepted.",
            "x ** Value(2) fails the implementation's type assertion.",
            "The power output stores only the base Value as a predecessor.",
            "The backward closure calculates and accumulates the derivative with respect to the base.",
            "The implementation does not calculate a gradient for the exponent.",
        ],
        "evidence": [
            {
                "file_path": "micrograd/engine.py",
                "start_line": 35,
                "end_line": 43,
            }
        ],
    },
    {
        "id": "H08",
        "case_kind": "neuron_initialization_and_repr",
        "answerable": True,
        "question": (
            "How is Neuron(3, nonlin=False) initialized, in what order are its "
            "parameters returned, and what exact representation string does "
            "it produce?"
        ),
        "expected_claims": [
            "It creates three weight Value objects.",
            "Each weight's data is sampled with random.uniform(-1, 1).",
            "The bias is initialized as Value(0).",
            "The nonlin flag is stored as false.",
            "parameters returns the weights followed by the bias.",
            "Its representation is LinearNeuron(3).",
        ],
        "evidence": [
            {
                "file_path": "micrograd/nn.py",
                "start_line": 13,
                "end_line": 28,
            }
        ],
    },
    {
        "id": "HU01",
        "case_kind": "unsupported_random_values",
        "answerable": False,
        "question": (
            "What exact three numerical weight values will Neuron(3) receive "
            "when it is constructed?"
        ),
        "expected_claims": [
            "Exact refusal: I don't know.",
            "The indexed corpus specifies a random distribution but no seed or exact draws.",
        ],
        "evidence": [],
    },
    {
        "id": "HU02",
        "case_kind": "unsupported_dependency_version",
        "answerable": False,
        "question": (
            "What exact PyTorch version is required to run micrograd's unit tests?"
        ),
        "expected_claims": [
            "Exact refusal: I don't know.",
            "The indexed corpus mentions PyTorch but does not specify a version.",
        ],
        "evidence": [],
    },
    {
        "id": "HU03",
        "case_kind": "unsupported_training_configuration",
        "answerable": False,
        "question": (
            "What exact learning rate and number of training iterations were "
            "used for the moon-dataset classifier?"
        ),
        "expected_claims": [
            "Exact refusal: I don't know.",
            "The indexed corpus mentions SGD but does not give those exact settings.",
            "The demo notebook is not part of the indexed retrieval files.",
        ],
        "evidence": [],
    },
    {
        "id": "HU04",
        "case_kind": "unsupported_excluded_file",
        "answerable": False,
        "question": (
            "What exact copyright year and copyright holder are written in "
            "the repository's LICENSE file?"
        ),
        "expected_claims": [
            "Exact refusal: I don't know.",
            "LICENSE is retained in the frozen snapshot for provenance but is not an indexed retrieval file.",
            "The indexed README states the MIT license but not the requested copyright details.",
        ],
        "evidence": [],
    },
]


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

    built_questions = [
        {
            **question,
            "evidence": [
                evidence_with_hash(snapshot, item)
                for item in question["evidence"]
            ],
        }
        for question in QUESTIONS
    ]

    if len(built_questions) != 12:
        raise ValueError("Expected exactly twelve questions.")

    if sum(q["answerable"] for q in built_questions) != 8:
        raise ValueError(
            "Expected eight answerable and four unanswerable questions."
        )

    ids = [question["id"] for question in built_questions]

    if len(ids) != len(set(ids)):
        raise ValueError("Question IDs must be unique.")

    if any(
        question["answerable"] != bool(question["evidence"])
        for question in built_questions
    ):
        raise ValueError(
            "Answerable questions need evidence and unanswerable questions "
            "must not have evidence."
        )

    return {
        "schema_version": 1,
        "purpose": (
            "micrograd_claim_faithfulness_preregistered_harder_holdout_"
            "same_frozen_corpus"
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
            "Twelve pre-registered harder questions over the same frozen "
            "micrograd corpus. The set was authored after reviewing the pilot, "
            "so it is not independently authored and does not test transfer to "
            "another repository. Notebooks, images, and LICENSE are excluded "
            "from retrieval according to the frozen manifest."
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
            "evals/micrograd_faithfulness_holdout_cases.json"
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
                "Existing holdout file differs; refusing to overwrite."
            )

        print("Holdout cases file reused and verified.")
    else:
        args.output.write_text(serialized, encoding="utf-8")
        print("Holdout cases file created and verified.")

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

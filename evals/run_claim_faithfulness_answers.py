"""Generate frozen answers for the micrograd faithfulness evaluation.

Default mode validates the frozen preflight and current generation
configuration without making API calls. Add --live to make one application
 generation attempt for each of the ten frozen questions.
 The Groq SDK retains the current application's default retry behaviour.
"""

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "Backend"

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from dotenv import load_dotenv
from groq import Groq

from query import MODEL_ID, SYSTEM_PROMPT, query_stream


DEFAULT_PREFLIGHT = (
    ROOT / "evals" / "claim_faithfulness_preflight.json"
)
DEFAULT_CASES = (
    ROOT / "evals" / "micrograd_faithfulness_cases.json"
)
DEFAULT_RUN_ROOT = (
    ROOT / "evals" / "claim_faithfulness_runs"
)

APPLICATION_FILES = (
    "embed.py",
    "embedding_model.py",
    "ingest.py",
    "main.py",
    "query.py",
)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def file_sha256(path):
    return sha256(Path(path).read_bytes())


def save_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def reconstruct_context(sources):
    return "\n\n".join(
        (
            f"[Source {source['source_number']}: "
            f"{source['file_path']}:"
            f"{source['start_line']}]\n"
            f"{source['text']}"
        )
        for source in sources
    )


def validate_inputs(preflight_path, cases_path):
    preflight = json.loads(
        preflight_path.read_text(encoding="utf-8")
    )
    cases = json.loads(
        cases_path.read_text(encoding="utf-8")
    )

    if preflight.get("mode") != (
        "retrieval_preflight_no_generation"
    ):
        raise ValueError("Unexpected preflight mode.")

    if (
        preflight.get("cases_sha256")
        != file_sha256(cases_path)
    ):
        raise ValueError(
            "Cases changed after the retrieval preflight."
        )

    if (
        preflight["summary"][
            "answerable_with_all_required_evidence"
        ]
        != 8
    ):
        raise ValueError(
            "Not all answerable questions have their "
            "required evidence."
        )

    if (
        preflight.get("snapshot_id")
        != cases.get("snapshot_id")
    ):
        raise ValueError(
            "Preflight and cases use different snapshots."
        )

    if (
        preflight.get("source_commit")
        != cases.get("source_commit")
    ):
        raise ValueError(
            "Preflight and cases use different commits."
        )

    expected_code_hashes = (
        preflight["application_code_sha256"]
    )

    for name in APPLICATION_FILES:
        current = file_sha256(BACKEND / name)

        if current != expected_code_hashes[name]:
            raise ValueError(
                f"Application code changed after preflight: "
                f"Backend/{name}"
            )

    case_lookup = {
        question["id"]: question
        for question in cases["questions"]
    }
    rows = preflight["questions"]

    if len(rows) != 10:
        raise ValueError(
            "Expected ten preflight questions."
        )

    if len({row["question_id"] for row in rows}) != 10:
        raise ValueError(
            "Preflight question IDs must be unique."
        )

    for row in rows:
        question_id = row["question_id"]

        if question_id not in case_lookup:
            raise ValueError(
                f"Unknown preflight question: {question_id}"
            )

        case = case_lookup[question_id]

        if row["question"] != case["question"]:
            raise ValueError(
                f"Question text differs for {question_id}."
            )

        if row["answerable"] != case["answerable"]:
            raise ValueError(
                f"Answerability differs for {question_id}."
            )

        sources = row["sources"]

        if len(sources) != 5:
            raise ValueError(
                f"Expected five sources for {question_id}."
            )

        if [
            source["source_number"]
            for source in sources
        ] != [1, 2, 3, 4, 5]:
            raise ValueError(
                f"Source numbering differs for {question_id}."
            )

        for source in sources:
            if (
                source["repo_url"]
                != cases["repository_url"]
            ):
                raise ValueError(
                    f"Repository differs for {question_id}."
                )

            if (
                source["branch"]
                != cases["source_commit"]
            ):
                raise ValueError(
                    f"Source commit differs for {question_id}."
                )

        if (
            reconstruct_context(sources)
            != row["labeled_context"]
        ):
            raise ValueError(
                f"Context reconstruction differs for "
                f"{question_id}."
            )

    return preflight, cases, case_lookup


def generate_answer(context, question, api_key):
    started = perf_counter()
    first_token_seconds = None
    parts = []

    try:
        for token in query_stream(
            context,
            question,
            api_key,
        ):
            if first_token_seconds is None:
                first_token_seconds = (
                    perf_counter() - started
                )

            parts.append(token)

        answer = "".join(parts)
        cited_numbers = sorted(
            {
                int(number)
                for number in re.findall(
                    r"\bSource\s+(\d+)\b",
                    answer,
                )
            }
        )
        invalid_citations = [
            number
            for number in cited_numbers
            if number not in {1, 2, 3, 4, 5}
        ]

        return {
            "status": "ok",
            "answer": answer,
            "answer_sha256": sha256(
                answer.encode("utf-8")
            ),
            "generation_seconds": (
                perf_counter() - started
            ),
            "time_to_first_token_seconds": (
                first_token_seconds
            ),
            "cited_source_numbers": cited_numbers,
            "invalid_citations": invalid_citations,
            "application_outcome": (
                "invalid_citation"
                if invalid_citations
                else "success"
            ),
            "exact_refusal": (
                answer.strip() == "I don't know."
            ),
        }

    except Exception as error:
        message = str(error).replace(
            api_key,
            "[REDACTED]",
        )

        return {
            "status": "error",
            "error_type": type(error).__name__,
            "error_message": message[:2000],
            "partial_answer": "".join(parts),
            "generation_seconds": (
                perf_counter() - started
            ),
            "time_to_first_token_seconds": (
                first_token_seconds
            ),
        }


def write_review(path, report, case_lookup):
    sections = [
        "# Micrograd claim faithfulness review",
        "",
        (
            "Expected claims and evidence below were never "
            "included in generation prompts."
        ),
        (
            "Split each generated answer into small factual "
            "claims. Label each claim SUPPORTED, "
            "CONTRADICTED, or NOT_IN_SOURCE."
        ),
        "",
    ]

    for row in report["questions"]:
        case = case_lookup[row["question_id"]]

        sections.extend(
            [
                f"## {row['question_id']}",
                "",
                row["question"],
                "",
                f"Answerable: {row['answerable']}",
                "",
                "### Generated answer",
                "",
                row.get(
                    "answer",
                    row.get(
                        "partial_answer",
                        "(not generated)",
                    ),
                ),
                "",
                "### Expected claims",
                "",
            ]
        )

        sections.extend(
            f"- {claim}"
            for claim in case["expected_claims"]
        )

        sections.extend(
            [
                "",
                "### Claim annotations",
                "",
                (
                    "| Claim | Exact claim text | Label | "
                    "Cited source | Evidence span | Rationale |"
                ),
                "|---|---|---|---|---|---|",
                "| C1 |  |  |  |  |  |",
                "",
                "### Retrieved sources",
                "",
            ]
        )

        for source in row["sources"]:
            sections.extend(
                [
                    (
                        f"Source {source['source_number']}: "
                        f"{source['file_path']}:"
                        f"{source['start_line']}-"
                        f"{source['end_line']}"
                    ),
                    "",
                    "````text",
                    source["text"],
                    "````",
                    "",
                ]
            )

    path.write_text(
        "\n".join(sections),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight",
        type=Path,
        default=DEFAULT_PREFLIGHT,
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES,
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=DEFAULT_RUN_ROOT,
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run one application generation attempt per question.",
    )
    args = parser.parse_args()

    preflight_path = args.preflight.resolve()
    cases_path = args.cases.resolve()

    preflight, cases, case_lookup = validate_inputs(
        preflight_path,
        cases_path,
    )

    print(
        "Validated frozen preflight, cases, sources, "
        "contexts, and application code."
    )
    print(f"Generation model: {MODEL_ID}")
    print("Questions: 10")

    if not args.live:
        print(
            "No API calls made. Add --live only after "
            "reviewing this validation."
        )
        return

    load_dotenv(BACKEND / ".env")
    load_dotenv(ROOT / ".env")

    api_key = os.environ.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing from Backend/.env "
            "and the project-root .env."
        )

    with Groq(
        api_key=api_key,
        max_retries=0,
    ) as client:
        available_models = {
            model.id
            for model in client.models.list().data
        }

    if MODEL_ID not in available_models:
        raise ValueError(
            f"Model is not available for this key: "
            f"{MODEL_ID}"
        )

    timestamp = datetime.now(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%S_%fZ")

    run_directory = args.run_root.resolve() / timestamp
    run_directory.mkdir(
        parents=True,
        exist_ok=False,
    )

    answers_path = run_directory / "answers.json"
    review_path = run_directory / "review.md"

    report = {
        "purpose": cases["purpose"],
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "repository_url": cases["repository_url"],
        "source_commit": cases["source_commit"],
        "snapshot_id": cases["snapshot_id"],
        "preflight_sha256": file_sha256(
            preflight_path
        ),
        "cases_sha256": file_sha256(cases_path),
        "runner_sha256": file_sha256(__file__),
        "query_py_sha256": file_sha256(
            BACKEND / "query.py"
        ),
        "generation_configuration": {
            "model": MODEL_ID,
            "system_prompt": SYSTEM_PROMPT,
            "stream": True,
            "temperature": (
                "provider default; omitted"
            ),
            "max_tokens": (
                "provider default; omitted"
            ),
            "seed": "omitted",
        },
        "questions": [],
        "limitations": (
            "Ten hand-authored development questions from "
            "a known repository; not an independent holdout. "
            "One stochastic generation per question. "
            "The reused application query_stream interface "
            "does not expose finish reason, token usage, or "
            "provider-reported cost. Claim grading remains "
            "manual and is not included in this raw run."
        ),
    }

    save_json(answers_path, report)

    for preflight_row in preflight["questions"]:
        row = {
            "question_id": (
                preflight_row["question_id"]
            ),
            "case_kind": preflight_row["case_kind"],
            "answerable": preflight_row["answerable"],
            "question": preflight_row["question"],
            "sources": preflight_row["sources"],
            "labeled_context": (
                preflight_row["labeled_context"]
            ),
        }

        row.update(
            generate_answer(
                row["labeled_context"],
                row["question"],
                api_key,
            )
        )
        report["questions"].append(row)
        save_json(answers_path, report)

        print(
            f"{row['question_id']}: "
            f"{row['status']} "
            f"{row.get('application_outcome', '')}"
        )

        if row["status"] != "ok":
            write_review(
                review_path,
                report,
                case_lookup,
            )
            raise RuntimeError(
                "Generation stopped after provider error. "
                f"Partial results saved at {answers_path}"
            )

    write_review(
        review_path,
        report,
        case_lookup,
    )

    print()
    print(f"Saved: {answers_path}")
    print(f"Review: {review_path}")
    print(
        "Upload answers.json for manual "
        "claim-by-claim annotation."
    )


if __name__ == "__main__":
    main()
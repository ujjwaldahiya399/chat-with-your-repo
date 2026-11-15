"""Run the retrieval preflight for the micrograd faithfulness evaluation.

This version makes no Groq calls. It builds an isolated in-memory Chroma
collection using the current application chunking and embedding code, retrieves
the top five sources for every frozen question, and measures whether the
required evidence spans are available.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "Backend"

if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

import chromadb

from embed import build_github_url, embed_file
from embedding_model import EmbeddingModel, MODEL_NAME
from ingest import get_chunks


DEFAULT_SNAPSHOT = (
    ROOT
    / "evals"
    / "corpus_snapshots"
    / "2d319a78dffd90aa"
)

DEFAULT_CASES = (
    ROOT
    / "evals"
    / "micrograd_faithfulness_cases.json"
)

DEFAULT_OUTPUT = (
    ROOT
    / "evals"
    / "claim_faithfulness_preflight.json"
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


def validate_inputs(snapshot, cases_path):
    manifest = json.loads(
        (snapshot / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    cases = json.loads(
        cases_path.read_text(encoding="utf-8")
    )

    manifest_core = {
        key: value
        for key, value in manifest.items()
        if key != "snapshot_id"
    }
    canonical = json.dumps(
        manifest_core,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    if sha256(canonical) != manifest.get("snapshot_id"):
        raise ValueError("Snapshot manifest digest is invalid.")

    if cases.get("snapshot_id") != manifest["snapshot_id"]:
        raise ValueError(
            "Cases and snapshot use different snapshot IDs."
        )

    if (
        cases.get("source_commit")
        != manifest["git_base_commit"]
    ):
        raise ValueError(
            "Cases and snapshot use different commits."
        )

    if (
        cases.get("repository_url")
        != manifest["repository_url"]
    ):
        raise ValueError(
            "Cases and snapshot use different repositories."
        )

    retrieval_files = sorted(
        record["path"]
        for record in manifest["files"]
        if record.get("retrieval_input") is True
    )

    if cases.get("retrieval_files") != retrieval_files:
        raise ValueError(
            "Cases retrieval-file list differs from manifest."
        )

    for record in manifest["files"]:
        path = snapshot / record["path"]
        data = path.read_bytes()

        if len(data) != record["size_bytes"]:
            raise ValueError(
                f"Snapshot size changed: {record['path']}"
            )

        if sha256(data) != record["sha256"]:
            raise ValueError(
                f"Snapshot content changed: {record['path']}"
            )

    questions = cases["questions"]

    if not questions:
        raise ValueError("Cases file must contain questions.")

    if (
        len({question["id"] for question in questions})
        != len(questions)
    ):
        raise ValueError("Question IDs must be unique.")

    answerable_count = sum(
        question["answerable"]
        for question in questions
    )

    if answerable_count == 0:
        raise ValueError(
            "Cases file must contain answerable questions."
        )

    if answerable_count == len(questions):
        raise ValueError(
            "Cases file must contain unanswerable questions."
        )

    for question in questions:
        if question["answerable"] != bool(
            question["evidence"]
        ):
            raise ValueError(
                f"Evidence configuration is invalid for "
                f"{question['id']}."
            )

        for evidence in question["evidence"]:
            path = snapshot / evidence["file_path"]
            lines = path.read_text(
                encoding="utf-8"
            ).splitlines(keepends=True)

            start = evidence["start_line"]
            end = evidence["end_line"]

            if not 1 <= start <= end <= len(lines):
                raise ValueError(
                    f"Invalid evidence span for "
                    f"{question['id']}: "
                    f"{evidence['file_path']}:{start}-{end}"
                )

            content = "".join(lines[start - 1:end])

            if (
                sha256(content.encode("utf-8"))
                != evidence["text_sha256"]
            ):
                raise ValueError(
                    f"Evidence text changed for "
                    f"{question['id']}: "
                    f"{evidence['file_path']}:{start}-{end}"
                )

    return manifest, cases


def source_end_line(start_line, text):
    line_count = max(1, len(text.splitlines()))
    return start_line + line_count - 1


def evidence_is_covered(evidence, sources):
    required = set(
        range(
            evidence["start_line"],
            evidence["end_line"] + 1,
        )
    )
    covered = set()

    for source in sources:
        if source["file_path"] != evidence["file_path"]:
            continue

        overlap_start = max(
            evidence["start_line"],
            source["start_line"],
        )
        overlap_end = min(
            evidence["end_line"],
            source["end_line"],
        )

        if overlap_start <= overlap_end:
            covered.update(
                range(overlap_start, overlap_end + 1)
            )

    return required.issubset(covered)


def build_isolated_index(snapshot, manifest):
    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        name="micrograd-faithfulness-preflight"
    )
    model = EmbeddingModel()

    retrieval_records = [
        record
        for record in manifest["files"]
        if record.get("retrieval_input") is True
    ]

    expected_chunks = 0

    for record in retrieval_records:
        path = snapshot / record["path"]
        expected_chunks += len(get_chunks(str(path)))

        embed_file(
            path,
            model=model,
            repo_root=snapshot,
            repo_url=manifest["repository_url"],
            branch=manifest["git_base_commit"],
            collection=collection,
        )

    actual_chunks = collection.count()

    if actual_chunks != expected_chunks:
        raise ValueError(
            f"Expected {expected_chunks} indexed chunks, "
            f"found {actual_chunks}."
        )

    return client, collection, model, actual_chunks


def retrieve_question(
    question,
    collection,
    model,
    repository_url,
    top_k,
):
    embedding = model.encode(
        [question["question"]]
    ).tolist()

    result = collection.query(
        query_embeddings=embedding,
        n_results=top_k,
        where={"repo_url": repository_url},
        include=[
            "documents",
            "metadatas",
            "distances",
        ],
    )

    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    if not (
        len(ids)
        == len(documents)
        == len(metadatas)
        == len(distances)
    ):
        raise ValueError(
            f"Inconsistent retrieval result for "
            f"{question['id']}."
        )

    sources = []

    for index, (
        chunk_id,
        content,
        metadata,
        distance,
    ) in enumerate(
        zip(
            ids,
            documents,
            metadatas,
            distances,
        ),
        start=1,
    ):
        start_line = metadata["start_line"]

        sources.append(
            {
                "source_number": index,
                "chunk_id": chunk_id,
                "file_path": metadata["file_path"],
                "start_line": start_line,
                "end_line": source_end_line(
                    start_line,
                    content,
                ),
                "repo_url": metadata["repo_url"],
                "branch": metadata["branch"],
                "github_url": build_github_url(metadata),
                "distance": distance,
                "text": content,
            }
        )

    labeled_context = "\n\n".join(
        (
            f"[Source {source['source_number']}: "
            f"{source['file_path']}:"
            f"{source['start_line']}]\n"
            f"{source['text']}"
        )
        for source in sources
    )

    evidence_results = [
        {
            **evidence,
            "retrieved": evidence_is_covered(
                evidence,
                sources,
            ),
        }
        for evidence in question["evidence"]
    ]

    all_required_evidence_retrieved = (
        all(
            item["retrieved"]
            for item in evidence_results
        )
        if question["answerable"]
        else None
    )

    return {
        "question_id": question["id"],
        "case_kind": question["case_kind"],
        "answerable": question["answerable"],
        "question": question["question"],
        "sources": sources,
        "labeled_context": labeled_context,
        "evidence_results": evidence_results,
        "all_required_evidence_retrieved": (
            all_required_evidence_retrieved
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=DEFAULT_SNAPSHOT,
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=DEFAULT_CASES,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
    )
    args = parser.parse_args()

    if args.top_k != 5:
        raise ValueError(
            "This baseline is locked to the application's "
            "default top_k=5."
        )

    snapshot = args.snapshot.resolve()
    cases_path = args.cases.resolve()
    output = args.output.resolve()

    manifest, cases = validate_inputs(
        snapshot,
        cases_path,
    )

    client, collection, model, chunk_count = (
        build_isolated_index(
            snapshot,
            manifest,
        )
    )

    try:
        rows = [
            retrieve_question(
                question,
                collection,
                model,
                manifest["repository_url"],
                args.top_k,
            )
            for question in cases["questions"]
        ]
    finally:
        client.clear_system_cache()

    fully_covered = sum(
        row["all_required_evidence_retrieved"] is True
        for row in rows
    )
    answerable_count = sum(
        row["answerable"]
        for row in rows
    )
    unanswerable_count = len(rows) - answerable_count

    report = {
        "purpose": cases["purpose"],
        "mode": "retrieval_preflight_no_generation",
        "repository_url": manifest["repository_url"],
        "source_commit": manifest["git_base_commit"],
        "snapshot_id": manifest["snapshot_id"],
        "cases_sha256": file_sha256(cases_path),
        "application_code_sha256": {
            "embed.py": file_sha256(
                BACKEND / "embed.py"
            ),
            "embedding_model.py": file_sha256(
                BACKEND / "embedding_model.py"
            ),
            "ingest.py": file_sha256(
                BACKEND / "ingest.py"
            ),
            "main.py": file_sha256(
                BACKEND / "main.py"
            ),
            "query.py": file_sha256(
                BACKEND / "query.py"
            ),
        },
        "retrieval_configuration": {
            "embedding_model": MODEL_NAME,
            "top_k": args.top_k,
            "repository_filter": manifest["repository_url"],
            "collection_type": "ephemeral",
            "indexed_chunks": chunk_count,
        },
        "summary": {
            "questions": len(rows),
            "answerable_questions": answerable_count,
            "unanswerable_questions": unanswerable_count,
            "answerable_with_all_required_evidence": (
                fully_covered
            ),
        },
        "questions": rows,
        "limitations": (
            "This preflight measures whether the labelled "
            "evidence spans are present in top-five retrieval. "
            "It does not measure answer correctness, claim "
            "faithfulness, citation support, or abstention."
        ),
    }

    save_json(output, report)

    print("Retrieval preflight completed.")
    print(f"Indexed chunks: {chunk_count}")
    print(f"Questions: {len(rows)}")
    print(
        "Answerable with all required evidence:",
        f"{fully_covered}/{answerable_count}",
    )
    print(f"Saved: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
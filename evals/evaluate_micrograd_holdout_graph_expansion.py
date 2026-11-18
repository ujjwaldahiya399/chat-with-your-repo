"""Evaluate AST relationship expansion on the frozen micrograd holdout.

The experiment reproduces semantic top-five retrieval, then expands each seed
through references to top-level Python definitions. Related definitions are
inserted breadth-first up to two hops while the final context remains fixed at
five full chunks. Evidence labels are used only after selection for scoring.
No generation API calls are made and application retrieval is unchanged.
"""

import argparse
import ast
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import run_claim_faithfulness as baseline


FINAL_K = 5
MAX_RELATION_DEPTH = 2

DEFAULT_CASES = (
    baseline.ROOT
    / "evals"
    / "micrograd_faithfulness_holdout_cases.json"
)
DEFAULT_BASELINE_PREFLIGHT = (
    baseline.ROOT
    / "evals"
    / "claim_faithfulness_holdout_preflight.json"
)
DEFAULT_OUTPUT = (
    baseline.ROOT
    / "evals"
    / "claim_faithfulness_holdout_graph_expanded_preflight.json"
)


def canonical_digest(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def chunk_id(manifest, file_path, start_line):
    return (
        f"{manifest['repository_url']}@"
        f"{manifest['git_base_commit']}:"
        f"{file_path}:{start_line}"
    )


def build_chunk_records(snapshot, manifest):
    records = {}

    for file_record in manifest["files"]:
        if file_record.get("retrieval_input") is not True:
            continue

        file_path = file_record["path"]
        path = snapshot / file_path

        for start_line, text in baseline.get_chunks(str(path)):
            key = chunk_id(manifest, file_path, start_line)
            if key in records:
                raise ValueError(f"Duplicate chunk ID: {key}")

            metadata = {
                "file_path": file_path,
                "start_line": start_line,
                "repo_url": manifest["repository_url"],
                "branch": manifest["git_base_commit"],
            }
            records[key] = {
                "chunk_id": key,
                "text": text,
                "metadata": metadata,
                "end_line": baseline.source_end_line(
                    start_line,
                    text,
                ),
            }

    return records


def parse_python(text, chunk_key):
    try:
        return ast.parse(text)
    except SyntaxError as error:
        raise ValueError(
            f"Indexed Python chunk is not parseable: {chunk_key}"
        ) from error


def build_relationship_graph(records):
    definitions = {}

    for key, record in records.items():
        if not record["metadata"]["file_path"].endswith(".py"):
            continue

        tree = parse_python(record["text"], key)
        for node in tree.body:
            if isinstance(
                node,
                (
                    ast.ClassDef,
                    ast.FunctionDef,
                    ast.AsyncFunctionDef,
                ),
            ):
                definitions.setdefault(node.name, []).append(key)

    graph = {key: [] for key in records}

    for key, record in records.items():
        if not record["metadata"]["file_path"].endswith(".py"):
            continue

        tree = parse_python(record["text"], key)
        references = sorted(
            (
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Name)
                and isinstance(node.ctx, ast.Load)
                and node.id in definitions
            ),
            key=lambda node: (
                node.lineno,
                node.col_offset,
                node.id,
            ),
        )

        for reference in references:
            for target in definitions[reference.id]:
                if target != key and target not in graph[key]:
                    graph[key].append(target)

    return definitions, graph


def validate_baseline_preflight(
    path,
    manifest,
    cases_path,
    cases,
    records,
):
    report = json.loads(path.read_text(encoding="utf-8"))

    if (
        report.get("mode") != "retrieval_preflight_no_generation"
        or report.get("snapshot_id") != manifest["snapshot_id"]
        or report.get("source_commit") != manifest["git_base_commit"]
        or report.get("cases_sha256")
        != baseline.file_sha256(cases_path)
    ):
        raise ValueError("Frozen baseline preflight identity differs.")

    expected_ids = [question["id"] for question in cases["questions"]]
    actual_ids = [row["question_id"] for row in report["questions"]]
    if actual_ids != expected_ids:
        raise ValueError("Frozen baseline question order differs.")

    for row in report["questions"]:
        if len(row["sources"]) != FINAL_K:
            raise ValueError("Frozen baseline does not contain five sources.")

        for source in row["sources"]:
            key = source["chunk_id"]
            if key not in records:
                raise ValueError(f"Unknown frozen chunk: {key}")
            record = records[key]
            if (
                source["text"] != record["text"]
                or source["file_path"]
                != record["metadata"]["file_path"]
                or source["start_line"]
                != record["metadata"]["start_line"]
                or source["end_line"] != record["end_line"]
            ):
                raise ValueError(f"Frozen chunk content differs: {key}")

    return report


def expand_semantic_sources(semantic_sources, graph):
    selected = []
    selection = {}

    for semantic_rank, source in enumerate(semantic_sources, start=1):
        if len(selected) >= FINAL_K:
            break

        seed = source["chunk_id"]
        if seed not in selection:
            selected.append(seed)
            selection[seed] = {
                "selection_origin": "semantic",
                "semantic_rank": semantic_rank,
                "relationship_depth": 0,
                "relationship_root": seed,
                "relationship_path": [seed],
            }

        queue = [(seed, 0, [seed])]
        queue_index = 0

        while queue_index < len(queue) and len(selected) < FINAL_K:
            current, depth, path = queue[queue_index]
            queue_index += 1

            if depth >= MAX_RELATION_DEPTH:
                continue

            for target in graph[current]:
                if target in selection:
                    continue

                target_path = path + [target]
                selected.append(target)
                selection[target] = {
                    "selection_origin": "ast_relationship",
                    "semantic_rank": None,
                    "relationship_depth": depth + 1,
                    "relationship_root": seed,
                    "relationship_path": target_path,
                }
                queue.append((target, depth + 1, target_path))

                if len(selected) >= FINAL_K:
                    break

    if len(selected) != FINAL_K:
        raise ValueError(
            f"Expected {FINAL_K} selected chunks, found {len(selected)}."
        )

    return selected, selection


def make_sources(selected_ids, selection, semantic_sources, records):
    semantic_lookup = {
        source["chunk_id"]: source for source in semantic_sources
    }
    sources = []

    for source_number, key in enumerate(selected_ids, start=1):
        record = records[key]
        metadata = record["metadata"]

        if key in semantic_lookup:
            source = copy.deepcopy(semantic_lookup[key])
        else:
            source = {
                "chunk_id": key,
                "file_path": metadata["file_path"],
                "start_line": metadata["start_line"],
                "end_line": record["end_line"],
                "repo_url": metadata["repo_url"],
                "branch": metadata["branch"],
                "github_url": baseline.build_github_url(metadata),
                "distance": None,
                "text": record["text"],
            }

        source["source_number"] = source_number
        source.update(selection[key])
        sources.append(source)

    return sources


def labeled_context(sources):
    return "\n\n".join(
        (
            f"[Source {source['source_number']}: "
            f"{source['file_path']}:{source['start_line']}]\n"
            f"{source['text']}"
        )
        for source in sources
    )


def score_coverage(question, sources):
    evidence_results = [
        {
            **evidence,
            "retrieved": baseline.evidence_is_covered(evidence, sources),
        }
        for evidence in question["evidence"]
    ]
    all_required = (
        all(item["retrieved"] for item in evidence_results)
        if question["answerable"]
        else None
    )
    return evidence_results, all_required


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=baseline.DEFAULT_SNAPSHOT,
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument(
        "--baseline-preflight",
        type=Path,
        default=DEFAULT_BASELINE_PREFLIGHT,
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    snapshot = args.snapshot.resolve()
    cases_path = args.cases.resolve()
    baseline_path = args.baseline_preflight.resolve()
    output = args.output.resolve()

    manifest, cases = baseline.validate_inputs(snapshot, cases_path)
    records = build_chunk_records(snapshot, manifest)
    definitions, graph = build_relationship_graph(records)
    frozen_baseline = validate_baseline_preflight(
        baseline_path,
        manifest,
        cases_path,
        cases,
        records,
    )
    frozen_rows = {
        row["question_id"]: row
        for row in frozen_baseline["questions"]
    }

    client, collection, model, chunk_count = baseline.build_isolated_index(
        snapshot,
        manifest,
    )

    if chunk_count != len(records):
        raise ValueError("Index and AST record counts differ.")

    try:
        fresh_rows = [
            baseline.retrieve_question(
                question,
                collection,
                model,
                manifest["repository_url"],
                FINAL_K,
            )
            for question in cases["questions"]
        ]
    finally:
        client.clear_system_cache()

    rows = []
    for question, fresh_row in zip(cases["questions"], fresh_rows):
        saved_row = frozen_rows[question["id"]]
        selected_ids, selection = expand_semantic_sources(
            fresh_row["sources"],
            graph,
        )
        expanded_sources = make_sources(
            selected_ids,
            selection,
            fresh_row["sources"],
            records,
        )
        evidence_results, expanded_full = score_coverage(
            question,
            expanded_sources,
        )
        _, saved_full = score_coverage(question, saved_row["sources"])
        _, fresh_full = score_coverage(question, fresh_row["sources"])

        saved_ids = [
            source["chunk_id"] for source in saved_row["sources"]
        ]
        fresh_ids = [
            source["chunk_id"] for source in fresh_row["sources"]
        ]

        rows.append(
            {
                "question_id": question["id"],
                "case_kind": question["case_kind"],
                "answerable": question["answerable"],
                "question": question["question"],
                "sources": expanded_sources,
                "labeled_context": labeled_context(expanded_sources),
                "evidence_results": evidence_results,
                "all_required_evidence_retrieved": expanded_full,
                "retrieval_comparison": {
                    "saved_semantic5_fully_covered": saved_full,
                    "fresh_semantic5_fully_covered": fresh_full,
                    "graph_expanded5_fully_covered": expanded_full,
                    "saved_order_matches_fresh": saved_ids == fresh_ids,
                    "saved_semantic5": saved_ids,
                    "fresh_semantic5": fresh_ids,
                    "graph_expanded5": selected_ids,
                },
            }
        )

        print(
            f"{question['id']}: "
            f"saved={saved_full} "
            f"fresh={fresh_full} "
            f"expanded={expanded_full}",
            flush=True,
        )

    answerable_rows = [row for row in rows if row["answerable"]]

    def fully_covered(field):
        return sum(
            row["retrieval_comparison"][field] is True
            for row in answerable_rows
        )

    summary = {
        "questions": len(rows),
        "answerable_questions": len(answerable_rows),
        "unanswerable_questions": len(rows) - len(answerable_rows),
        "answerable_with_all_required_evidence": sum(
            row["all_required_evidence_retrieved"] is True
            for row in answerable_rows
        ),
        "comparison": {
            "saved_semantic5_fully_covered": fully_covered(
                "saved_semantic5_fully_covered"
            ),
            "fresh_semantic5_fully_covered": fully_covered(
                "fresh_semantic5_fully_covered"
            ),
            "graph_expanded5_fully_covered": fully_covered(
                "graph_expanded5_fully_covered"
            ),
        },
    }

    graph_payload = {
        key: targets for key, targets in graph.items() if targets
    }
    definition_payload = {
        name: keys for name, keys in sorted(definitions.items())
    }

    report = {
        "purpose": "micrograd_holdout_ast_relationship_expansion_experiment",
        "mode": "retrieval_preflight_no_generation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository_url": manifest["repository_url"],
        "source_commit": manifest["git_base_commit"],
        "snapshot_id": manifest["snapshot_id"],
        "cases_sha256": baseline.file_sha256(cases_path),
        "baseline_preflight_sha256": baseline.file_sha256(baseline_path),
        "runner_sha256": baseline.file_sha256(__file__),
        "application_code_sha256": {
            name: baseline.file_sha256(baseline.BACKEND / name)
            for name in (
                "embed.py",
                "embedding_model.py",
                "ingest.py",
                "main.py",
                "query.py",
            )
        },
        "retrieval_configuration": {
            "embedding_model": baseline.MODEL_NAME,
            "semantic_seed_count": FINAL_K,
            "final_k": FINAL_K,
            "maximum_relationship_depth": MAX_RELATION_DEPTH,
            "relationship_type": (
                "ast.Name Load references to indexed top-level Python "
                "class and function definitions"
            ),
            "selection_order": (
                "semantic seeds in rank order; breadth-first related "
                "definitions inserted after each seed"
            ),
            "repository_filter": manifest["repository_url"],
            "collection_type": "ephemeral",
            "indexed_chunks": chunk_count,
            "defined_symbols": definition_payload,
            "relationship_graph": graph_payload,
            "relationship_graph_sha256": canonical_digest(graph_payload),
        },
        "summary": summary,
        "questions": rows,
        "limitations": (
            "Retrieval-only post-hoc experiment on the frozen holdout. "
            "Relationship selection does not use evidence labels, but the "
            "strategy was chosen after inspecting two holdout failures. It "
            "recognizes only direct ast.Name references to top-level Python "
            "definitions, follows at most two hops, and can displace semantic "
            "chunks that may matter for other questions. Unanswerable-answer "
            "behavior and production-scale latency are not measured."
        ),
    }

    baseline.save_json(output, report)

    comparison = summary["comparison"]
    print()
    print("Retrieval comparison on answerable questions:")
    print(
        "Saved semantic top-five:",
        f"{comparison['saved_semantic5_fully_covered']}/"
        f"{len(answerable_rows)}",
    )
    print(
        "Fresh semantic top-five:",
        f"{comparison['fresh_semantic5_fully_covered']}/"
        f"{len(answerable_rows)}",
    )
    print(
        "Graph-expanded top-five:",
        f"{comparison['graph_expanded5_fully_covered']}/"
        f"{len(answerable_rows)}",
    )
    print("Saved:", output)


if __name__ == "__main__":
    main()

"""Compare frozen semantic top-five retrieval with cross-encoder reranking.

The experiment builds the same isolated micrograd index as the faithfulness
preflight, retrieves up to 20 semantic candidates, scores deterministic
overlapping windows with the existing cross-encoder, and returns the original
full chunks for the best five candidates. It makes no generation API calls and
does not modify the application index.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
from time import perf_counter

import run_claim_faithfulness as baseline
from evaluate_semantic_reranking import (
    FINAL_K,
    PAIR_LIMIT,
    RERANKER,
    rank_scores,
    weight_digest,
)


CANDIDATE_DEPTH = 20
OVERLAP_LINES = 8
EXPECTED_PACKAGES = {
    "sentence-transformers": "5.6.0",
    "transformers": "5.22.0",
    "torch": "2.13.0",
    "chromadb": "1.5.9",
}
EXPECTED_WEIGHTS_SHA256 = (
    "e46232fbfc2760adb858264ed7d8fb56661fde6720f62231fdf578041e300409"
)
EXPECTED_VOCABULARY_SHA256 = (
    "373b066fa218cddd79eec036c10bec16e439381f94d6845bf3106ef8194cff36"
)

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
    / "claim_faithfulness_holdout_reranked_preflight.json"
)


def canonical_digest(value):
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def pair_token_count(tokenizer, question, text):
    return len(
        tokenizer(
            question,
            text,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
    )


def build_rerank_windows(question, text, tokenizer):
    """Return complete-line windows whose question pair fits PAIR_LIMIT."""
    if pair_token_count(tokenizer, question, text) <= PAIR_LIMIT:
        return [
            {
                "start_line_offset": 0,
                "end_line_offset": max(0, len(text.splitlines()) - 1),
                "pair_tokens": pair_token_count(tokenizer, question, text),
                "text": text,
            }
        ]

    lines = text.splitlines(keepends=True)
    windows = []
    start = 0

    while start < len(lines):
        end = start
        accepted_text = None
        accepted_tokens = None

        while end < len(lines):
            candidate_text = "".join(lines[start : end + 1])
            token_count = pair_token_count(
                tokenizer,
                question,
                candidate_text,
            )

            if token_count > PAIR_LIMIT:
                break

            accepted_text = candidate_text
            accepted_tokens = token_count
            end += 1

        if accepted_text is None:
            raise ValueError(
                "A single source line exceeds the reranker pair limit."
            )

        windows.append(
            {
                "start_line_offset": start,
                "end_line_offset": end - 1,
                "pair_tokens": accepted_tokens,
                "text": accepted_text,
            }
        )

        if end == len(lines):
            break

        start = max(start + 1, end - OVERLAP_LINES)

    if "".join(lines) != text:
        raise ValueError("Window input changed source text.")

    covered_lines = set()
    for window in windows:
        covered_lines.update(
            range(
                window["start_line_offset"],
                window["end_line_offset"] + 1,
            )
        )

    if covered_lines != set(range(len(lines))):
        raise ValueError("Reranker windows do not cover the full chunk.")

    return windows


def retrieve_candidates(
    question,
    collection,
    model,
    repository_url,
    candidate_depth,
):
    started = perf_counter()
    embedding = model.encode([question]).tolist()
    result = collection.query(
        query_embeddings=embedding,
        n_results=min(candidate_depth, collection.count()),
        where={"repo_url": repository_url},
        include=["documents", "metadatas", "distances"],
    )
    search_seconds = perf_counter() - started

    ids = result["ids"][0]
    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    if not (
        ids
        and len(ids)
        == len(documents)
        == len(metadatas)
        == len(distances)
        and len(ids) == len(set(ids))
    ):
        raise ValueError("Candidate retrieval returned inconsistent results.")

    candidates = []
    for chunk_id, text, metadata, distance in zip(
        ids,
        documents,
        metadatas,
        distances,
    ):
        candidates.append(
            {
                "chunk_id": chunk_id,
                "text": text,
                "metadata": metadata,
                "distance": float(distance),
            }
        )

    return candidates, search_seconds


def candidate_to_source(candidate, source_number, score=None, window=None):
    metadata = candidate["metadata"]
    start_line = metadata["start_line"]
    source = {
        "source_number": source_number,
        "chunk_id": candidate["chunk_id"],
        "file_path": metadata["file_path"],
        "start_line": start_line,
        "end_line": baseline.source_end_line(
            start_line,
            candidate["text"],
        ),
        "repo_url": metadata["repo_url"],
        "branch": metadata["branch"],
        "github_url": baseline.build_github_url(metadata),
        "distance": candidate["distance"],
        "text": candidate["text"],
    }

    if score is not None:
        source["reranker_score"] = float(score)
    if window is not None:
        source["reranker_winning_window"] = {
            key: value
            for key, value in window.items()
            if key != "text"
        }

    return source


def make_sources(candidates, selected_indices, scores, winning_windows):
    return [
        candidate_to_source(
            candidates[candidate_index],
            source_number,
            scores[candidate_index],
            winning_windows[candidate_index],
        )
        for source_number, candidate_index in enumerate(
            selected_indices,
            start=1,
        )
    ]


def make_plain_sources(candidates):
    return [
        candidate_to_source(candidate, source_number)
        for source_number, candidate in enumerate(candidates, start=1)
    ]


def coverage(question, sources):
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


def labeled_context(sources):
    return "\n\n".join(
        (
            f"[Source {source['source_number']}: "
            f"{source['file_path']}:{source['start_line']}]\n"
            f"{source['text']}"
        )
        for source in sources
    )


def validate_baseline_preflight(path, manifest, cases_path, cases):
    report = json.loads(path.read_text(encoding="utf-8"))
    if (
        report.get("mode") != "retrieval_preflight_no_generation"
        or report.get("snapshot_id") != manifest["snapshot_id"]
        or report.get("source_commit") != manifest["git_base_commit"]
        or report.get("cases_sha256") != baseline.file_sha256(cases_path)
    ):
        raise ValueError("Frozen baseline preflight identity differs.")

    expected_ids = [question["id"] for question in cases["questions"]]
    actual_ids = [row["question_id"] for row in report["questions"]]
    if actual_ids != expected_ids:
        raise ValueError("Frozen baseline question order differs.")
    return report


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
    parser.add_argument(
        "--candidate-depth",
        type=int,
        default=CANDIDATE_DEPTH,
    )
    args = parser.parse_args()

    if args.candidate_depth < FINAL_K:
        raise ValueError("Candidate depth must be at least final_k.")

    packages = {name: version(name) for name in EXPECTED_PACKAGES}
    if packages != EXPECTED_PACKAGES:
        raise ValueError(
            f"Package versions differ: expected {EXPECTED_PACKAGES}, "
            f"found {packages}"
        )

    snapshot = args.snapshot.resolve()
    cases_path = args.cases.resolve()
    baseline_path = args.baseline_preflight.resolve()
    output = args.output.resolve()

    manifest, cases = baseline.validate_inputs(snapshot, cases_path)
    frozen_baseline = validate_baseline_preflight(
        baseline_path,
        manifest,
        cases_path,
        cases,
    )
    frozen_rows = {
        row["question_id"]: row
        for row in frozen_baseline["questions"]
    }

    client, collection, model, chunk_count = baseline.build_isolated_index(
        snapshot,
        manifest,
    )

    try:
        candidate_rows = []
        for question in cases["questions"]:
            candidates, search_seconds = retrieve_candidates(
                question["question"],
                collection,
                model,
                manifest["repository_url"],
                args.candidate_depth,
            )
            candidate_rows.append(
                {
                    "question": question,
                    "candidates": candidates,
                    "search_seconds": search_seconds,
                }
            )

        from sentence_transformers import CrossEncoder

        print("Loading and verifying the reranker once...", flush=True)
        reranker = CrossEncoder(RERANKER, max_length=PAIR_LIMIT)
        weights_sha256 = weight_digest(reranker)
        vocabulary_sha256 = canonical_digest(
            reranker.tokenizer.get_vocab()
        )
        if weights_sha256 != EXPECTED_WEIGHTS_SHA256:
            raise ValueError("Reranker weights differ from the frozen model.")
        if vocabulary_sha256 != EXPECTED_VOCABULARY_SHA256:
            raise ValueError("Reranker vocabulary differs from the frozen model.")
        if type(reranker.tokenizer).__name__ != "BertTokenizer":
            raise ValueError("Reranker tokenizer class differs.")

        reranker_identity = {
            "model_name": RERANKER,
            "weights_sha256": weights_sha256,
            "max_length": PAIR_LIMIT,
            "tokenizer_class": type(reranker.tokenizer).__name__,
            "vocabulary_sha256": vocabulary_sha256,
            "device": str(getattr(reranker, "device", "unknown")),
        }

        rows = []
        for item in candidate_rows:
            question = item["question"]
            candidates = item["candidates"]
            pairs = []
            pair_owners = []
            candidate_windows = []

            for candidate_index, candidate in enumerate(candidates):
                windows = build_rerank_windows(
                    question["question"],
                    candidate["text"],
                    reranker.tokenizer,
                )
                candidate_windows.append(windows)
                for window_index, window in enumerate(windows):
                    if window["pair_tokens"] > PAIR_LIMIT:
                        raise ValueError("Reranker window exceeds pair limit.")
                    pairs.append((question["question"], window["text"]))
                    pair_owners.append((candidate_index, window_index))

            started = perf_counter()
            window_scores = reranker.predict(
                pairs,
                batch_size=16,
                show_progress_bar=False,
            )
            rerank_seconds = perf_counter() - started

            scores = [float("-inf")] * len(candidates)
            winning_windows = [None] * len(candidates)
            for score, (candidate_index, window_index) in zip(
                window_scores,
                pair_owners,
            ):
                numeric_score = float(score)
                if numeric_score > scores[candidate_index]:
                    scores[candidate_index] = numeric_score
                    winning_windows[candidate_index] = candidate_windows[
                        candidate_index
                    ][window_index]

            if any(window is None for window in winning_windows):
                raise ValueError("A candidate has no reranker score.")

            candidate_ids = [
                candidate["chunk_id"] for candidate in candidates
            ]
            selected_indices = rank_scores(
                candidate_ids,
                scores,
                FINAL_K,
            )
            reranked_sources = make_sources(
                candidates,
                selected_indices,
                scores,
                winning_windows,
            )
            fresh_semantic_sources = make_plain_sources(
                candidates[:FINAL_K]
            )
            pool_sources = make_plain_sources(candidates)
            saved_sources = frozen_rows[question["id"]]["sources"]

            evidence_results, reranked_full = coverage(
                question,
                reranked_sources,
            )
            _, fresh_full = coverage(question, fresh_semantic_sources)
            _, pool_full = coverage(question, pool_sources)
            _, saved_full = coverage(question, saved_sources)

            rows.append(
                {
                    "question_id": question["id"],
                    "case_kind": question["case_kind"],
                    "answerable": question["answerable"],
                    "question": question["question"],
                    "sources": reranked_sources,
                    "labeled_context": labeled_context(reranked_sources),
                    "evidence_results": evidence_results,
                    "all_required_evidence_retrieved": reranked_full,
                    "retrieval_comparison": {
                        "saved_semantic5_fully_covered": saved_full,
                        "fresh_semantic5_fully_covered": fresh_full,
                        "reranked5_fully_covered": reranked_full,
                        "candidate_pool_fully_covered": pool_full,
                        "saved_order_matches_fresh": [
                            source["chunk_id"] for source in saved_sources
                        ]
                        == candidate_ids[:FINAL_K],
                        "saved_semantic5": [
                            source["chunk_id"] for source in saved_sources
                        ],
                        "fresh_semantic5": candidate_ids[:FINAL_K],
                        "reranked5": [
                            candidates[index]["chunk_id"]
                            for index in selected_indices
                        ],
                        "candidate_pool": candidate_ids,
                        "search_seconds": item["search_seconds"],
                        "rerank_seconds": rerank_seconds,
                        "reranker_scores": {
                            chunk_id: score
                            for chunk_id, score in zip(candidate_ids, scores)
                        },
                        "reranker_window_counts": {
                            candidate["chunk_id"]: len(windows)
                            for candidate, windows in zip(
                                candidates,
                                candidate_windows,
                            )
                        },
                    },
                }
            )

            print(
                f"{question['id']}: "
                f"saved={saved_full} "
                f"fresh={fresh_full} "
                f"reranked={reranked_full} "
                f"pool={pool_full}",
                flush=True,
            )
    finally:
        client.clear_system_cache()

    answerable_rows = [row for row in rows if row["answerable"]]

    def fully_covered(method):
        return sum(
            row["retrieval_comparison"][method] is True
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
            "reranked5_fully_covered": fully_covered(
                "reranked5_fully_covered"
            ),
            "candidate_pool_fully_covered": fully_covered(
                "candidate_pool_fully_covered"
            ),
        },
    }

    report = {
        "purpose": "micrograd_holdout_reranked_retrieval_experiment",
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
            "candidate_depth_requested": args.candidate_depth,
            "candidate_depth_used": min(
                args.candidate_depth,
                chunk_count,
            ),
            "final_k": FINAL_K,
            "repository_filter": manifest["repository_url"],
            "collection_type": "ephemeral",
            "indexed_chunks": chunk_count,
            "reranker": reranker_identity,
            "reranker_windowing": {
                "pair_token_limit": PAIR_LIMIT,
                "overlap_lines": OVERLAP_LINES,
                "chunk_score": "maximum window score",
                "returned_content": "original full chunk",
            },
            "packages": packages,
        },
        "summary": summary,
        "questions": rows,
        "limitations": (
            "Retrieval-only experiment on the frozen holdout. The cross-encoder "
            "reranks a semantic candidate pool and returns five original chunks. "
            "Labels are used only after ranking to score evidence coverage. No "
            "answers are generated. The micrograd corpus has fewer than the "
            "requested 20 candidates, so this run reranks the complete 13-chunk "
            "corpus. Maximum-window aggregation gives longer chunks more scoring "
            "opportunities, and production-scale candidate latency is not measured."
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
        "Reranked top-five:",
        f"{comparison['reranked5_fully_covered']}/"
        f"{len(answerable_rows)}",
    )
    print(
        "Candidate pool:",
        f"{comparison['candidate_pool_fully_covered']}/"
        f"{len(answerable_rows)}",
    )
    print("Saved:", output)


if __name__ == "__main__":
    main()

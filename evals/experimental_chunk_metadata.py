"""Attach source and parent identities to frozen experimental chunks.

This audit reads a snapshot and builds records in memory. It does not encode
embeddings, write files, or open a database. Keep this file beside
experimental_chunker_v2.py. The splitter and embedding-input format are reused.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path, PurePosixPath
import re

import experimental_chunker_v2 as chunker


def _digest(value):
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def canonical_repo_id(value):
    """Accept a GitHub host/owner/repository identity, without credentials."""
    value = value.strip().lower().removesuffix(".git")
    if not re.fullmatch(r"github\.com/[a-z0-9_.-]+/[a-z0-9_.-]+", value):
        raise ValueError("repo_id must have the form github.com/owner/repository")
    if any(part in {".", ".."} for part in value.split("/")):
        raise ValueError("Invalid repository identity")
    return value


def _verified_files(manifest, sources):
    core = {key: value for key, value in manifest.items() if key != "snapshot_id"}
    if manifest.get("schema_version") != 1 or _digest(core) != manifest.get("snapshot_id"):
        raise ValueError("Invalid snapshot manifest identity")
    if not re.fullmatch(r"[0-9a-f]{40}", manifest.get("git_base_commit", "")):
        raise ValueError("Expected a full Git base commit")
    files = {}
    for item in manifest["files"]:
        path = item["path"]
        relative = PurePosixPath(path)
        if (relative.is_absolute() or ".." in relative.parts
                or str(relative) != path or "\\" in path or path in files):
            raise ValueError(f"Invalid or duplicate source path: {path}")
        if item["origin"] == "git":
            if item.get("source_commit") != manifest["git_base_commit"]:
                raise ValueError(f"Source commit disagrees with snapshot: {path}")
        elif item["origin"] != "local" or item.get("source_commit") is not None:
            raise ValueError(f"Invalid source provenance: {path}")
        files[path] = item
    if not files or set(files) != set(sources):
        raise ValueError("Sources must match the complete snapshot file list")
    for path, source in sources.items():
        raw = source.encode("utf-8")
        if (len(raw) != files[path]["size_bytes"]
                or hashlib.sha256(raw).hexdigest() != files[path]["sha256"]):
            raise ValueError(f"Source bytes disagree with manifest: {path}")
    return files


def _parents(source, file_path):
    """Use the splitter's original top-level partitions as parents."""
    if Path(file_path).suffix.lower() == ".py":
        return chunker.partition_python_source(source)
    if not source:
        return []
    return [{
        "kind": "text", "symbol": "<document>",
        "start_line": 1, "end_line": len(source.splitlines(keepends=True)),
        "source_text": source,
    }]


def build_chunk_records(repo_id, manifest, sources, tokenizer, budget=220):
    """Return source_text, exact embedding_input, and scalar metadata per chunk.

    Parent IDs describe original top-level functions/classes, module regions,
    or complete text documents. Nested methods retain their top-level class
    parent. IDs are source identities: model/index configuration must also be
    recorded by the future index builder, in its separate experiment manifest.
    """
    repo_id = canonical_repo_id(repo_id)
    files = _verified_files(manifest, sources)
    chunker_sha256 = hashlib.sha256(Path(chunker.__file__).read_bytes()).hexdigest()
    records = []
    for file_path in sorted(sources):
        source = sources[file_path]
        file_info = files[file_path]
        parents = _parents(source, file_path)
        spans = chunker.split_source(source, file_path, tokenizer, budget)
        chunker.validate_spans(source, spans, file_path, tokenizer, budget)
        source_identity = {
            "repo_id": repo_id,
            "snapshot_id": manifest["snapshot_id"],
            "file_path": file_path,
            "file_sha256": file_info["sha256"],
        }
        for span in spans:
            if not span["source_text"].strip():
                continue
            containing = [
                parent for parent in parents
                if parent["start_line"] <= span["start_line"]
                and span["end_line"] <= parent["end_line"]
            ]
            if len(containing) != 1:
                raise ValueError(f"Chunk must belong to exactly one parent: {file_path}")
            parent = containing[0]
            if (parent["kind"], parent["symbol"]) != (span["kind"], span["symbol"]):
                raise ValueError(f"Chunk scope disagrees with its parent: {file_path}")
            parent_identity = {
                **source_identity,
                "parent_kind": parent["kind"],
                "symbol": parent["symbol"] or "<module>",
                "parent_start_line": parent["start_line"],
                "parent_end_line": parent["end_line"],
            }
            parent_id = "parent:" + _digest(parent_identity)
            chunk_id = "chunk:" + _digest({
                **source_identity,
                "start_line": span["start_line"],
                "end_line": span["end_line"],
            })
            embedding_input = chunker.build_embedding_input(span, file_path)
            metadata = {
                **parent_identity,
                "metadata_schema_version": 1,
                "parent_id": parent_id,
                "start_line": span["start_line"],
                "end_line": span["end_line"],
                "source_origin": file_info["origin"],
                "chunker_sha256": chunker_sha256,
                "token_count": chunker.count_tokens(embedding_input, tokenizer),
                "token_budget": budget,
            }
            if file_info["origin"] == "git":
                metadata["source_commit"] = file_info["source_commit"]
            # No None values: Chroma metadata uses scalar values. A local-only
            # file is identified by its snapshot/hash and has no source_commit.
            if any(type(value) not in (str, int, float, bool) for value in metadata.values()):
                raise ValueError("Metadata must contain only supported scalar values")
            records.append({
                "chunk_id": chunk_id,
                "source_text": span["source_text"],
                "embedding_input": embedding_input,
                "metadata": metadata,
            })
    if len({record["chunk_id"] for record in records}) != len(records):
        raise ValueError("Duplicate chunk IDs detected")
    return records


def run_self_tests():
    """Identity/provenance checks using a deterministic counter, not Bert."""
    def counter(text, **kwargs):
        if kwargs != {"add_special_tokens": True, "truncation": False}:
            raise AssertionError("Tokenization options changed")
        return {"input_ids": range(len(text) + 2)}

    def fixture_manifest(sources, revision="a" * 40):
        files = []
        for path, source in sorted(sources.items()):
            raw = source.encode("utf-8")
            local = path.endswith("NOTES.md")
            files.append({
                "path": path, "sha256": hashlib.sha256(raw).hexdigest(),
                "size_bytes": len(raw), "origin": "local" if local else "git",
                "source_commit": None if local else revision,
            })
        core = {"schema_version": 1, "git_base_commit": revision, "files": files}
        return {**core, "snapshot_id": _digest(core)}

    source = "@decorator\ndef work():\n" + "".join(
        f"    value_{i} = {i}\n" for i in range(18)
    )
    sources = {"Backend/a.py": source, "Backend/b.py": source,
               "Backend/NOTES.md": "# Notes\r\nनमस्ते 🌿\r\n"}
    manifest = fixture_manifest(sources)
    repo = "github.com/example/repository"

    def build(repo_id=repo, snapshot=manifest, inputs=sources, budget=150):
        return build_chunk_records(repo_id, snapshot, inputs, counter, budget)

    records = build()
    if records != build():
        raise AssertionError("IDs or records changed on an identical run")
    print("PASS: deterministic records and IDs")

    first = [r for r in records if r["metadata"]["file_path"] == "Backend/a.py"]
    if (len(first) < 2 or len({r["metadata"]["parent_id"] for r in first}) != 1
            or len({r["chunk_id"] for r in first}) != len(first)):
        raise AssertionError("Sibling identity failed")
    if any((r["metadata"]["parent_start_line"], r["metadata"]["parent_end_line"])
           != (1, 20) for r in first):
        raise AssertionError("Exact decorated parent bounds failed")
    print("PASS: shared parent, distinct children, exact parent bounds")

    second = [r for r in records if r["metadata"]["file_path"] == "Backend/b.py"]
    if first[0]["metadata"]["parent_id"] == second[0]["metadata"]["parent_id"]:
        raise AssertionError("Same-name functions in different files collided")
    other_repo = build(repo_id="github.com/example/other")
    if {r["chunk_id"] for r in records} & {r["chunk_id"] for r in other_repo}:
        raise AssertionError("Different repositories collided")
    print("PASS: repository and file identities prevent collisions")

    changed = {**sources, "Backend/NOTES.md": sources["Backend/NOTES.md"] + "New text.\n"}
    other_snapshot = build(snapshot=fixture_manifest(changed), inputs=changed)
    if ({r["chunk_id"] for r in records} & {r["chunk_id"] for r in other_snapshot}
            or {r["metadata"]["parent_id"] for r in records}
            & {r["metadata"]["parent_id"] for r in other_snapshot}):
        raise AssertionError("Different snapshots collided")
    print("PASS: snapshot identities keep editions separate")

    for record in records:
        meta = record["metadata"]
        lines = sources[meta["file_path"]].splitlines(keepends=True)
        if record["source_text"] != "".join(lines[meta["start_line"] - 1:meta["end_line"]]):
            raise AssertionError("Source reference failed")
        expected_input = (f"File: {meta['file_path']}\nScope: {meta['symbol']}\nSource:\n"
                          + record["source_text"])
        if record["embedding_input"] != expected_input or meta["token_count"] > 150:
            raise AssertionError("Embedding input or budget changed")
        if (meta["source_origin"] == "git") != ("source_commit" in meta):
            raise AssertionError("Git/local provenance failed")
    print("PASS: exact chunk references, inputs, budgets, and Git/local provenance")

    wider = build(budget=190)
    if ({r["metadata"]["parent_id"] for r in records}
            != {r["metadata"]["parent_id"] for r in wider}):
        raise AssertionError("Parent IDs depend on child packing")
    print("PASS: parent identities survive changes to child boundaries")

    for invalid_sources in [{**sources, "Backend/a.py": source + "\n"},
                            {"Backend/a.py": source}]:
        try:
            build(inputs=invalid_sources)
        except ValueError:
            pass
        else:
            raise AssertionError("Unverified or incomplete sources were accepted")
    print("PASS: changed or incomplete source inputs rejected")

    if canonical_repo_id("GITHUB.COM/Example/Repository.git") != repo:
        raise AssertionError("Repository normalization failed")
    try:
        canonical_repo_id("https://user:secret@github.com/example/repository")
    except ValueError:
        pass
    else:
        raise AssertionError("Credential-bearing URL was accepted as repo_id")
    print("PASS: canonical repository identity and invalid-URL rejection")
    print("All 8 metadata self-test groups passed (deterministic counter).")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--repo-id")
    parser.add_argument("--budget", type=int, default=220)
    args = parser.parse_args()
    if args.self_test:
        run_self_tests()
        return
    if not args.snapshot or not args.repo_id:
        parser.error("provide --snapshot DIRECTORY and --repo-id HOST/OWNER/REPOSITORY")
    try:
        repo_id = canonical_repo_id(args.repo_id)
        manifest, sources = chunker.load_source_snapshot(args.snapshot)
        _verified_files(manifest, sources)
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        if not 0 < args.budget <= model.max_seq_length:
            raise ValueError(f"budget must be between 1 and {model.max_seq_length}")
        records = build_chunk_records(repo_id, manifest, sources, model.tokenizer, args.budget)
        if not records:
            raise ValueError("The snapshot produced no indexable records")
    except (OSError, ValueError, SyntaxError, AssertionError, KeyError) as error:
        parser.exit(1, f"Metadata audit failed: {error}\n")

    counts = Counter(record["metadata"]["file_path"] for record in records)
    print(f"Repository: {repo_id}")
    print(f"Snapshot ID: {manifest['snapshot_id']}")
    for file_path in sorted(sources):
        print(f"PASS {file_path}: {counts[file_path]} metadata records")
    sample = next((r for r in records if r["metadata"]["symbol"] == "embed_file"
                   and r["metadata"]["start_line"] > r["metadata"]["parent_start_line"]), records[0])
    print("\nExample record identity and metadata:")
    print(json.dumps({"chunk_id": sample["chunk_id"], **sample["metadata"]}, indent=2))
    print(f"\nMetadata audit passed: {len(records)} unique chunks across {len(sources)} files.")
    print("No files, embeddings, or databases were written.")


if __name__ == "__main__":
    main()

"""Freeze selected files from an external Git repository.

The resulting snapshot records exact source bytes, hashes, repository URL,
commit SHA, and whether each file belongs to the retrieval corpus.
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path, PurePosixPath


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def git_bytes(root, *args):
    return subprocess.check_output(["git", *args], cwd=root)


def validate_relative_path(value):
    path = PurePosixPath(value)

    if path.is_absolute() or ".." in path.parts or str(path) in {"", "."}:
        raise ValueError(f"Unsafe repository-relative path: {value}")

    return path.as_posix()


def verify_snapshot(destination, manifest):
    saved_manifest = json.loads(
        (destination / "manifest.json").read_text(encoding="utf-8")
    )

    if saved_manifest != manifest:
        raise RuntimeError(
            "Existing snapshot manifest differs; refusing to overwrite."
        )

    for record in manifest["files"]:
        path = destination / record["path"]
        data = path.read_bytes()

        if len(data) != record["size_bytes"]:
            raise RuntimeError(
                f"Snapshot size changed: {record['path']}"
            )

        if sha256(data) != record["sha256"]:
            raise RuntimeError(
                f"Snapshot content changed: {record['path']}"
            )


def freeze_external_corpus(
    source_root,
    destination_root,
    repository_url,
    revision,
    retrieval_files,
    provenance_files,
):
    source_root = Path(source_root).resolve()
    destination_root = Path(destination_root).resolve()

    revision = git_bytes(
        source_root,
        "rev-parse",
        "--verify",
        f"{revision}^{{commit}}",
    ).decode("ascii").strip()

    status = git_bytes(
        source_root,
        "status",
        "--porcelain",
    ).decode("utf-8")

    if status:
        raise RuntimeError(
            "External source checkout contains uncommitted changes."
        )

    committed_paths = set(
        git_bytes(
            source_root,
            "ls-tree",
            "-r",
            "--name-only",
            revision,
        ).decode("utf-8").splitlines()
    )

    selected = {}

    for name in retrieval_files:
        name = validate_relative_path(name)
        selected[name] = True

    for name in provenance_files:
        name = validate_relative_path(name)

        if name in selected:
            raise ValueError(
                f"File cannot be both retrieval and provenance input: {name}"
            )

        selected[name] = False

    if not selected:
        raise ValueError("At least one file must be selected.")

    payloads = {}
    records = []

    for name in sorted(selected):
        if name not in committed_paths:
            raise RuntimeError(
                f"Selected file is absent from commit {revision}: {name}"
            )

        local_path = source_root / name

        if not local_path.is_file():
            raise FileNotFoundError(
                f"Selected local file is missing: {name}"
            )

        committed_data = git_bytes(
            source_root,
            "show",
            f"{revision}:{name}",
        )
        local_data = local_path.read_bytes()

        if local_data != committed_data:
            raise RuntimeError(
                f"Working file differs from commit {revision}: {name}"
            )

        payloads[name] = committed_data
        records.append(
            {
                "path": name,
                "sha256": sha256(committed_data),
                "size_bytes": len(committed_data),
                "origin": "git",
                "source_commit": revision,
                "retrieval_input": selected[name],
            }
        )

    core = {
        "schema_version": 2,
        "repository_url": repository_url.rstrip("/").removesuffix(".git"),
        "git_base_commit": revision,
        "files": records,
    }

    canonical = json.dumps(
        core,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    snapshot_id = sha256(canonical)
    manifest = {**core, "snapshot_id": snapshot_id}
    destination = destination_root / snapshot_id[:16]

    if destination.exists():
        verify_snapshot(destination, manifest)
        return destination, manifest, True

    destination_root.mkdir(parents=True, exist_ok=True)

    staging = Path(
        tempfile.mkdtemp(
            prefix=".staging-",
            dir=destination_root,
        )
    )

    try:
        for name, data in payloads.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)

            with target.open("xb") as stream:
                stream.write(data)

        with (staging / "manifest.json").open(
            "x",
            encoding="utf-8",
        ) as stream:
            json.dump(
                manifest,
                stream,
                indent=2,
                ensure_ascii=False,
            )
            stream.write("\n")

        verify_snapshot(staging, manifest)
        staging.rename(destination)

    finally:
        if staging.exists():
            shutil.rmtree(staging)

    return destination, manifest, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument(
        "--destination-root",
        type=Path,
        default=Path("evals/corpus_snapshots"),
    )
    parser.add_argument("--repository-url", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument(
        "--retrieval-file",
        action="append",
        default=[],
    )
    parser.add_argument(
        "--provenance-file",
        action="append",
        default=[],
    )
    args = parser.parse_args()

    try:
        destination, manifest, reused = freeze_external_corpus(
            source_root=args.source_root,
            destination_root=args.destination_root,
            repository_url=args.repository_url,
            revision=args.revision,
            retrieval_files=args.retrieval_file,
            provenance_files=args.provenance_file,
        )
    except (
        OSError,
        RuntimeError,
        subprocess.CalledProcessError,
        ValueError,
    ) as error:
        parser.exit(1, f"Snapshot not created: {error}\n")

    print(
        "Snapshot reused and verified."
        if reused
        else "Snapshot created and verified."
    )
    print(f"Snapshot: {destination}")
    print(f"Manifest: {destination / 'manifest.json'}")
    print(f"Files verified: {len(manifest['files'])}")
    print(f"Snapshot ID: {manifest['snapshot_id']}")


if __name__ == "__main__":
    main()
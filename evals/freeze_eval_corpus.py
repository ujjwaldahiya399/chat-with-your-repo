"""Freeze the eight selected evaluation inputs without rebuilding any index.

Committed inputs must match the chosen revision byte for byte. NOTES.md is
the sole explicitly allowed local-only input. Each stored file has a SHA-256
hash; the manifest distinguishes committed and local-only provenance.

Run from the project root:
    Backend/.venv/bin/python evals/freeze_eval_corpus.py --revision COMMIT
"""

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path


SOURCE_FILES = [
    "Backend/NOTES.md",
    "Backend/ask.py",
    "Backend/embed.py",
    "Backend/embed_repo.py",
    "Backend/ingest.py",
    "Backend/query.py",
    "Backend/requirements.txt",
    "Backend/test_decorator.py",
]
ALLOWED_LOCAL_ONLY = {"Backend/NOTES.md"}


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def git_bytes(root, *args):
    return subprocess.check_output(["git", *args], cwd=root)


def verify_snapshot(destination, manifest):
    saved = json.loads((destination / "manifest.json").read_text(encoding="utf-8"))
    if saved != manifest:
        raise RuntimeError("Existing snapshot manifest differs; refusing to overwrite.")
    for record in manifest["files"]:
        data = (destination / record["path"]).read_bytes()
        if len(data) != record["size_bytes"] or sha256(data) != record["sha256"]:
            raise RuntimeError(f"Snapshot content was changed: {record['path']}")


def freeze_corpus(root, revision):
    root = Path(root).resolve()
    revision = git_bytes(
        root, "rev-parse", "--verify", f"{revision}^{{commit}}"
    ).decode("ascii").strip()
    committed_paths = set(git_bytes(
        root, "ls-tree", "-r", "--name-only", revision
    ).decode("utf-8").splitlines())

    payloads = {}
    records = []
    for name in SOURCE_FILES:
        local_path = root / name
        if not local_path.is_file():
            raise FileNotFoundError(f"Required local input is missing: {name}")
        local = local_path.read_bytes()

        if name in committed_paths:
            committed = git_bytes(root, "show", f"{revision}:{name}")
            if local != committed:
                raise RuntimeError(
                    f"Working file differs from the selected commit: {name}. "
                    "Review the difference before freezing this corpus."
                )
            data = committed
            origin = "git"
            source_commit = revision
        elif name in ALLOWED_LOCAL_ONLY:
            data = local
            origin = "local"
            source_commit = None
        else:
            raise RuntimeError(
                f"Required input is absent from the selected commit: {name}"
            )

        payloads[name] = data
        records.append({
            "path": name,
            "sha256": sha256(data),
            "size_bytes": len(data),
            "origin": origin,
            "source_commit": source_commit,
        })

    core = {
        "schema_version": 1,
        "git_base_commit": revision,
        "files": records,
    }
    canonical = json.dumps(
        core, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    snapshot_id = sha256(canonical)
    manifest = {**core, "snapshot_id": snapshot_id}
    parent = root / "evals" / "corpus_snapshots"
    destination = parent / snapshot_id[:16]

    if destination.exists():
        verify_snapshot(destination, manifest)
        return destination, manifest, True

    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".staging-", dir=parent))
    try:
        for name, data in payloads.items():
            target = staging / name
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(data)
        with (staging / "manifest.json").open("x", encoding="utf-8") as stream:
            json.dump(manifest, stream, indent=2, ensure_ascii=False)
            stream.write("\n")
        verify_snapshot(staging, manifest)
        staging.rename(destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)

    return destination, manifest, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    args = parser.parse_args()

    try:
        destination, manifest, reused = freeze_corpus(
            args.project_root, args.revision
        )
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as error:
        parser.exit(1, f"Snapshot not created: {error}\n")

    root = args.project_root.resolve()
    relative = destination.relative_to(root)
    print("Snapshot reused and verified." if reused else "Snapshot created and verified.")
    print(f"Snapshot: {relative}")
    print(f"Manifest: {relative / 'manifest.json'}")
    print(f"Files verified: {len(manifest['files'])}")
    local_only = [r["path"] for r in manifest["files"] if r["origin"] == "local"]
    print(f"Local-only inputs: {', '.join(local_only) or '(none)'}")
    print(f"Snapshot ID: {manifest['snapshot_id']}")


if __name__ == "__main__":
    main()


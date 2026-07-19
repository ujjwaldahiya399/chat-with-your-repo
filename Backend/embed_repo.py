import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import chromadb

from embed import embed_file, CHROMA_DB_PATH, COLLECTION_NAME

SKIP_DIRS = {".git", "venv", "__pycache__", "node_modules", "chroma_db"}
ALLOWED_EXTENSIONS = {".py", ".txt", ".md"}


def embed_repo(folder_path, *, repo_root=None, repo_url=None, branch=None):
    """Walk a folder and embed every relevant file in it, skipping the rest.

    repo_root/repo_url/branch are passed straight through to embed_file() —
    see its docstring. Left unset (the plain local-folder CLI usage below),
    behavior is unchanged from before this file supported repo-URL ingestion.
    """
    dirs_skipped = 0
    files_walked = 0
    files_skipped_extension = 0
    files_embedded = 0
    files_failed = 0

    for dirpath, dirnames, filenames in os.walk(folder_path, topdown=True):
        keep = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        dirs_skipped += len(dirnames) - len(keep)
        dirnames[:] = keep

        for filename in filenames:
            files_walked += 1
            file_path = os.path.join(dirpath, filename)

            if Path(filename).suffix not in ALLOWED_EXTENSIONS:
                files_skipped_extension += 1
                continue

            try:
                embed_file(file_path, repo_root=repo_root, repo_url=repo_url, branch=branch)
                files_embedded += 1
            except SystemExit:
                print(f"Warning: skipped {file_path} (no content to embed)", file=sys.stderr)
                files_failed += 1
            except Exception as e:
                print(f"Warning: failed to embed {file_path}: {e}", file=sys.stderr)
                files_failed += 1

    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = client.get_or_create_collection(COLLECTION_NAME)

    print()
    print(f"Files walked: {files_walked}")
    print(f"Skipped: {dirs_skipped} directories, {files_skipped_extension} files (extension)")
    print(f"Embedded: {files_embedded}")
    print(f"Failed: {files_failed}")
    print(f"Collection '{COLLECTION_NAME}' total chunk count: {collection.count()}")

    return files_embedded, collection.count()


def parse_github_url(url):
    """Extract (owner, repo, canonical_https_url) from a github.com HTTPS URL.

    Only the https://github.com/<owner>/<repo>[.git][/] form is supported —
    that's the form the citation links need anyway (a browsable URL), so
    there's no reason to also accept ssh git@github.com:... URLs here.
    """
    match = re.match(r"https://github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$", url.strip())
    if not match:
        raise ValueError(f"Not a github.com HTTPS URL: {url}")
    owner, repo = match.group(1), match.group(2)
    return owner, repo, f"https://github.com/{owner}/{repo}"


def clone_repo(url, branch="main"):
    """Shallow-clone a GitHub repo into a fresh temp dir.

    Shells out to the system `git` binary via subprocess rather than using
    GitPython: git is already a hard requirement for this project (it's a
    git repo), so this adds zero new dependencies, where GitPython would be
    a new package for something `git clone` already does in one line.
    --depth 1 skips full history since only one commit's file tree is
    needed for embedding.
    """
    owner, repo, repo_url = parse_github_url(url)
    dest = Path(tempfile.mkdtemp(prefix=f"{repo}-"))
    result = subprocess.run(
        ["git", "clone", "--depth", "1", "--branch", branch, url, str(dest)],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        shutil.rmtree(dest, ignore_errors=True)
        raise RuntimeError(f"git clone failed: {result.stderr.strip()}")
    return dest, repo_url


def embed_repo_from_url(github_url, branch="main"):
    """Clone a GitHub repo and embed every file in it, tagging each chunk
    with repo_url/branch so retrieval can build a real GitHub citation link.
    """
    dest, repo_url = clone_repo(github_url, branch=branch)
    try:
        return embed_repo(str(dest), repo_root=dest, repo_url=repo_url, branch=branch)
    finally:
        shutil.rmtree(dest, ignore_errors=True)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python embed_repo.py <folder_path>", file=sys.stderr)
        sys.exit(1)

    folder_arg = sys.argv[1]
    folder = Path(folder_arg)

    if not folder.exists():
        print(f"Error: Folder not found: {folder_arg}", file=sys.stderr)
        sys.exit(1)

    if not folder.is_dir():
        print(f"Error: Not a directory: {folder_arg}", file=sys.stderr)
        sys.exit(1)

    embed_repo(folder_arg)

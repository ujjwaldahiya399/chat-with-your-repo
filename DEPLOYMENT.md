# Deployment contingencies and what actually happened

This started as a list of risks to check if a deploy failed. The
backend is now live on Google Cloud Run and the frontend on Vercel
(https://chat-with-your-repo.vercel.app), both verified end to end
against the live URLs — so the sections below are updated with what
actually happened, not just what might have.

## 1. requirements.txt was pinned on macOS

Render builds on Linux. Packages with native/compiled components
(`sentence-transformers`, `torch`, `chromadb`) installing cleanly here
doesn't guarantee a matching prebuilt wheel exists for Linux under the
same version pin.

**Outcome:** this did not end up being the blocker. Switching to a
CPU-only torch build resolved cleanly on Linux with no wheel issues.

## 2. ChromaDB + system SQLite version

ChromaDB often needs a newer SQLite than what's bundled in some base
Python Docker images. If the deploy fails with an error mentioning
`sqlite3` version:

- Add `pysqlite3-binary` to `requirements.txt`
- Add this shim to the very top of `main.py`, before any other imports:

```python
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
```

**Outcome:** never triggered. No sqlite-related error was hit on
Render, HF Spaces, or Cloud Run, so this shim was never applied — it
remains an unhit contingency, left here in case a future base image
change surfaces it.

## 3. Hugging Face Spaces

Same free-tier caveat as Render, different platform: HF Spaces' free
CPU Basic tier can also idle/restart on inactivity, which resets the
container's filesystem. `chroma_db/` is built at ingest time and
isn't committed to git, so it won't persist across restarts there
either — unless the paid persistent-storage add-on is configured.

**Outcome:** abandoned, but not because of the filesystem caveat.
Partway through this deployment, HF Spaces made the Docker SDK a
paid-only feature (a platform policy change, not a config error on
this project's part), which ruled out Spaces for a free deployment.
Moved to Google Cloud Run instead.

## 4. Google Cloud Run

Cloud Run's free tier is usage-based (2M requests, 360k vCPU-seconds,
180k GiB-seconds/month) and scales to zero when idle, rather than the
fixed always-on ceiling Render/HF's free tiers impose — a low-traffic
portfolio demo should stay within it. Same filesystem caveat as the
other two platforms applies here too: Cloud Run's container filesystem
is ephemeral between deployments/cold starts, so `chroma_db/` won't
persist without a mounted Cloud Storage volume (not set up here).
Accepted limitation, not something to solve now.

## 5. What actually broke on Cloud Run (and the fixes)

This is where the real deployment issues showed up, in the order hit:

**a) Default memory/CPU too low for model startup.** Cloud Run's
default (512MiB memory, fractional CPU) isn't enough for the
sentence-transformers/torch startup load — the service failed to
start under it. Fixed by deploying with:

```
--memory=2Gi --cpu=2 --cpu-boost
```

**b) Missing `git` binary in the container.** The Dockerfile was
originally written for Hugging Face Spaces (base image
`python:3.13-slim`), which doesn't include `git`. `clone_repo()` in
`embed_repo.py` shells out to the system `git` via `subprocess`, so
repo ingestion crashed on Cloud Run — confirmed via a real traceback:
`FileNotFoundError: [Errno 2] No such file or directory: 'git'`. Fixed
by adding a `RUN apt-get install -y --no-install-recommends git`
step before the pip install layer.

**c) Both fixes were confirmed live**, not just locally — by hitting
the deployed Cloud Run endpoints directly with `curl` after each
change (a repo-embed request to exercise `clone_repo()`, and an ask
request to exercise the model startup path), not just by testing on
the local machine where `git` and memory headroom were never an
issue.

## 6. Vercel: Root Directory must be set explicitly

The repo has both `Backend/` and `Frontend/` at the root, so Vercel's
auto-detection doesn't find the frontend from the repo root. The
first deploy attempt failed with `vite: command not found`. Fixed by
setting **Root Directory** to `Frontend` explicitly in Project
Settings → Build and Deployment.

# Deployment contingencies

Known risks, not yet hit. Check here first if a Render deploy fails.

## 1. requirements.txt was pinned on macOS

Render builds on Linux. Packages with native/compiled components
(`sentence-transformers`, `torch`, `chromadb`) installing cleanly here
doesn't guarantee a matching prebuilt wheel exists for Linux under the
same version pin. If `pip install` fails during Render's build step,
this is the first thing to check.

## 2. ChromaDB + system SQLite version

ChromaDB often needs a newer SQLite than what's bundled in some base
Python Docker images (including Render's). If the deploy fails with an
error mentioning `sqlite3` version:

- Add `pysqlite3-binary` to `requirements.txt`
- Add this shim to the very top of `main.py`, before any other imports:

```python
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
```

Do not apply this preemptively — only add it if a real deploy fails
with a sqlite-related error, to confirm that's actually the cause
before changing code that currently works.

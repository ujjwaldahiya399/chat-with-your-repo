"""Isolated Python chunking experiment. No embeddings or database writes.

Run --self-test for structural checks with a deterministic character counter.
Use --revision COMMIT to audit one committed file, or --snapshot DIRECTORY
--summary-only to audit all inputs in a frozen corpus with all-MiniLM-L6-v2.

The generic splitter keeps source partitions and statements intact when they
fit. Oversized compound statements are split recursively. When no smaller
statement is available, source is split between complete lines. A single line
plus context that cannot fit causes an explicit ValueError. Returned chunks
are source excerpts and need not execute independently. No manually tuned
query_result/retrieved_chunks grouping is applied.
"""

import argparse
import ast
import hashlib
import json
import subprocess
from pathlib import Path


def partition_python_source(source):
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)
    spans = []
    cursor = 1

    def add_span(start, end, kind, symbol=""):
        if start <= end:
            spans.append({
                "kind": kind,
                "symbol": symbol,
                "start_line": start,
                "end_line": end,
                "source_text": "".join(lines[start - 1:end]),
            })

    for node in tree.body:
        if not isinstance(
            node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
        ):
            continue
        start = (
            node.decorator_list[0].lineno
            if node.decorator_list else node.lineno
        )
        add_span(cursor, start - 1, "module")
        add_span(start, node.end_lineno, "definition", node.name)
        cursor = node.end_lineno + 1

    add_span(cursor, len(lines), "module")
    return spans


def build_embedding_input(span, file_path):
    scope = span["symbol"] or "<module>"
    context = f"File: {file_path}\nScope: {scope}\nSource:\n"
    return context + span["source_text"]


def count_tokens(text, tokenizer):
    return len(tokenizer(
        text, add_special_tokens=True, truncation=False
    )["input_ids"])


def _validate_budget(budget):
    if not isinstance(budget, int) or isinstance(budget, bool) or budget <= 0:
        raise ValueError("budget must be a positive integer")


def _split_span_by_lines(span, lines, file_path, tokenizer, budget):
    """Shared whole-line packing for text documents and oversized Python leaves."""
    def piece(start, end):
        return {
            **span,
            "start_line": start,
            "end_line": end,
            "source_text": "".join(lines[start - 1:end]),
        }

    def fits(candidate):
        text = build_embedding_input(candidate, file_path)
        return count_tokens(text, tokenizer) <= budget

    if fits(span):
        return [span]
    pieces = []
    start = span["start_line"]
    for end in range(start, span["end_line"] + 1):
        candidate = piece(start, end)
        if fits(candidate):
            continue
        if end == start:
            raise ValueError(
                f"Source line {end} plus context exceeds the budget."
            )
        pieces.append(piece(start, end - 1))
        start = end
        if not fits(piece(start, end)):
            raise ValueError(
                f"Source line {end} plus context exceeds the budget."
            )
    pieces.append(piece(start, span["end_line"]))
    return pieces


def split_python_source(source, file_path, tokenizer, budget=220):
    """Return contiguous source spans within a final-input token budget.

    Source identity, revision metadata and retrieval IDs belong to the caller.
    This function never loads or executes the input source.
    """
    _validate_budget(budget)
    lines = source.splitlines(keepends=True)
    tree = ast.parse(source)

    def slice_span(parent, start, end):
        return {
            **parent,
            "start_line": start,
            "end_line": end,
            "source_text": "".join(lines[start - 1:end]),
        }

    def fits(span):
        text = build_embedding_input(span, file_path)
        return count_tokens(text, tokenizer) <= budget

    def split_lines(span):
        return _split_span_by_lines(
            span, lines, file_path, tokenizer, budget
        )

    def statement_children(node):
        children = []
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.stmt):
                children.append(child)
            else:
                # ExceptHandler and match_case wrap statement bodies.
                body = getattr(child, "body", None)
                if isinstance(body, list):
                    children.extend(
                        item for item in body if isinstance(item, ast.stmt)
                    )
        return sorted(children, key=lambda item: item.lineno)

    def divide(span, nodes):
        if fits(span):
            return [span]
        if not nodes:
            return split_lines(span)

        pieces = []
        cursor = span["start_line"]
        for node in nodes:
            # Multiple statements may share a physical source line.
            if node.end_lineno < cursor:
                continue
            piece = slice_span(span, cursor, node.end_lineno)
            pieces.extend(divide(piece, statement_children(node)))
            cursor = node.end_lineno + 1

        if cursor <= span["end_line"]:
            tail = slice_span(span, cursor, span["end_line"])
            pieces.extend(divide(tail, []))

        packed = []
        for piece in pieces:
            if packed:
                candidate = slice_span(
                    span, packed[-1]["start_line"], piece["end_line"]
                )
                if fits(candidate):
                    packed[-1] = candidate
                    continue
            packed.append(piece)
        return packed

    result = []
    for span in partition_python_source(source):
        roots = [
            node for node in tree.body
            if span["start_line"] <= node.lineno
            and node.end_lineno <= span["end_line"]
        ]
        result.extend(divide(span, roots))
    return result


def split_text_source(source, file_path, tokenizer, budget=220):
    """Keep fitting UTF-8 text whole; otherwise pack complete original lines."""
    _validate_budget(budget)
    lines = source.splitlines(keepends=True)
    if not lines:
        return []
    span = {
        "kind": "text",
        "symbol": "<document>",
        "start_line": 1,
        "end_line": len(lines),
        "source_text": source,
    }
    return _split_span_by_lines(span, lines, file_path, tokenizer, budget)


def split_source(source, file_path, tokenizer, budget=220):
    suffix = Path(file_path).suffix.lower()
    if suffix == ".py":
        return split_python_source(source, file_path, tokenizer, budget)
    if suffix in {".md", ".txt"}:
        return split_text_source(source, file_path, tokenizer, budget)
    raise ValueError(f"Unsupported source extension: {suffix or '(none)'}")


def validate_spans(source, spans, file_path, tokenizer, budget):
    """Check exact reconstruction, contiguous line locations, and input sizes."""
    lines = source.splitlines(keepends=True)
    if "".join(span["source_text"] for span in spans) != source:
        raise AssertionError("Source reconstruction failed")
    cursor = 1
    for span in spans:
        start, end = span["start_line"], span["end_line"]
        if start != cursor or end < start:
            raise AssertionError("Invalid or noncontiguous line references")
        if span["source_text"] != "".join(lines[start - 1:end]):
            raise AssertionError("Source text disagrees with line references")
        text = build_embedding_input(span, file_path)
        if count_tokens(text, tokenizer) > budget:
            raise AssertionError("Final embedding input exceeds budget")
        cursor = end + 1
    if cursor != len(lines) + 1:
        raise AssertionError("Source line coverage is incomplete")


def run_self_tests():
    # This adapter tests mechanics, not the Bert model's token counts.
    def counter(text, **kwargs):
        if kwargs != {"add_special_tokens": True, "truncation": False}:
            raise AssertionError("Tokenization options changed")
        return {"input_ids": range(len(text) + 2)}

    fixtures = {
        "module": (
            "import os\n\n"
            + "\n".join(f"value_{i} = {i}" for i in range(25)) + "\n"
        ),
        "decorated function": (
            "@decorator\ndef work():\n"
            + "".join(f"    value_{i} = {i}\n" for i in range(25))
            + "\nprint('done')\n"
        ),
        "class and methods": (
            "class Worker:\n    label = 'x'\n\n    def first(self):\n"
            + "".join(f"        value_{i} = {i}\n" for i in range(25))
            + "\n    def second(self):\n        return 'ok'\n"
        ),
        "if/else": (
            "if enabled:\n"
            + "".join(f"    value_{i} = {i}\n" for i in range(12))
            + "else:\n"
            + "".join(f"    other_{i} = {i}\n" for i in range(12))
        ),
        "try/except/finally": (
            "try:\n"
            + "".join(f"    value_{i} = {i}\n" for i in range(12))
            + "except ValueError:\n"
            + "".join(f"    other_{i} = {i}\n" for i in range(12))
            + "finally:\n    cleanup()\n"
        ),
        "Unicode": (
            "def example():\n"
            + "".join(f"    text_{i} = 'नमस्ते 🌿'\n" for i in range(20))
        ),
        "empty source": "",
        "same-line statements": (
            "a = 1; b = 2\n"
            + "\n".join(f"value_{i} = {i}" for i in range(25))
        ),
        "multiline docstring": (
            "def embed_file(path):\n    \"\"\"Description.\n\n"
            + "".join("    A line of explanatory documentation.\n" for _ in range(30))
            + "    \"\"\"\n    return path\n"
        ),
        "prompt with comments": (
            "\n" + "".join("# A line explaining the prompt.\n" for _ in range(12))
            + "SYSTEM_PROMPT = (\n"
            + "".join('    "An instruction for the language model. "\n' for _ in range(20))
            + ")\n"
        ),
        "multiline comments": "".join("# Documentation line.\n" for _ in range(40)),
        "CRLF and Unicode": (
            "def example():\r\n    \"\"\"\r\n"
            + "".join("    नमस्ते 🌿 documentation.\r\n" for _ in range(20))
            + "    \"\"\"\r\n    return 1\r\n"
        ),
    }
    for name, source in fixtures.items():
        spans = split_python_source(source, "fixture.py", counter, budget=180)
        validate_spans(source, spans, "fixture.py", counter, budget=180)
        print(f"PASS: {name}")

    small = "@decorator\ndef f():\n    return 1\n\nprint('after')\n"
    spans = partition_python_source(small)
    definition = next(span for span in spans if span["symbol"] == "f")
    if (definition["start_line"], definition["end_line"]) != (1, 3):
        raise AssertionError("Decorator or exact definition boundary failed")
    if "".join(span["source_text"] for span in spans) != small:
        raise AssertionError("Small source reconstruction failed")
    print("PASS: exact decorated-definition boundary")

    for name, source in [
        ("oversized line", "text = '" + "x" * 500 + "'\n"),
        ("oversized comment", "#" + "x" * 500 + "\n"),
    ]:
        try:
            split_python_source(source, "fixture.py", counter, budget=180)
        except ValueError:
            print(f"PASS: explicit failure for {name}")
        else:
            raise AssertionError(f"Expected explicit failure for {name}")

    try:
        split_python_source("def broken(:\n", "fixture.py", counter, budget=180)
    except SyntaxError:
        print("PASS: explicit parse failure")
    else:
        raise AssertionError("Expected parse failure")
    text_fixtures = {
        "empty Markdown": ("fixture.md", ""),
        "whitespace text": ("fixture.txt", "\n  \t\n"),
        "short requirements": ("requirements.txt", "chromadb\nsentence-transformers\n"),
        "long Markdown": (
            "NOTES.md",
            "# Notes\n\n" + "".join(f"Paragraph line {i}: retained text.\n" for i in range(50)),
        ),
        "CRLF text": ("fixture.txt", "पहला 🌿\r\nदूसरा\r\n" * 25),
        "no final newline": ("fixture.md", "A paragraph.\n" * 30 + "Final line."),
    }
    for name, (file_path, source) in text_fixtures.items():
        spans = split_source(source, file_path, counter, budget=180)
        validate_spans(source, spans, file_path, counter, budget=180)
        if name == "short requirements" and len(spans) != 1:
            raise AssertionError("Small text was unnecessarily split")
        if name == "whitespace text" and any(s["source_text"].strip() for s in spans):
            raise AssertionError("Whitespace-only input became indexable")
        print(f"PASS: {name}")
    try:
        split_source("x" * 500, "fixture.txt", counter, budget=180)
    except ValueError:
        print("PASS: explicit failure for oversized text line")
    else:
        raise AssertionError("Expected oversized text-line failure")
    try:
        split_source("data", "fixture.bin", counter, budget=180)
    except ValueError:
        print("PASS: unsupported extension rejected")
    else:
        raise AssertionError("Expected unsupported-extension failure")

    print("All structural self-tests passed (deterministic counter).")


def load_source_snapshot(directory):
    """Verify manifest identity and read verified bytes once into memory."""
    directory = Path(directory).resolve()
    manifest = json.loads(
        (directory / "manifest.json").read_text(encoding="utf-8")
    )
    core = {key: value for key, value in manifest.items() if key != "snapshot_id"}
    canonical = json.dumps(
        core, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != manifest.get("snapshot_id"):
        raise ValueError("Snapshot manifest fingerprint does not match")
    if manifest.get("schema_version") != 1:
        raise ValueError("Unsupported snapshot schema")
    sources = {}
    for record in manifest["files"]:
        name = record["path"]
        relative = Path(name)
        target = (directory / relative).resolve()
        if relative.is_absolute() or ".." in relative.parts or not target.is_relative_to(directory):
            raise ValueError(f"Invalid snapshot source path: {name}")
        if name in sources:
            raise ValueError(f"Duplicate snapshot source path: {name}")
        raw = target.read_bytes()
        if len(raw) != record["size_bytes"] or hashlib.sha256(raw).hexdigest() != record["sha256"]:
            raise ValueError(f"Snapshot source fingerprint does not match: {name}")
        sources[name] = raw.decode("utf-8")
    if not sources:
        raise ValueError("Snapshot has no source files")
    return manifest, sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("--revision", help="Git revision to read without checkout")
    inputs.add_argument("--snapshot", type=Path, help="Frozen corpus directory")
    parser.add_argument("--file", help="One source path; default: all snapshot files or Backend/ask.py")
    parser.add_argument("--budget", type=int, default=220)
    parser.add_argument("--summary-only", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        run_self_tests()
        return
    if not (args.revision or args.snapshot):
        parser.error("provide --revision COMMIT, --snapshot DIRECTORY, or --self-test")

    if args.snapshot:
        manifest, sources = load_source_snapshot(args.snapshot)
        if args.file:
            if args.file not in sources:
                parser.error("requested file is not in the snapshot manifest")
            sources = {args.file: sources[args.file]}
        print(f"Snapshot ID: {manifest['snapshot_id']}")
    else:
        revision = subprocess.check_output(
            ["git", "rev-parse", "--verify", f"{args.revision}^{{commit}}"],
            encoding="utf-8",
        ).strip()
        file_path = args.file or "Backend/ask.py"
        source = subprocess.check_output(
            ["git", "show", f"{revision}:{file_path}"]
        ).decode("utf-8")
        sources = {file_path: source}
        print(f"Revision: {revision}")

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    if not 0 < args.budget <= model.max_seq_length:
        parser.error(
            f"budget must be between 1 and model limit {model.max_seq_length}"
        )

    print(f"Final-input target: {args.budget} tokens | model limit: {model.max_seq_length}")
    failures = []
    total_indexable = 0
    for file_path, source in sources.items():
        try:
            spans = split_source(source, file_path, model.tokenizer, args.budget)
            validate_spans(source, spans, file_path, model.tokenizer, args.budget)
        except (ValueError, SyntaxError, AssertionError) as error:
            failures.append(file_path)
            print(f"FAIL {file_path}: {error}")
            continue
        index_spans = [span for span in spans if span["source_text"].strip()]
        sizes = [
            count_tokens(build_embedding_input(span, file_path), model.tokenizer)
            for span in index_spans
        ]
        total_indexable += len(index_spans)
        print(
            f"PASS {file_path}: {len(index_spans)} indexable spans, "
            f"{len(spans) - len(index_spans)} blank spans skipped, "
            f"largest={max(sizes, default=0)} tokens"
        )
        if not args.summary_only:
            for number, (span, size) in enumerate(zip(index_spans, sizes), start=1):
                print(
                    f"  Chunk {number}: lines {span['start_line']}-"
                    f"{span['end_line']}, {size} tokens"
                )
    if failures:
        parser.exit(1, f"Corpus validation incomplete: {len(failures)} file(s) failed.\n")
    print(
        f"All {len(sources)} selected files passed token-budget, "
        "source-preservation, and line-reference checks."
    )
    print(f"Total indexable spans: {total_indexable}")


if __name__ == "__main__":
    main()


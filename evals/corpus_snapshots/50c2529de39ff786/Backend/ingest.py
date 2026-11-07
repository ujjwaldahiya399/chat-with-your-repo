import sys
import ast
from pathlib import Path


def _def_start_line(definition):
    """Return the 1-indexed line where a definition (including decorators) starts."""
    if definition.decorator_list:
        return definition.decorator_list[0].lineno
    return definition.lineno


def chunk_python_file(file_path):
    """Split a Python file into chunks by function/class boundary."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.splitlines(keepends=True)

    try:
        tree = ast.parse(content, filename=file_path)
    except SyntaxError as e:
        print(f"Error parsing {file_path}: {e}", file=sys.stderr)
        return chunk_fixed_size(file_path, lines)

    chunks = []
    definitions = []

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            definitions.append(node)

    if not definitions:
        # No functions or classes, return the whole file as one chunk
        return [(1, ''.join(lines))]

    # Sort by line number (should already be sorted)
    definitions.sort(key=lambda n: n.lineno)

    # Add module-level code before first definition
    first_def_line = _def_start_line(definitions[0])
    if first_def_line > 1:
        chunk_lines = lines[0:first_def_line - 1]
        chunk_content = ''.join(chunk_lines)
        chunks.append((1, chunk_content))

    # Create chunks for each definition, including gap to next definition
    for i, definition in enumerate(definitions):
        # Check if definition has decorators; start at first decorator if present
        start_line_1indexed = _def_start_line(definition)
        start_line = start_line_1indexed - 1  # Convert to 0-indexed

        # Determine end line: extend to next definition or end of file
        if i < len(definitions) - 1:
            # Include everything up to (but not including) the next definition
            next_def_line = _def_start_line(definitions[i + 1])
            end_line = next_def_line - 1
        else:
            # Last definition: include to end of file
            end_line = len(lines)

        chunk_lines = lines[start_line:end_line]
        chunk_content = ''.join(chunk_lines)
        chunks.append((start_line_1indexed, chunk_content))

    return chunks


def chunk_fixed_size(file_path, lines=None, chunk_size=50):
    """Split a file into fixed-size chunks."""
    if lines is None:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()

    chunks = []
    for i in range(0, len(lines), chunk_size):
        start_line = i + 1  # Line numbers are 1-indexed for display
        chunk_lines = lines[i:i + chunk_size]
        chunk_content = ''.join(chunk_lines)
        chunks.append((start_line, chunk_content))

    return chunks


def get_chunks(file_path):
    """Validate the file exists and split it into chunks."""
    path = Path(file_path)

    if not path.exists():
        print(f"Error: File not found: {file_path}", file=sys.stderr)
        sys.exit(1)

    if path.suffix == '.py':
        return chunk_python_file(file_path)
    return chunk_fixed_size(file_path)


def ingest_file(file_path):
    """Ingest a file and split into chunks."""
    chunks = get_chunks(file_path)

    # Print chunks
    for start_line, content in chunks:
        print(f"=== Starting at line {start_line} ===")
        print(content, end='')
        print()


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python ingest.py <file_path>", file=sys.stderr)
        sys.exit(1)

    ingest_file(sys.argv[1])

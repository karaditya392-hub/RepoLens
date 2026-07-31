"""RepoLens Phase 1: clone a GitHub repo and extract function/class-level chunks.

Usage:
    python chunker.py <github-repo-url>

Python files are parsed with the `ast` module into function- and class-level
chunks. Other source files (js, ts, go, etc.) are kept as whole-file chunks so
they aren't lost. Binaries, lockfiles, and vendored/build directories are
skipped.
"""

import ast
import shutil
import stat
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

EXCLUDED_DIRS = {
    ".git", "node_modules", "venv", ".venv", "env", ".env", "__pycache__",
    "dist", "build", "target", "out", ".next", ".nuxt", "vendor",
    ".tox", ".mypy_cache", ".pytest_cache", ".ruff_cache", "site-packages",
    "coverage", ".idea", ".vscode", "eggs", ".eggs",
}

EXCLUDED_FILES = {
    "package-lock.json", "yarn.lock", "pnpm-lock.yaml", "poetry.lock",
    "Pipfile.lock", "uv.lock", "Cargo.lock", "composer.lock", "Gemfile.lock",
    "go.sum",
}

SOURCE_EXTENSIONS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".rb",
    ".c", ".h", ".cpp", ".hpp", ".cs", ".php", ".swift", ".kt", ".scala",
    ".sh", ".sql", ".md", ".toml", ".yaml", ".yml", ".cfg", ".ini",
}

MAX_FILE_BYTES = 512 * 1024  # skip anything over 512 KB; not human-written source


@dataclass
class Chunk:
    name: str          # e.g. "MyClass.my_method" or "<file>"
    kind: str          # "function" | "class" | "method" | "file"
    file_path: str     # path relative to repo root, forward slashes
    start_line: int
    end_line: int
    code: str
    docstring: str | None


def clone_repo(url: str, dest: Path) -> None:
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(dest)],
        check=True, capture_output=True, text=True,
    )


def iter_source_files(root: Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in EXCLUDED_DIRS for part in rel.parts):
            continue
        if path.name in EXCLUDED_FILES:
            continue
        if path.suffix.lower() not in SOURCE_EXTENSIONS:
            continue
        if path.stat().st_size > MAX_FILE_BYTES:
            continue
        yield path, rel


def chunk_python_file(source: str, rel_path: str) -> list[Chunk]:
    tree = ast.parse(source)
    lines = source.splitlines()
    chunks: list[Chunk] = []

    def add(node, name: str, kind: str):
        end = node.end_lineno or node.lineno
        chunks.append(Chunk(
            name=name,
            kind=kind,
            file_path=rel_path,
            start_line=node.lineno,
            end_line=end,
            code="\n".join(lines[node.lineno - 1:end]),
            docstring=ast.get_docstring(node),
        ))

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            add(node, node.name, "function")
        elif isinstance(node, ast.ClassDef):
            add(node, node.name, "class")
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    add(child, f"{node.name}.{child.name}", "method")
    return chunks


def chunk_file(path: Path, rel: Path) -> list[Chunk]:
    rel_path = rel.as_posix()
    try:
        source = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return []

    if path.suffix.lower() == ".py":
        try:
            py_chunks = chunk_python_file(source, rel_path)
        except SyntaxError:
            py_chunks = []
        if py_chunks:
            return py_chunks
        # fall through: empty or unparseable module becomes a file chunk

    if not source.strip():
        return []
    return [Chunk(
        name="<file>",
        kind="file",
        file_path=rel_path,
        start_line=1,
        end_line=source.count("\n") + 1,
        code=source,
        docstring=None,
    )]


def extract_chunks(repo_url: str) -> list[Chunk]:
    tmp = Path(tempfile.mkdtemp(prefix="repolens_"))
    try:
        clone_repo(repo_url, tmp)
        chunks: list[Chunk] = []
        for path, rel in iter_source_files(tmp):
            chunks.extend(chunk_file(path, rel))
        return chunks
    finally:
        # git makes read-only objects on Windows; clear the bit before deleting
        def on_error(func, p, _exc):
            Path(p).chmod(stat.S_IWRITE)
            func(p)
        shutil.rmtree(tmp, onerror=on_error)


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python chunker.py <github-repo-url>", file=sys.stderr)
        sys.exit(1)

    chunks = extract_chunks(sys.argv[1])

    for i, c in enumerate(chunks, 1):
        header = f"[{i}] {c.kind}: {c.name}  ({c.file_path}:{c.start_line}-{c.end_line})"
        print(header)
        if c.docstring:
            first_line = c.docstring.strip().splitlines()[0]
            print(f"    docstring: {first_line}")
        preview = c.code.strip().splitlines()
        for line in preview[:3]:
            print(f"    | {line}")
        if len(preview) > 3:
            print(f"    | ... ({len(preview) - 3} more lines)")
        print()

    print(f"Total: {len(chunks)} chunks from {len({c.file_path for c in chunks})} files")


if __name__ == "__main__":
    main()

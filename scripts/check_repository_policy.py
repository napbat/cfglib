#!/usr/bin/env python3
"""Enforce source-size and Cargo package-layout repository policies."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MAX_SOURCE_LINES = 1_000
SOURCE_SUFFIXES = {
    ".bash",
    ".c",
    ".cc",
    ".cpp",
    ".cs",
    ".cxx",
    ".fish",
    ".go",
    ".h",
    ".hh",
    ".hpp",
    ".java",
    ".js",
    ".jsx",
    ".kt",
    ".kts",
    ".mjs",
    ".ps1",
    ".py",
    ".pyi",
    ".rb",
    ".rs",
    ".sh",
    ".swift",
    ".ts",
    ".tsx",
    ".zsh",
}
KEBAB_CASE = re.compile(r"^[a-z][a-z0-9]*(?:-[a-z0-9]+)*$")
SNAKE_CASE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)*$")
DECORATIVE_DIVIDER = re.compile(r"^\s*//.*[=_─═-]{3,}\s*$")
CARGO_TARGET_DIRECTORIES = ("benches", "examples", "tests")
PUBLIC_FUNCTION = re.compile(
    r"^pub\s+(?:const\s+)?(?:async\s+)?(?:unsafe\s+)?fn\s+([a-z][a-z0-9_]*)\s*(?:<|\()",
    re.MULTILINE,
)
PUBLIC_USE = re.compile(r"^pub use\s+(.+?);", re.MULTILINE | re.DOTALL)
BENCHMARK_API_LIST = re.compile(
    r"PUBLIC_API_FUNCTIONS:\s*&\[&str\]\s*=\s*&\[(.*?)\];",
    re.DOTALL,
)


def run(command: list[str]) -> bytes:
    """Run a repository command and return its standard output."""
    result = subprocess.run(
        command,
        cwd=REPOSITORY_ROOT,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(f"{' '.join(command)} failed: {detail}")
    return result.stdout


def repository_files() -> list[Path]:
    """Return tracked and non-ignored untracked repository files."""
    output = run(
        [
            "git",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ]
    )
    return sorted(
        REPOSITORY_ROOT / os.fsdecode(raw_path)
        for raw_path in output.split(b"\0")
        if raw_path
    )


def is_source(path: Path, data: bytes) -> bool:
    """Return whether a repository file is hand-maintained source code."""
    return path.suffix.lower() in SOURCE_SUFFIXES or data.startswith(b"#!")


def is_generated(data: bytes) -> bool:
    """Recognize an explicit generated-file header, never a path heuristic."""
    header = data[:4_096].decode(errors="replace").casefold()
    return "@generated" in header or (
        "generated" in header and "do not edit" in header
    )


def physical_line_count(data: bytes) -> int:
    """Count physical lines, including a final line without a newline."""
    if not data:
        return 0
    return data.count(b"\n") + int(not data.endswith(b"\n"))


def check_source_sizes(files: list[Path], violations: list[str]) -> int:
    """Check every hand-maintained source file against the line ceiling."""
    checked = 0
    for path in files:
        if not path.is_file():
            continue
        data = path.read_bytes()
        if not is_source(path, data) or is_generated(data):
            continue
        checked += 1
        line_count = physical_line_count(data)
        if line_count > MAX_SOURCE_LINES:
            relative = path.relative_to(REPOSITORY_ROOT).as_posix()
            violations.append(
                f"{relative}: {line_count} lines exceeds the "
                f"{MAX_SOURCE_LINES}-line source limit"
            )
    return checked


def check_comment_style(files: list[Path], violations: list[str]) -> None:
    """Reject decorative comments that substitute typography for structure."""
    for path in files:
        if not path.is_file():
            continue
        data = path.read_bytes()
        if not is_source(path, data) or is_generated(data):
            continue
        relative = path.relative_to(REPOSITORY_ROOT).as_posix()
        for line_number, line in enumerate(data.decode(errors="replace").splitlines(), 1):
            if DECORATIVE_DIVIDER.fullmatch(line):
                violations.append(
                    f"{relative}:{line_number}: decorative divider comments are forbidden"
                )


def cargo_package_roots() -> tuple[Path, list[Path]]:
    """Return Cargo's workspace root and local package roots."""
    metadata = json.loads(
        run(
            [
                "cargo",
                "metadata",
                "--format-version",
                "1",
                "--no-deps",
            ]
        )
    )
    workspace_root = Path(metadata["workspace_root"]).resolve()
    package_roots = sorted(
        {
            Path(package["manifest_path"]).resolve().parent
            for package in metadata["packages"]
            if package["source"] is None
        }
    )
    return workspace_root, package_roots


def check_workspace_files(workspace_root: Path, violations: list[str]) -> None:
    """Check conventional files at the Cargo workspace root."""
    for name in ("Cargo.toml",):
        if not (workspace_root / name).is_file():
            relative = workspace_root.relative_to(REPOSITORY_ROOT).as_posix()
            location = f"{relative}/" if relative != "." else ""
            violations.append(f"{location}{name}: missing from Cargo workspace root")


def check_snake_case_module(path: Path, label: str, violations: list[str]) -> None:
    """Require Rust module files and directories to use snake_case."""
    if not SNAKE_CASE.fullmatch(path.stem):
        violations.append(f"{label}: Rust module names must use snake_case")


def check_multi_file_target(
    target_root: Path,
    relative: Path,
    violations: list[str],
) -> None:
    """Check one file under tests/examples/benches/src/bin."""
    label = (target_root.relative_to(REPOSITORY_ROOT) / relative).as_posix()
    parts = relative.parts
    if len(parts) == 1:
        if relative.suffix == ".rs" and not KEBAB_CASE.fullmatch(relative.stem):
            violations.append(f"{label}: Cargo target names must use kebab-case")
        return

    target_name = parts[0]
    if not KEBAB_CASE.fullmatch(target_name):
        violations.append(
            f"{label}: multi-file Cargo target directory must use kebab-case"
        )
    if not (target_root / target_name / "main.rs").is_file():
        violations.append(
            f"{label}: multi-file Cargo target '{target_name}' requires main.rs"
        )
    for directory in parts[1:-1]:
        check_snake_case_module(Path(directory), label, violations)
    if relative.suffix == ".rs" and relative.name != "main.rs":
        check_snake_case_module(relative, label, violations)


def check_package_source(
    package_root: Path,
    path: Path,
    violations: list[str],
) -> None:
    """Check one Rust source path against Cargo's package conventions."""
    relative = path.relative_to(package_root)
    label = path.relative_to(REPOSITORY_ROOT).as_posix()
    parts = relative.parts

    if relative.name == "mod.rs":
        violations.append(
            f"{label}: mod.rs is forbidden; use the named-file module style "
            "(parent.rs beside a parent/ directory)"
        )

    if len(parts) == 1 and relative.name == "build.rs":
        return
    if not parts or parts[0] not in ("src", *CARGO_TARGET_DIRECTORIES):
        violations.append(
            f"{label}: Rust source belongs in src, tests, examples, or benches"
        )
        return

    if parts[0] in CARGO_TARGET_DIRECTORIES:
        check_multi_file_target(
            package_root / parts[0],
            Path(*parts[1:]),
            violations,
        )
        return

    if len(parts) >= 2 and parts[1] == "bin":
        check_multi_file_target(
            package_root / "src" / "bin",
            Path(*parts[2:]),
            violations,
        )
        return

    for directory in parts[1:-1]:
        check_snake_case_module(Path(directory), label, violations)
    check_snake_case_module(relative, label, violations)


def check_cargo_layout(
    files: list[Path],
    workspace_root: Path,
    package_roots: list[Path],
    violations: list[str],
) -> None:
    """Check the applicable Cargo Book package-layout conventions."""
    check_workspace_files(workspace_root, violations)
    for package_root in package_roots:
        if not (package_root / "Cargo.toml").is_file():
            label = package_root.relative_to(REPOSITORY_ROOT).as_posix()
            violations.append(f"{label}/Cargo.toml: missing package manifest")
        conventional_roots = ("lib.rs", "main.rs")
        if not any((package_root / "src" / name).is_file() for name in conventional_roots):
            label = package_root.relative_to(REPOSITORY_ROOT).as_posix()
            violations.append(f"{label}/src: expected lib.rs or main.rs")

    rust_files = [path for path in files if path.suffix == ".rs" and path.is_file()]
    for path in rust_files:
        containing_roots = [root for root in package_roots if path.is_relative_to(root)]
        if not containing_roots:
            label = path.relative_to(REPOSITORY_ROOT).as_posix()
            violations.append(f"{label}: Rust source is outside a Cargo package")
            continue
        package_root = max(containing_roots, key=lambda root: len(root.parts))
        check_package_source(package_root, path, violations)


def facade_export_names(source: str) -> set[str]:
    """Return item names explicitly re-exported by the crate facade."""
    names: set[str] = set()
    for statement in PUBLIC_USE.findall(source):
        if "{" in statement:
            items = statement.split("{", 1)[1].rsplit("}", 1)[0]
        else:
            items = statement
        for item in items.split(","):
            item = item.strip()
            if not item:
                continue
            exported = item.rsplit(" as ", 1)[-1].rsplit("::", 1)[-1].strip()
            if SNAKE_CASE.fullmatch(exported):
                names.add(exported)
    return names


def check_benchmark_api_inventory(violations: list[str]) -> int:
    """Keep the performance inventory aligned with facade free functions."""
    crate_root = REPOSITORY_ROOT / "crates" / "cfglib"
    lib_path = crate_root / "src" / "lib.rs"
    inventory_path = crate_root / "benches" / "performance" / "api_cases.rs"
    if not inventory_path.is_file():
        violations.append(
            "crates/cfglib/benches/performance/api_cases.rs: missing benchmark API inventory"
        )
        return 0

    lib_source = lib_path.read_text(encoding="utf-8")
    exported = facade_export_names(lib_source)
    definitions: set[str] = set()
    for source_path in (crate_root / "src").rglob("*.rs"):
        definitions.update(
            PUBLIC_FUNCTION.findall(source_path.read_text(encoding="utf-8"))
        )
    expected = exported & definitions

    inventory_source = inventory_path.read_text(encoding="utf-8")
    inventory_match = BENCHMARK_API_LIST.search(inventory_source)
    if inventory_match is None:
        violations.append(
            "crates/cfglib/benches/performance/api_cases.rs: "
            "PUBLIC_API_FUNCTIONS inventory is missing"
        )
        return 0
    actual = set(re.findall(r'"([a-z][a-z0-9_]*)"', inventory_match.group(1)))

    missing = sorted(expected - actual)
    stale = sorted(actual - expected)
    if missing:
        violations.append(
            "crates/cfglib/benches/performance/api_cases.rs: "
            f"public facade functions missing from benchmark inventory: {missing}"
        )
    if stale:
        violations.append(
            "crates/cfglib/benches/performance/api_cases.rs: "
            f"non-facade functions in benchmark inventory: {stale}"
        )
    return len(expected)


def main() -> int:
    """Run all repository-policy checks."""
    try:
        files = repository_files()
        workspace_root, package_roots = cargo_package_roots()
    except (OSError, RuntimeError, json.JSONDecodeError) as error:
        print(f"repository policy check failed to start: {error}", file=sys.stderr)
        return 2

    violations: list[str] = []
    source_count = check_source_sizes(files, violations)
    check_comment_style(files, violations)
    check_cargo_layout(files, workspace_root, package_roots, violations)
    benchmark_api_count = check_benchmark_api_inventory(violations)

    if violations:
        print("Repository policy violations:", file=sys.stderr)
        for violation in sorted(set(violations)):
            print(f"  - {violation}", file=sys.stderr)
        return 1

    print(
        "Repository policy checks passed "
        f"({source_count} source files, {len(package_roots)} Cargo packages, "
        f"{benchmark_api_count} benchmarked facade functions)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

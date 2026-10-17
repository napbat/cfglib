#!/usr/bin/env python3
"""Enforce the repository policies AGENTS.md states and tools do not."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CRATE_ROOT = REPOSITORY_ROOT / "crates" / "cfglib"
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
NAMESPACE_MODULES = ("analysis", "dataflow", "graph", "ir", "transform")
VERSION = re.compile(r"\b\d+\.\d+(?:\.\d+)?\b")
TOOLCHAIN_PIN = re.compile(r"^\s*toolchain:\s*[\"']?(\d+\.\d+(?:\.\d+)?)[\"']?\s*$")
TOOLCHAIN_CHANNEL = re.compile(r"^\s*channel\s*=\s*[\"'](\d+\.\d+(?:\.\d+)?)[\"']")
INNER_LINT_ATTRIBUTE = re.compile(r"^\s*#!\[(?:allow|warn|expect)\(", re.MULTILINE)
CLIPPY_SETTING = re.compile(r"^\s*([A-Za-z0-9_-]+)\s*=")
COUNT_PREFIX_NAME = re.compile(
    r"\bpub(?:\([^)]*\))?\s+(?:(?:const|unsafe|async)\s+)*(?:fn|const|static)\s+(num_\w*)"
    r"|\bpub(?:\([^)]*\))?\s+(num_\w*)\s*:"
)
TEST_MODULE = re.compile(
    r"#\[cfg\(test\)\]\s*(?:#\[[^\]]*\]\s*)*(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)\s*([;{])"
)
BRITISH_SPELLING = re.compile(
    r"\b(?:(?:behaviour|colour|honour|neighbour|favour)\w*|centre[ds]?|licence|artefacts?|grey"
    r"|(?:labell|modell|cancell|travell|signall|levell)(?:ed|ing|er|ers)"
    r"|(?:analys|paralys)(?:e|ed|ing)"
    r"|(?:initial|normal|optim|recogn|serial|organ|synchron|special|minim|maxim|summar"
    r"|final|general|material|util|custom|visual|parametr|canonical|categor|character"
    r"|prioriti|standard|local|stabil|linear|vector|sanit)is(?:e|es|ed|ing|ation|ations)"
    r")\b",
    re.IGNORECASE,
)
SPELLING_SUFFIXES = {".md", ".py", ".rs", ".toml", ".yml"}


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


def cargo_metadata() -> dict:
    """Return Cargo's metadata for the workspace's own packages."""
    return json.loads(
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


def cargo_package_roots(metadata: dict) -> tuple[Path, list[Path]]:
    """Return Cargo's workspace root and local package roots."""
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
    lib_path = CRATE_ROOT / "src" / "lib.rs"
    inventory_path = CRATE_ROOT / "benches" / "performance" / "api_cases.rs"
    if not inventory_path.is_file():
        violations.append(
            "crates/cfglib/benches/performance/api_cases.rs: missing benchmark API inventory"
        )
        return 0

    lib_source = lib_path.read_text(encoding="utf-8")
    exported = facade_export_names(lib_source)
    definitions: set[str] = set()
    for source_path in (CRATE_ROOT / "src").rglob("*.rs"):
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


def relative_label(path: Path) -> str:
    """Return a repository-relative POSIX path for a violation message."""
    return path.relative_to(REPOSITORY_ROOT).as_posix()


def version_tuple(version: str) -> tuple[int, ...]:
    """Parse `1.85` or `1.85.0` into a comparable tuple, ignoring a zero patch."""
    parts = [int(part) for part in version.split(".")]
    while len(parts) > 2 and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def check_msrv_sources(
    metadata: dict,
    files: list[Path],
    violations: list[str],
) -> None:
    """Keep the manifest's `rust-version` the only place the MSRV is written."""
    packages = [package for package in metadata["packages"] if package["source"] is None]
    for package in packages:
        if package.get("rust_version") is None:
            manifest = relative_label(Path(package["manifest_path"]))
            violations.append(f"{manifest}: package declares no rust-version")
    declared = {package["rust_version"] for package in packages if package.get("rust_version")}
    if len(declared) != 1:
        if len(declared) > 1:
            violations.append(f"packages declare different rust-version values: {sorted(declared)}")
        return
    msrv = declared.pop()
    msrv_key = version_tuple(msrv)

    reads_manifest = False
    for path in files:
        if not path.is_file():
            continue
        label = relative_label(path)
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        if path.parent == REPOSITORY_ROOT / ".github" / "workflows":
            reads_manifest = reads_manifest or any("rust_version" in line for line in lines)
            for number, line in enumerate(lines, 1):
                pin = TOOLCHAIN_PIN.match(line)
                if pin:
                    violations.append(
                        f"{label}:{number}: toolchain pin {pin.group(1)} duplicates "
                        f"rust-version {msrv}; read it from the manifest instead"
                    )
        elif path.name == "rust-toolchain.toml":
            for number, line in enumerate(lines, 1):
                channel = TOOLCHAIN_CHANNEL.match(line)
                if channel and version_tuple(channel.group(1)) < msrv_key:
                    violations.append(
                        f"{label}:{number}: toolchain {channel.group(1)} is older than "
                        f"rust-version {msrv}"
                    )
        elif path.suffix == ".md":
            for number, line in enumerate(lines, 1):
                if "msrv" not in line.casefold():
                    continue
                for found in VERSION.findall(line):
                    if version_tuple(found) != msrv_key:
                        violations.append(
                            f"{label}:{number}: states MSRV {found}, but the manifest's "
                            f"rust-version is {msrv}"
                        )
    if not reads_manifest:
        violations.append(
            ".github/workflows: no workflow reads the manifest's rust_version for an MSRV job"
        )


def check_lint_policy(files: list[Path], violations: list[str]) -> None:
    """Reject module-wide lint attributes and a tuned Clippy configuration."""
    for path in files:
        if not path.is_file():
            continue
        if path.suffix == ".rs":
            text = path.read_text(encoding="utf-8", errors="replace")
            for match in INNER_LINT_ATTRIBUTE.finditer(text):
                number = text.count("\n", 0, match.start()) + 1
                violations.append(
                    f"{relative_label(path)}:{number}: crate- and module-wide lint "
                    "attributes are forbidden; set lint levels in the workspace manifest "
                    "or fix the code"
                )
        elif path.name in ("clippy.toml", ".clippy.toml"):
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                setting = CLIPPY_SETTING.match(line)
                if setting and not setting.group(1).startswith("disallowed-"):
                    violations.append(
                        f"{relative_label(path)}:{number}: `{setting.group(1)}` tunes Clippy; "
                        "fix the code instead of weakening the workspace lint configuration"
                    )


def check_spelling(files: list[Path], violations: list[str]) -> None:
    """Require American spelling in identifiers, documentation, and comments."""
    for path in files:
        if (
            not path.is_file()
            or path.suffix not in SPELLING_SUFFIXES
            or path == Path(__file__).resolve()
        ):
            continue
        found: dict[str, int] = {}
        for number, line in enumerate(
            path.read_text(encoding="utf-8", errors="replace").splitlines(), 1
        ):
            for match in BRITISH_SPELLING.finditer(line):
                found.setdefault(match.group(0).casefold(), number)
        if found:
            words = ", ".join(f"`{word}` (line {number})" for word, number in found.items())
            violations.append(
                f"{relative_label(path)}: British spelling {words}; use American English"
            )


def module_source(parent: Path, name: str) -> str:
    """Return the source of the out-of-line module `name` declared in `parent`."""
    directory = parent.parent if parent.name in ("lib.rs", "main.rs") else parent.with_suffix("")
    for candidate in (directory / f"{name}.rs", directory / name / "mod.rs"):
        if candidate.is_file():
            return candidate.read_text(encoding="utf-8")
    return ""


def check_crate_naming(violations: list[str]) -> None:
    """Check the naming rules AGENTS.md states for the library crate's source."""
    for path in sorted((CRATE_ROOT / "src").rglob("*.rs")):
        text = path.read_text(encoding="utf-8")
        label = relative_label(path)
        for match in COUNT_PREFIX_NAME.finditer(text):
            name = match.group(1) or match.group(2)
            number = text.count("\n", 0, match.start()) + 1
            violations.append(
                f"{label}:{number}: `{name}` should end in `_count`; "
                "`num_*` names are not used"
            )
        for match in TEST_MODULE.finditer(text):
            name, delimiter = match.groups()
            body = text[match.end() :] if delimiter == "{" else module_source(path, name)
            if name != "tests" and "#[test]" in body:
                number = text.count("\n", 0, match.start()) + 1
                violations.append(
                    f"{label}:{number}: test module `{name}` must be named `tests`"
                )


def check_facade_layout(violations: list[str]) -> None:
    """Keep namespace modules free of items and the crate root one `pub use` per module."""
    for name in NAMESPACE_MODULES:
        path = CRATE_ROOT / "src" / f"{name}.rs"
        offending = [
            number
            for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
            if line
            and line[0] not in " \t)]}"
            and not line.startswith(("//", "#[", "#!["))
            and not re.match(r"(?:pub(?:\([^)]*\))? )?mod [a-z][a-z0-9_]*;", line)
        ]
        if offending:
            violations.append(
                f"{relative_label(path)}:{offending[0]}: namespace module `{name}` only "
                f"declares child modules ({len(offending)} other top-level lines); move items "
                "into a child module and re-export them at the crate root"
            )

    lib_source = (CRATE_ROOT / "src" / "lib.rs").read_text(encoding="utf-8")
    seen: set[str] = set()
    for statement in PUBLIC_USE.findall(lib_source):
        statement = " ".join(statement.split())
        module = statement.split("::{", 1)[0] if "{" in statement else statement.rsplit("::", 1)[0]
        if module in seen:
            violations.append(
                f"crates/cfglib/src/lib.rs: more than one `pub use` from `{module}`; "
                "the root facade re-exports each module in one statement"
            )
        seen.add(module)


def main() -> int:
    """Run all repository-policy checks."""
    try:
        files = repository_files()
        metadata = cargo_metadata()
        workspace_root, package_roots = cargo_package_roots(metadata)
    except (OSError, RuntimeError, json.JSONDecodeError) as error:
        print(f"repository policy check failed to start: {error}", file=sys.stderr)
        return 2

    violations: list[str] = []
    source_count = check_source_sizes(files, violations)
    check_comment_style(files, violations)
    check_cargo_layout(files, workspace_root, package_roots, violations)
    benchmark_api_count = check_benchmark_api_inventory(violations)
    check_msrv_sources(metadata, files, violations)
    check_lint_policy(files, violations)
    check_spelling(files, violations)
    check_crate_naming(violations)
    check_facade_layout(violations)

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

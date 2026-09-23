#!/usr/bin/env python3
"""Generate and verify the committed HTML oracle corpus.

The profile environments are deliberately separate.  From PowerShell, set the
same variables for every command and keep the invocations locked and offline:

    $env:UV_CACHE_DIR = 'C:\\Users\\johnm\\.herdr\\worktrees\\sphinx-ultra\\html-oracle-gen\\.uv-cache'
    $env:UV_PYTHON_INSTALL_DIR = 'C:\\Users\\johnm\\.herdr\\worktrees\\sphinx-ultra\\html-oracle-gen\\.uv-python'
    $env:UV_PYTHON_PREFERENCE = 'only-managed'
    $env:PYTHONNOUSERSITE = '1'

    uv run --locked --offline --python 3.12 --project tools/oracle_profiles/core \
        python -m pytest tools/test_gen_html_oracle.py -q
    uv run --locked --offline --python 3.12 --project tools/oracle_profiles/core \
        python tools/gen_html_oracle.py --config tools/html_oracle_cases.toml \
        --out tests/fixtures/html_oracle --profile core -j 4
    uv run --locked --offline --python 3.12 --project tools/oracle_profiles/local_needs \
        python tools/gen_html_oracle.py --config tools/html_oracle_cases.toml \
        --out tests/fixtures/html_oracle --profile local_needs \
        --needs-root C:\\Users\\johnm\\Documents\\repos\\sphinx-needs -j 4
    uv run --locked --offline --python 3.12 --project tools/oracle_profiles/core \
        python tools/gen_html_oracle.py --config tools/html_oracle_cases.toml \
        --out tests/fixtures/html_oracle --verify --profile core
    uv run --locked --offline --python 3.12 --project tools/oracle_profiles/local_needs \
        python tools/gen_html_oracle.py --config tools/html_oracle_cases.toml \
        --out tests/fixtures/html_oracle --verify --profile local_needs \
        --needs-root C:\\Users\\johnm\\Documents\\repos\\sphinx-needs

The source fixtures are read-only inputs.  The generated profile subtrees are
replaced atomically and can be regenerated independently.
"""

from __future__ import annotations

import hashlib
import ast
import importlib.util
import json
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, TypedDict


CaseStatus = Literal[
    "built",
    "build-error",
    "reference-crash",
    "excluded-network",
    "excluded-plantuml",
]
FileStorage = Literal["input", "ref", "blob"]

STATUSES = frozenset(
    {
        "built",
        "build-error",
        "reference-crash",
        "excluded-network",
        "excluded-plantuml",
    }
)
STORAGES = frozenset({"input", "ref", "blob"})
PROFILE_FIELDS = frozenset(
    {
        "sphinx",
        "docutils",
        "needs_version",
        "needs_commit",
        "needs_tree",
        "lock_path",
        "lock_sha256",
        "determinism_shims",
    }
)
ORIGIN_FIELDS = frozenset(
    {"source_set", "origin_path", "pytest_node_ids", "variants_not_captured"}
)
FILE_FIELDS = frozenset({"logical_path", "storage", "storage_path", "sha256", "size"})
CASE_FIELDS = frozenset(
    {
        "profile",
        "source_set",
        "case_id",
        "status",
        "exit_code",
        "warnings",
        "excluded_reason",
        "origin",
        "input_files",
        "input_sha256",
        "tree_sha256",
        "files",
        "needs_json",
        "needs_status",
        "needs_exit_code",
        "needs_warnings",
    }
)
INDEX_FIELDS = frozenset({"schema_version", "generator", "profiles", "cases"})
HEX64_RE = re.compile(r"^[0-9a-f]{64}$")


class ProfileRecord(TypedDict):
    sphinx: str
    docutils: str
    needs_version: str | None
    needs_commit: str | None
    needs_tree: str | None
    lock_path: str
    lock_sha256: str
    determinism_shims: list[str]


class OriginRecord(TypedDict):
    source_set: str
    origin_path: str
    pytest_node_ids: list[str]
    variants_not_captured: bool


class FileRecord(TypedDict):
    logical_path: str
    storage: FileStorage
    storage_path: str
    sha256: str
    size: int


class CaseRecord(TypedDict):
    profile: str
    source_set: str
    case_id: str
    status: CaseStatus
    exit_code: int | None
    warnings: str
    excluded_reason: str | None
    origin: OriginRecord
    input_files: list[FileRecord]
    input_sha256: str
    tree_sha256: str
    files: list[FileRecord]
    needs_json: FileRecord | None
    needs_status: CaseStatus | None
    needs_exit_code: int | None
    needs_warnings: str | None


class IndexDocument(TypedDict):
    schema_version: int
    generator: str
    profiles: dict[str, ProfileRecord]
    cases: list[CaseRecord]


class SchemaError(ValueError):
    """Raised when an index or one of its referenced artifacts is invalid."""


class DiscoveryError(ValueError):
    """Raised when the configured source corpus cannot be materialized."""


@dataclass
class DiscoveredCase:
    profile: str
    source_set: str
    case_id: str
    origin_path: str
    files: dict[str, bytes]
    pytest_node_ids: list[str] = field(default_factory=list)
    variants_not_captured: bool = False
    status: CaseStatus = "built"
    excluded_reason: str | None = None
    confoverrides: dict[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> tuple[str, str, str]:
        return self.profile, self.source_set, self.case_id


BASE_CONF_PY = (
    "project = 'html-oracle'\n"
    "extensions = []\n"
    "master_doc = 'index'\n"
    "exclude_patterns = ['_build']\n"
    "smartquotes = False\n"
    "keep_warnings = True\n"
)


def _python_conf(overrides: dict[str, Any], *, project: str = "html-oracle") -> bytes:
    lines = [
        f"project = {project!r}",
        "extensions = []",
        "master_doc = 'index'",
        "exclude_patterns = ['_build']",
    ]
    for key in sorted(overrides):
        lines.append(f"{key} = {overrides[key]!r}")
    lines.append("")
    return ("\n".join(lines)).encode("utf-8")


def _safe_case_base(identifier: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_.-]+", "-", identifier).strip("-.")
    return safe or "case"


def assign_case_ids(identifiers: list[str]) -> dict[str, str]:
    """Return deterministic short IDs, suffixing truncations and collisions."""
    if len(set(identifiers)) != len(identifiers):
        raise DiscoveryError("duplicate source identifiers")
    bases = {identifier: _safe_case_base(identifier) for identifier in identifiers}
    groups: dict[str, list[str]] = {}
    for identifier, base in bases.items():
        groups.setdefault(base[:48], []).append(identifier)
    result: dict[str, str] = {}
    for identifier in sorted(identifiers):
        base = bases[identifier]
        needs_suffix = len(base) > 48 or len(groups[base[:48]]) > 1
        if needs_suffix:
            result[identifier] = f"{base[:48]}-{hashlib.sha256(identifier.encode('utf-8')).hexdigest()[:8]}"
        else:
            result[identifier] = base
    if len(set(result.values())) != len(result):
        raise DiscoveryError("case ID collision after sanitization")
    return result


def _read_tree_bytes(source: Path) -> dict[str, bytes]:
    source = source.resolve()
    if not source.is_dir():
        raise DiscoveryError(f"source directory does not exist: {source}")
    files: dict[str, bytes] = {}
    for path in sorted(source.rglob("*")):
        if path.is_symlink():
            raise DiscoveryError(f"symlink is not allowed in source tree: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(source).as_posix()
        _safe_relative_path(relative, f"source path {relative}")
        files[relative] = path.read_bytes()
    return files


def materialize_case(case: DiscoveredCase, destination: Path) -> None:
    """Materialize one discovered case, rejecting symlinked destinations."""
    destination = Path(destination)
    if destination.exists() and destination.is_symlink():
        raise DiscoveryError(f"materialization destination is a symlink: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    for relative, data in sorted(case.files.items()):
        safe = _safe_relative_path(relative, f"case file {relative}")
        path = destination / Path(safe)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.is_symlink():
            raise DiscoveryError(f"materialization destination is a symlink: {path}")
        path.write_bytes(data)


def _literal_strings(tree: ast.AST) -> list[str]:
    return [node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str)]


def classify_project(files: dict[str, bytes]) -> tuple[CaseStatus | None, str | None]:
    """Classify statically excluded projects before importing their conf.py."""
    conf_bytes = files.get("conf.py", b"")
    try:
        tree = ast.parse(conf_bytes.decode("utf-8"), filename="conf.py")
    except (SyntaxError, UnicodeDecodeError):
        tree = ast.Module(body=[], type_ignores=[])
    strings = _literal_strings(tree)
    remote = any(re.search(r"https?://", value) for value in strings)
    plantuml = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            plantuml |= any(alias.name.startswith("sphinxcontrib.plantuml") for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            plantuml |= (node.module or "").startswith("sphinxcontrib.plantuml")
            plantuml |= (node.module == "sphinxcontrib" and any(alias.name == "plantuml" for alias in node.names))
        elif isinstance(node, ast.Call):
            function_name = node.func.attr if isinstance(node.func, ast.Attribute) else None
            if function_name == "setup_extension":
                plantuml |= any(
                    isinstance(argument, ast.Constant)
                    and argument.value == "sphinxcontrib.plantuml"
                    for argument in node.args
                )
        plantuml |= "sphinxcontrib.plantuml" in strings
    if remote:
        return "excluded-network", "conf.py references a remote URL"
    if plantuml:
        return "excluded-plantuml", "conf.py loads sphinxcontrib.plantuml"
    return None, None


def _load_inventory_projects(path: Path) -> list[dict[str, Any]]:
    spec = importlib.util.spec_from_file_location("html_oracle_inventory_fixture", path)
    if spec is None or spec.loader is None:
        raise DiscoveryError(f"cannot import inventory fixture: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    projects = getattr(module, "SPHINX_PROJECTS", None)
    if not isinstance(projects, list):
        raise DiscoveryError("gen_inventory_fixture.py has no SPHINX_PROJECTS list")
    return projects


def _scan_needs_provenance(
    needs_root: Path, project_names: set[str]
) -> dict[str, tuple[list[str], bool]]:
    tests_root = needs_root / "packages" / "sphinx-needs" / "tests"
    matches: dict[str, set[str]] = {name: set() for name in project_names}
    variants_by_project = {name: False for name in project_names}
    for path in sorted(tests_root.rglob("*.py")):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(path))
        except (OSError, UnicodeDecodeError, SyntaxError):
            continue
        relative = path.relative_to(needs_root).as_posix()

        class Visitor(ast.NodeVisitor):
            def __init__(self) -> None:
                self.scopes: list[str] = []

            def _visit_scope(self, node: ast.AST, name: str) -> None:
                self.scopes.append(name)
                segment = ast.get_source_segment(source, node) or ""
                strings = _literal_strings(node)
                for project_name in project_names:
                    needle = f"doc_test/{project_name}".replace("\\", "/")
                    alternate = f"doc_test\\{project_name}"
                    matched = any(
                        project_name in value
                        and ("doc_test" in value or needle in value or alternate in value)
                        for value in strings
                    )
                    if matched:
                        suffix = "::" + "::".join(self.scopes) if self.scopes else ""
                        matches[project_name].add(relative + suffix)
                        if "confoverrides" in segment or re.search(
                            r"\b(?:builder|buildername)\s*=\s*['\"](?!html['\"])", segment
                        ):
                            variants_by_project[project_name] = True
                self.generic_visit(node)
                self.scopes.pop()

            def visit_ClassDef(self, node: ast.ClassDef) -> None:
                self._visit_scope(node, node.name)

            def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
                self._visit_scope(node, node.name)

            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
                self._visit_scope(node, node.name)

        Visitor().visit(tree)
    return {
        name: (sorted(matches[name]), variants_by_project[name])
        for name in sorted(project_names)
    }


def _scan_needs_nodes(needs_root: Path, project_name: str) -> tuple[list[str], bool]:
    """Scan one project name; retained as a small public test helper."""
    return _scan_needs_provenance(needs_root, {project_name})[project_name]


def _needs_root_paths(needs_root: Path) -> tuple[Path, Path]:
    root = Path(needs_root).resolve()
    package_root = root / "packages" / "sphinx-needs"
    source_root = package_root / "src" / "sphinx_needs"
    doc_test_root = package_root / "tests" / "doc_test"
    if not package_root.is_dir() or not source_root.is_dir() or not doc_test_root.is_dir():
        raise DiscoveryError("needs-root must contain packages/sphinx-needs/src and tests/doc_test")
    return root, doc_test_root


def _discover_needs_cases(needs_root: Path, profile: str, expected_count: int) -> list[DiscoveredCase]:
    root, doc_test_root = _needs_root_paths(needs_root)
    directories = [
        path
        for path in sorted(doc_test_root.iterdir())
        if path.is_dir() and (path / "conf.py").is_file()
    ]
    if len(directories) != expected_count:
        raise DiscoveryError(f"expected {expected_count} needs projects, found {len(directories)}")
    ids = assign_case_ids([path.name for path in directories])
    provenance = _scan_needs_provenance(root, {path.name for path in directories})
    cases = []
    for path in directories:
        files = _read_tree_bytes(path)
        status, reason = classify_project(files)
        node_ids, variants = provenance[path.name]
        cases.append(
            DiscoveredCase(
                profile=profile,
                source_set="sphinx_needs_doc_tests",
                case_id=ids[path.name],
                origin_path=path.relative_to(root).as_posix(),
                files=files,
                pytest_node_ids=node_ids,
                variants_not_captured=variants,
                status=status or "built",
                excluded_reason=reason,
            )
        )
    return cases


def _source_file_name(name: str) -> str:
    return name if Path(name).suffix else f"{name}.rst"


def discover_cases(
    repo_root: Path,
    config_path: Path,
    *,
    profile: str,
    needs_root: Path | None = None,
) -> list[DiscoveredCase]:
    """Discover and materialize-in-memory all cases for one profile."""
    repo_root = Path(repo_root).resolve()
    with Path(config_path).open("rb") as stream:
        config = tomllib.load(stream)
    source_sets = config.get("source_sets")
    if not isinstance(source_sets, list):
        raise DiscoveryError("config must contain [[source_sets]]")
    selected = [entry for entry in source_sets if entry.get("profile") == profile]
    if not selected:
        raise DiscoveryError(f"no source sets configured for profile {profile}")
    cases: list[DiscoveredCase] = []
    for entry in selected:
        kind = entry.get("kind")
        source_set = entry["name"]
        source = repo_root / entry["source"]
        if kind == "docutils_snippets" or kind == "sphinx_read_snippets":
            fixture = json.loads(source.read_text(encoding="utf-8"))
            fixture_cases = fixture["cases"]
            if len(fixture_cases) != entry["count"]:
                raise DiscoveryError(f"{source_set}: expected {entry['count']} cases, found {len(fixture_cases)}")
            identifiers = [str(item["name"]) for item in fixture_cases]
            ids = assign_case_ids(identifiers)
            for index, item in enumerate(fixture_cases):
                files = {
                    "conf.py": BASE_CONF_PY.encode("utf-8"),
                    "index.rst": str(item["rst"]).encode("utf-8"),
                }
                cases.append(
                    DiscoveredCase(
                        profile=profile,
                        source_set=source_set,
                        case_id=f"{source_set.removesuffix('_snippets')}-{index + 1:04d}",
                        origin_path=f"{entry['source']}[{index}]",
                        files=files,
                    )
                )
        elif kind == "environment_projects":
            fixture = json.loads(source.read_text(encoding="utf-8"))
            projects = fixture["projects"]
            if len(projects) != entry["project_count"]:
                raise DiscoveryError(f"{source_set}: expected {entry['project_count']} projects, found {len(projects)}")
            document_count = sum(len(project["files"]) for project in projects)
            if document_count < entry["document_floor"]:
                raise DiscoveryError(f"{source_set}: expected at least {entry['document_floor']} documents, found {document_count}")
            ids = assign_case_ids([str(project["name"]) for project in projects])
            for index, project in enumerate(projects):
                files = {"conf.py": _python_conf(project.get("conf", {}))}
                for name, contents in project["files"].items():
                    files[_source_file_name(name)] = str(contents).encode("utf-8")
                for name, contents in project.get("data_files", {}).items():
                    files[name] = str(contents).encode("utf-8")
                status, reason = classify_project(files)
                cases.append(
                    DiscoveredCase(
                        profile=profile,
                        source_set=source_set,
                        case_id=ids[str(project["name"])],
                        origin_path=f"{entry['source']}[{index}]",
                        files=files,
                        status=status or "built",
                        excluded_reason=reason,
                        confoverrides=dict(project.get("conf", {})),
                    )
                )
        elif kind == "html_projects":
            project_names = entry["projects"]
            if len(project_names) != entry["count"]:
                raise DiscoveryError(f"{source_set}: configured project count mismatch")
            for project_name in project_names:
                project_root = source / project_name
                files = _read_tree_bytes(project_root)
                status, reason = classify_project(files)
                cases.append(
                    DiscoveredCase(
                        profile=profile,
                        source_set=source_set,
                        case_id=assign_case_ids([project_name])[project_name],
                        origin_path=f"{entry['source']}/{project_name}",
                        files=files,
                        status=status or "built",
                        excluded_reason=reason,
                    )
                )
        elif kind == "inventory_projects":
            projects = _load_inventory_projects(source)
            if len(projects) != entry["count"]:
                raise DiscoveryError(f"{source_set}: expected {entry['count']} projects, found {len(projects)}")
            ids = assign_case_ids([str(project["name"]) for project in projects])
            for index, project in enumerate(projects):
                files = {"conf.py": _python_conf(project.get("conf", {}), project=project["project"])}
                for name, contents in project["files"].items():
                    files[_source_file_name(name)] = str(contents).encode("utf-8")
                status, reason = classify_project(files)
                cases.append(
                    DiscoveredCase(
                        profile=profile,
                        source_set=source_set,
                        case_id=ids[str(project["name"])],
                        origin_path=f"{entry['source']}:SPHINX_PROJECTS[{index}]",
                        files=files,
                        status=status or "built",
                        excluded_reason=reason,
                        confoverrides={"smartquotes": False, **project.get("conf", {})},
                    )
                )
        elif kind == "sphinx_needs_doc_tests":
            if needs_root is None:
                raise DiscoveryError("--needs-root is required for local_needs discovery")
            cases.extend(_discover_needs_cases(needs_root, profile, entry["count"]))
        else:
            raise DiscoveryError(f"unknown source-set kind: {kind!r}")
    ensure_unique_case_keys(cases)
    return sorted(cases, key=lambda case: case.key)


def ensure_unique_case_keys(cases: list[DiscoveredCase]) -> None:
    keys = [case.key for case in cases]
    if len(keys) != len(set(keys)):
        raise DiscoveryError("duplicate discovery case key")


def assert_discovery_keys_equal(
    cases: list[DiscoveredCase], ledger_cases: list[dict[str, Any]]
) -> None:
    discovered = sorted(case.key for case in cases)
    ledger = sorted(
        (case.get("profile"), case.get("source_set"), case.get("case_id"))
        for case in ledger_cases
    )
    if discovered != ledger:
        raise DiscoveryError("discovery and ledger key sets mismatch")


def canonical_hash(entries: list[tuple[str, str]]) -> str:
    payload = "".join(
        f"{path}\0{digest}\n" for path, digest in sorted(entries)
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise SchemaError(message)


def _exact_fields(value: object, expected: frozenset[str], label: str) -> None:
    _require(isinstance(value, dict), f"{label} must be an object")
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    _require(not missing, f"{label} missing field(s): {', '.join(missing)}")
    _require(not unknown, f"{label} has unknown field(s): {', '.join(unknown)}")


def _string(value: object, label: str) -> str:
    _require(isinstance(value, str), f"{label} must be a string")
    return value


def _optional_string(value: object, label: str) -> None:
    _require(value is None or isinstance(value, str), f"{label} must be string or null")


def _hash(value: object, label: str) -> str:
    text = _string(value, label)
    _require(HEX64_RE.fullmatch(text) is not None, f"{label} must be a SHA-256 hex digest")
    return text


def _safe_relative_path(value: object, label: str) -> str:
    text = _string(value, label)
    _require(text and not Path(text).is_absolute(), f"{label} must be relative")
    _require(not re.match(r"^[A-Za-z]:", text), f"{label} must not have a drive prefix")
    normalized = text.replace("\\", "/")
    parts = normalized.split("/")
    _require(all(part not in {"", ".", ".."} for part in parts), f"{label} contains an unsafe path")
    _require("/" not in normalized or all(part != ".." for part in parts), f"{label} escapes its profile")
    return normalized


def _file_record(
    value: object,
    *,
    label: str,
    profile_root: Path,
    allowed_storages: frozenset[str],
) -> FileRecord:
    _exact_fields(value, FILE_FIELDS, label)
    assert isinstance(value, dict)
    logical_path = _safe_relative_path(value["logical_path"], f"{label}.logical_path")
    storage = _string(value["storage"], f"{label}.storage")
    _require(storage in STORAGES, f"{label}.storage has unknown value {storage!r}")
    _require(storage in allowed_storages, f"{label}.storage {storage!r} is not allowed")
    storage_path = _safe_relative_path(value["storage_path"], f"{label}.storage_path")
    expected_root = {"input": "inputs", "ref": "refs", "blob": "blobs"}[storage]
    _require(
        storage_path == expected_root or storage_path.startswith(expected_root + "/"),
        f"{label}.storage_path must be below {expected_root}/",
    )
    digest = _hash(value["sha256"], f"{label}.sha256")
    size = value["size"]
    _require(isinstance(size, int) and not isinstance(size, bool) and size >= 0, f"{label}.size must be non-negative integer")
    path = profile_root / Path(storage_path)
    _require(path.is_relative_to(profile_root), f"{label}.storage_path escapes its profile")
    _require(path.exists(), f"{label}.storage_path does not exist: {storage_path}")
    _require(not path.is_symlink(), f"{label}.storage_path must not be a symlink: {storage_path}")
    _require(path.is_file(), f"{label}.storage_path is not a file: {storage_path}")
    data = path.read_bytes()
    _require(len(data) == size, f"{label}.size mismatch for {storage_path}")
    actual_digest = hashlib.sha256(data).hexdigest()
    _require(actual_digest == digest, f"{label}.hash mismatch for {storage_path}")
    return {
        "logical_path": logical_path,
        "storage": storage,
        "storage_path": storage_path,
        "sha256": digest,
        "size": size,
    }


def _validate_profile_record(value: object, label: str) -> ProfileRecord:
    _exact_fields(value, PROFILE_FIELDS, label)
    assert isinstance(value, dict)
    sphinx = _string(value["sphinx"], f"{label}.sphinx")
    docutils = _string(value["docutils"], f"{label}.docutils")
    _optional_string(value["needs_version"], f"{label}.needs_version")
    _optional_string(value["needs_commit"], f"{label}.needs_commit")
    _optional_string(value["needs_tree"], f"{label}.needs_tree")
    lock_path = _safe_relative_path(value["lock_path"], f"{label}.lock_path")
    lock_sha256 = _hash(value["lock_sha256"], f"{label}.lock_sha256")
    shims = value["determinism_shims"]
    _require(isinstance(shims, list) and all(isinstance(item, str) for item in shims), f"{label}.determinism_shims must be a string array")
    return {
        "sphinx": sphinx,
        "docutils": docutils,
        "needs_version": value["needs_version"],
        "needs_commit": value["needs_commit"],
        "needs_tree": value["needs_tree"],
        "lock_path": lock_path,
        "lock_sha256": lock_sha256,
        "determinism_shims": list(shims),
    }


def _validate_origin(value: object, label: str) -> OriginRecord:
    _exact_fields(value, ORIGIN_FIELDS, label)
    assert isinstance(value, dict)
    source_set = _string(value["source_set"], f"{label}.source_set")
    origin_path = _safe_relative_path(value["origin_path"], f"{label}.origin_path")
    node_ids = value["pytest_node_ids"]
    _require(isinstance(node_ids, list) and all(isinstance(item, str) for item in node_ids), f"{label}.pytest_node_ids must be a string array")
    variants = value["variants_not_captured"]
    _require(isinstance(variants, bool), f"{label}.variants_not_captured must be boolean")
    return {
        "source_set": source_set,
        "origin_path": origin_path,
        "pytest_node_ids": list(node_ids),
        "variants_not_captured": variants,
    }


def _validate_case(
    value: object,
    *,
    index: int,
    profile_root: Path,
    profiles: dict[str, ProfileRecord],
) -> CaseRecord:
    label = f"cases[{index}]"
    _exact_fields(value, CASE_FIELDS, label)
    assert isinstance(value, dict)
    profile = _string(value["profile"], f"{label}.profile")
    _require(profile in profiles, f"{label}.profile is unknown: {profile}")
    source_set = _string(value["source_set"], f"{label}.source_set")
    case_id = _string(value["case_id"], f"{label}.case_id")
    _require(case_id and re.fullmatch(r"[A-Za-z0-9_.-]+", case_id) is not None, f"{label}.case_id is unsafe")
    status = _string(value["status"], f"{label}.status")
    _require(status in STATUSES, f"{label}.status has unknown value {status!r}")
    exit_code = value["exit_code"]
    _require(exit_code is None or (isinstance(exit_code, int) and not isinstance(exit_code, bool)), f"{label}.exit_code must be integer or null")
    warnings = _string(value["warnings"], f"{label}.warnings")
    excluded_reason = value["excluded_reason"]
    _optional_string(excluded_reason, f"{label}.excluded_reason")
    origin = _validate_origin(value["origin"], f"{label}.origin")
    _require(origin["source_set"] == source_set, f"{label}.origin.source_set differs from case source_set")
    input_values = value["input_files"]
    files_values = value["files"]
    _require(isinstance(input_values, list), f"{label}.input_files must be an array")
    _require(isinstance(files_values, list), f"{label}.files must be an array")
    excluded = status.startswith("excluded-")
    if excluded:
        _require(not input_values and not files_values, f"excluded case {label} must not have files")
    input_files = [
        _file_record(
            item,
            label=f"{label}.input_files[{i}]",
            profile_root=profile_root,
            allowed_storages=frozenset({"input"}),
        )
        for i, item in enumerate(input_values)
    ]
    files = [
        _file_record(
            item,
            label=f"{label}.files[{i}]",
            profile_root=profile_root,
            allowed_storages=frozenset({"ref", "blob"}),
        )
        for i, item in enumerate(files_values)
    ]
    needs_json_value = value["needs_json"]
    needs_json = None
    if needs_json_value is not None:
        needs_json = _file_record(
            needs_json_value,
            label=f"{label}.needs_json",
            profile_root=profile_root,
            allowed_storages=frozenset({"ref"}),
        )
    needs_status = value["needs_status"]
    _require(needs_status is None or needs_status in STATUSES, f"{label}.needs_status has unknown value {needs_status!r}")
    needs_exit_code = value["needs_exit_code"]
    _require(needs_exit_code is None or (isinstance(needs_exit_code, int) and not isinstance(needs_exit_code, bool)), f"{label}.needs_exit_code must be integer or null")
    needs_warnings = value["needs_warnings"]
    _optional_string(needs_warnings, f"{label}.needs_warnings")

    if excluded:
        _require(exit_code is None, f"excluded case {label} must have null exit_code")
        _require(warnings == "", f"excluded case {label} must have empty warnings")
        _require(isinstance(excluded_reason, str) and excluded_reason, f"excluded case {label} needs excluded_reason")
        _require(not input_files and not files and needs_json is None, f"excluded case {label} must not have files")
        _require(needs_status is None and needs_exit_code is None and needs_warnings is None, f"excluded case {label} must not have needs fields")
    else:
        _require(isinstance(exit_code, int), f"{status} case {label} must have exit_code")
        _require(excluded_reason is None, f"{status} case {label} must have null excluded_reason")
        if status == "built":
            _require(exit_code == 0, f"built case {label} must have exit_code 0")
        else:
            _require(exit_code != 0, f"{status} case {label} must have nonzero exit_code")
    if profile == "core":
        _require(needs_json is None and needs_status is None and needs_exit_code is None and needs_warnings is None, f"core case {label} must have null needs fields")

    input_seen: set[str] = set()
    for record in input_files:
        _require(record["logical_path"] not in input_seen, f"duplicate input logical path in {label}: {record['logical_path']}")
        input_seen.add(record["logical_path"])
    file_seen: set[str] = set()
    for record in files:
        _require(record["logical_path"] not in file_seen, f"duplicate output logical path in {label}: {record['logical_path']}")
        file_seen.add(record["logical_path"])
    input_hash = _hash(value["input_sha256"], f"{label}.input_sha256")
    tree_hash = _hash(value["tree_sha256"], f"{label}.tree_sha256")
    _require(input_hash == canonical_hash([(item["logical_path"], item["sha256"]) for item in input_files]), f"{label}.input_sha256 mismatch")
    _require(tree_hash == canonical_hash([(item["logical_path"], item["sha256"]) for item in files]), f"{label}.tree_sha256 mismatch")
    return {
        "profile": profile,
        "source_set": source_set,
        "case_id": case_id,
        "status": status,
        "exit_code": exit_code,
        "warnings": warnings,
        "excluded_reason": excluded_reason,
        "origin": origin,
        "input_files": input_files,
        "input_sha256": input_hash,
        "tree_sha256": tree_hash,
        "files": files,
        "needs_json": needs_json,
        "needs_status": needs_status,
        "needs_exit_code": needs_exit_code,
        "needs_warnings": needs_warnings,
    }


def validate_index_document(document: object, profile_root: Path) -> IndexDocument:
    """Validate an index and all referenced artifacts below one profile root."""
    _exact_fields(document, INDEX_FIELDS, "index")
    assert isinstance(document, dict)
    _require(document["schema_version"] == 1, "index.schema_version must be 1")
    _require(document["generator"] == "html-oracle/1", "index.generator must be html-oracle/1")
    profiles_value = document["profiles"]
    _require(isinstance(profiles_value, dict) and profiles_value, "index.profiles must be a non-empty object")
    profiles = {
        name: _validate_profile_record(value, f"profiles[{name!r}]")
        for name, value in profiles_value.items()
    }
    cases_value = document["cases"]
    _require(isinstance(cases_value, list), "index.cases must be an array")
    profile_root = Path(profile_root).resolve()
    cases: list[CaseRecord] = []
    keys: list[tuple[str, str, str]] = []
    for index, value in enumerate(cases_value):
        _require(isinstance(value, dict), f"cases[{index}] must be an object")
        raw_key = tuple(value.get(field) for field in ("profile", "source_set", "case_id"))
        _require(all(isinstance(part, str) for part in raw_key), f"cases[{index}] key fields must be strings")
        _require(raw_key not in keys, f"duplicate case key: {'/'.join(raw_key)}")
        keys.append(raw_key)
    _require(keys == sorted(keys), "cases must be sorted by profile, source_set, case_id")
    keys = []
    for index, value in enumerate(cases_value):
        case = _validate_case(value, index=index, profile_root=profile_root, profiles=profiles)
        key = (case["profile"], case["source_set"], case["case_id"])
        _require(key not in keys, f"duplicate case key: {'/'.join(key)}")
        keys.append(key)
        cases.append(case)
    return {
        "schema_version": 1,
        "generator": "html-oracle/1",
        "profiles": profiles,
        "cases": cases,
    }

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

from tools.gen_html_oracle import (
    DiscoveryError,
    DiscoveredCase,
    SchemaError,
    assert_discovery_keys_equal,
    StorageError,
    atomic_swap_profile,
    capture_output_tree,
    classify_project,
    discover_cases,
    ensure_unique_case_keys,
    generate_profile,
    validate_needs_metadata,
    validate_index_document,
    store_blob,
)


def _file_record(storage_path: str, content: bytes, *, logical_path: str = "index.rst"):
    return {
        "logical_path": logical_path,
        "storage": "input",
        "storage_path": storage_path,
        "sha256": hashlib.sha256(content).hexdigest(),
        "size": len(content),
    }


def _document(tmp_path: Path) -> dict:
    input_path = tmp_path / "inputs" / "docutils_snippets" / "case" / "index.rst"
    input_path.parent.mkdir(parents=True)
    content = b"Hello\n"
    input_path.write_bytes(content)
    input_record = _file_record(
        "inputs/docutils_snippets/case/index.rst", content
    )
    return {
        "schema_version": 1,
        "generator": "html-oracle/1",
        "profiles": {
            "core": {
                "sphinx": "9.1.0",
                "docutils": "0.22.4",
                "needs_version": None,
                "needs_commit": None,
                "needs_tree": None,
                "lock_path": "tools/oracle_profiles/core/uv.lock",
                "lock_sha256": "0" * 64,
                "determinism_shims": ["uuid.uuid4=counter"],
            }
        },
        "cases": [
            {
                "profile": "core",
                "source_set": "docutils_snippets",
                "case_id": "case",
                "status": "built",
                "exit_code": 0,
                "warnings": "",
                "excluded_reason": None,
                "origin": {
                    "source_set": "docutils_snippets",
                    "origin_path": "tests/fixtures/doctree_differential.json[0]",
                    "pytest_node_ids": [],
                    "variants_not_captured": False,
                },
                "input_files": [input_record],
                "input_sha256": "0" * 64,
                "tree_sha256": "0" * 64,
                "files": [],
                "needs_json": None,
                "needs_status": None,
                "needs_exit_code": None,
                "needs_warnings": None,
            }
        ],
    }


def test_rejects_unknown_status(tmp_path):
    document = _document(tmp_path)
    document["cases"][0]["status"] = "unknown"
    with pytest.raises(SchemaError, match="status"):
        validate_index_document(document, tmp_path)


def test_rejects_missing_input_files(tmp_path):
    document = _document(tmp_path)
    del document["cases"][0]["input_files"]
    with pytest.raises(SchemaError, match="input_files"):
        validate_index_document(document, tmp_path)


def test_rejects_missing_input_hash(tmp_path):
    document = _document(tmp_path)
    del document["cases"][0]["input_sha256"]
    with pytest.raises(SchemaError, match="input_sha256"):
        validate_index_document(document, tmp_path)


def test_rejects_storage_path_escape(tmp_path):
    document = _document(tmp_path)
    document["cases"][0]["input_files"][0]["storage_path"] = (
        "inputs/docutils_snippets/case/../../escape"
    )
    with pytest.raises(SchemaError, match="path"):
        validate_index_document(document, tmp_path)


def test_rejects_hash_mismatch(tmp_path):
    document = _document(tmp_path)
    document["cases"][0]["input_files"][0]["sha256"] = "1" * 64
    with pytest.raises(SchemaError, match="hash"):
        validate_index_document(document, tmp_path)


def test_rejects_duplicate_case_key(tmp_path):
    document = _document(tmp_path)
    duplicate = copy.deepcopy(document["cases"][0])
    document["cases"].append(duplicate)
    with pytest.raises(SchemaError, match="duplicate"):
        validate_index_document(document, tmp_path)


def test_rejects_null_exit_code_for_built(tmp_path):
    document = _document(tmp_path)
    document["cases"][0]["exit_code"] = None
    with pytest.raises(SchemaError, match="exit_code"):
        validate_index_document(document, tmp_path)


def test_rejects_files_on_excluded_case(tmp_path):
    document = _document(tmp_path)
    case = document["cases"][0]
    case["status"] = "excluded-network"
    case["exit_code"] = None
    case["excluded_reason"] = "remote fetch"
    case["input_files"] = []
    case["input_sha256"] = "0" * 64
    case["files"] = [
        {
            "logical_path": "index.html",
            "storage": "ref",
            "storage_path": "refs/docutils_snippets/case/index.html",
            "sha256": "0" * 64,
            "size": 0,
        }
    ]
    with pytest.raises(SchemaError, match="excluded"):
        validate_index_document(document, tmp_path)


def test_discovers_the_complete_core_corpus():
    repo_root = Path(__file__).resolve().parents[1]
    cases = discover_cases(
        repo_root,
        repo_root / "tools" / "html_oracle_cases.toml",
        profile="core",
    )
    counts = {}
    for case in cases:
        counts[case.source_set] = counts.get(case.source_set, 0) + 1
    assert counts == {
        "docutils_snippets": 735,
        "environment_projects": 29,
        "html_projects": 7,
        "inventory_projects": 4,
        "sphinx_read_snippets": 489,
    }
    assert all(case.files for case in cases)


def test_discovers_all_local_needs_projects():
    repo_root = Path(__file__).resolve().parents[1]
    cases = discover_cases(
        repo_root,
        repo_root / "tools" / "html_oracle_cases.toml",
        profile="local_needs",
        needs_root=Path(r"C:\Users\johnm\Documents\repos\sphinx-needs"),
    )
    assert len(cases) == 142
    assert {case.source_set for case in cases} == {"sphinx_needs_doc_tests"}
    assert all(case.files.get("conf.py") for case in cases)


def test_rejects_duplicate_discovery_keys():
    case = DiscoveredCase(
        profile="core",
        source_set="fixture",
        case_id="same",
        origin_path="fixture",
        files={"index.rst": b""},
    )
    with pytest.raises(DiscoveryError, match="duplicate"):
        ensure_unique_case_keys([case, case])


def test_rejects_discovery_ledger_key_mismatch():
    case = DiscoveredCase(
        profile="core",
        source_set="fixture",
        case_id="same",
        origin_path="fixture",
        files={"index.rst": b""},
    )
    with pytest.raises(DiscoveryError, match="mismatch"):
        assert_discovery_keys_equal([case], [])


def test_rejects_needs_root_without_pinned_subtree(tmp_path):
    repo_root = Path(__file__).resolve().parents[1]
    with pytest.raises(DiscoveryError, match="needs-root"):
        discover_cases(
            repo_root,
            repo_root / "tools" / "html_oracle_cases.toml",
            profile="local_needs",
            needs_root=tmp_path,
        )


def test_plantuml_app_extension_is_excluded_and_network_wins():
    files = {
        "conf.py": b"extensions = []\napp.setup_extension('sphinxcontrib.plantuml')\n",
        "index.rst": b"index\n=====\n",
    }
    assert classify_project(files) == ("excluded-plantuml", "conf.py loads sphinxcontrib.plantuml")
    both = {
        "conf.py": b"extensions = ['sphinxcontrib.plantuml']\nintersphinx_mapping = {'x': ('https://example.invalid', None)}\n",
        "index.rst": b"index\n=====\n",
    }
    assert classify_project(both)[0] == "excluded-network"


def _run_runner(tmp_path: Path, *, conf: str, index: str = "Title\n=====\n"):
    source = tmp_path / "src"
    output = tmp_path / "out"
    doctree = tmp_path / "doctree"
    warnings = tmp_path / "warnings.txt"
    source.mkdir()
    (source / "conf.py").write_text(conf, encoding="utf-8")
    (source / "index.rst").write_text(index, encoding="utf-8")
    command = [
        sys.executable,
        str(Path(__file__).with_name("html_oracle_runner.py")),
        "--profile",
        "core",
        "--sourcedir",
        str(source),
        "--outputdir",
        str(output),
        "--doctree-dir",
        str(doctree),
        "--builder",
        "html",
        "--warnings-file",
        str(warnings),
    ]
    return subprocess.run(command, capture_output=True, text=True), output, warnings


def test_child_runner_denies_network_before_sphinx(tmp_path):
    result, output, _warnings = _run_runner(
        tmp_path,
        conf=(
            "project = 'network-test'\n"
            "extensions = []\n"
            "master_doc = 'index'\n"
            "import socket\n"
            "socket.create_connection(('example.invalid', 80))\n"
        ),
    )
    assert result.returncode != 0
    assert "network disabled by html oracle" in result.stderr
    assert not (output / "index.html").exists()


def test_child_runner_uses_pinned_core_versions(tmp_path):
    result, output, warnings = _run_runner(
        tmp_path,
        conf=(
            "project = 'version-test'\n"
            "extensions = []\n"
            "master_doc = 'index'\n"
        ),
    )
    assert result.returncode == 0, result.stderr
    assert (output / "index.html").is_file()
    assert warnings.read_text(encoding="utf-8") == ""


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("module_file", Path("C:/wrong/sphinx_needs/__init__.py")),
        ("module_version", "0.0.0"),
        ("commit", "0" * 40),
        ("tree", "0" * 40),
        ("status", " M packages/sphinx-needs/src/sphinx_needs/__init__.py"),
    ],
)
def test_needs_provenance_rejects_wrong_observation(field, value):
    root = Path(r"C:\Users\johnm\Documents\repos\sphinx-needs")
    observations = {
        "module_file": root / "packages/sphinx-needs/src/sphinx_needs/__init__.py",
        "module_version": "8.5.0",
        "commit": "58bcb59d861da95f2aca79f343e8bae6ec5c1250",
        "tree": "958172a89defcec69704f6b9d61e482e7c4e8409",
        "status": "",
    }
    observations[field] = value
    with pytest.raises(RuntimeError, match="needs provenance"):
        validate_needs_metadata(root, **observations)


def test_needs_provenance_ignores_dirt_outside_package_subtree():
    root = Path(r"C:\Users\johnm\Documents\repos\sphinx-needs")
    validate_needs_metadata(
        root,
        module_file=root / "packages/sphinx-needs/src/sphinx_needs/__init__.py",
        module_version="8.5.0",
        commit="58bcb59d861da95f2aca79f343e8bae6ec5c1250",
        tree="958172a89defcec69704f6b9d61e482e7c4e8409",
        status="",
    )


def test_repeated_static_bytes_use_one_blob(tmp_path):
    blob_root = tmp_path / "blobs"
    first = store_blob(blob_root, b"same static bytes")
    second = store_blob(blob_root, b"same static bytes")
    assert first == second
    assert list(blob_root.iterdir()) == [blob_root / first]


def test_capture_preserves_all_logical_paths_and_changes_hashes(tmp_path):
    output = tmp_path / "output"
    output.joinpath("_static").mkdir(parents=True)
    output.joinpath("_images").mkdir()
    output.joinpath("_static/theme.css").write_bytes(b"same")
    output.joinpath("_images/logo.png").write_bytes(b"same")
    output.joinpath("index.html").write_bytes(b"one")
    records = capture_output_tree(output, tmp_path, "set", "case")
    assert [record["logical_path"] for record in records] == [
        "_images/logo.png",
        "_static/theme.css",
        "index.html",
    ]
    assert {record["storage"] for record in records if record["logical_path"].startswith("_")} == {"blob"}
    index_record = next(record for record in records if record["logical_path"] == "index.html")
    output.joinpath("index.html").write_bytes(b"two")
    changed = capture_output_tree(output, tmp_path, "set", "case")
    changed_record = next(record for record in changed if record["logical_path"] == "index.html")
    assert changed_record["sha256"] != index_record["sha256"]


def test_capture_rejects_symlink_before_file_check(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    target = tmp_path / "target.txt"
    target.write_bytes(b"target")
    try:
        (output / "link.txt").symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(StorageError, match="symlink"):
        capture_output_tree(output, tmp_path, "set", "case")


def test_capture_rejects_absolute_root_leaks(tmp_path):
    output = tmp_path / "output"
    output.mkdir()
    (output / "index.html").write_text(str(tmp_path / "source"), encoding="utf-8")
    with pytest.raises(StorageError, match="root leak"):
        capture_output_tree(
            output,
            tmp_path,
            "set",
            "case",
            root_paths=[tmp_path / "source", tmp_path / "build", tmp_path / "cache"],
        )


def test_atomic_swap_failure_preserves_old_profile(tmp_path, monkeypatch):
    final = tmp_path / "core"
    staging = tmp_path / "core.staging"
    final.mkdir()
    (final / "sentinel").write_text("old", encoding="utf-8")
    staging.mkdir()
    (staging / "sentinel").write_text("new", encoding="utf-8")
    monkeypatch.setenv("HTML_ORACLE_INJECT_FAILURE_AFTER", "0")
    with pytest.raises(RuntimeError, match="injected"):
        atomic_swap_profile(staging, final, case_count=0)
    assert (final / "sentinel").read_text(encoding="utf-8") == "old"


def test_generate_profile_writes_valid_atomic_case_tree(tmp_path):
    config = tmp_path / "cases.toml"
    config.write_text(
        """[[source_sets]]
name = \"html_projects\"
profile = \"core\"
kind = \"html_projects\"
source = \"tests/fixtures\"
projects = [\"basic\"]
count = 1
""",
        encoding="utf-8",
    )
    output = tmp_path / "oracle"
    document = generate_profile(
        REPO_ROOT,
        config,
        output,
        profile="core",
        jobs=1,
    )
    assert len(document["cases"]) == 1
    assert document["cases"][0]["status"] == "built"
    assert (output / "core" / "index.json").is_file()
    assert not (output / "core.staging").exists()

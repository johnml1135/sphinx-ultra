import copy
import hashlib
import json
from pathlib import Path

import pytest

from tools.gen_html_oracle import SchemaError, validate_index_document


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

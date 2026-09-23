mod support {
    pub mod html_oracle;
}

use serde_json::json;
use support::html_oracle::IndexDocument;

fn minimal_case() -> serde_json::Value {
    json!({
        "profile": "core",
        "source_set": "synthetic",
        "case_id": "case-1",
        "status": "built",
        "exit_code": 0,
        "warnings": "",
        "excluded_reason": null,
        "origin": {
            "source_set": "synthetic",
            "origin_path": "synthetic.json[0]",
            "pytest_node_ids": [],
            "variants_not_captured": false
        },
        "input_files": [],
        "input_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "tree_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "files": [],
        "needs_json": null,
        "needs_status": null,
        "needs_exit_code": null,
        "needs_warnings": null
    })
}

fn minimal_index(case_record: serde_json::Value) -> serde_json::Value {
    json!({
        "schema_version": 1,
        "generator": "html-oracle/1",
        "profiles": {
            "core": {
                "sphinx": "9.1.0",
                "docutils": "0.22.4",
                "needs_version": null,
                "needs_commit": null,
                "needs_tree": null,
                "lock_path": "tools/oracle_profiles/core/uv.lock",
                "lock_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
                "determinism_shims": ["uuid.uuid4=counter"]
            }
        },
        "cases": [case_record]
    })
}

#[test]
fn schema_deserializes_complete_index_document() {
    let document = IndexDocument::from_value(minimal_index(minimal_case()))
        .expect("complete schema should deserialize and validate");
    assert_eq!(document.schema_version, 1);
    assert_eq!(document.cases.len(), 1);
}

#[test]
fn schema_rejects_unknown_status() {
    let mut case = minimal_case();
    case["status"] = json!("not-a-status");
    assert!(IndexDocument::from_value(minimal_index(case)).is_err());
}

#[test]
fn schema_rejects_missing_input_files() {
    let mut case = minimal_case();
    case.as_object_mut().unwrap().remove("input_files");
    assert!(IndexDocument::from_value(minimal_index(case)).is_err());
}

#[test]
fn schema_rejects_missing_input_hash() {
    let mut case = minimal_case();
    case.as_object_mut().unwrap().remove("input_sha256");
    assert!(IndexDocument::from_value(minimal_index(case)).is_err());
}

#[test]
fn schema_rejects_path_escape() {
    let mut case = minimal_case();
    case["input_files"] = json!([{
        "logical_path": "index.rst",
        "storage": "input",
        "storage_path": "inputs/../outside/index.rst",
        "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "size": 0
    }]);
    assert!(IndexDocument::from_value(minimal_index(case)).is_err());
}

#[test]
fn schema_rejects_hash_mismatch() {
    let mut case = minimal_case();
    case["input_files"] = json!([{
        "logical_path": "index.rst",
        "storage": "input",
        "storage_path": "inputs/synthetic/case-1/index.rst",
        "sha256": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff",
        "size": 5
    }]);
    let root = tempfile::tempdir().unwrap();
    let profile_root = root.path().join("core");
    std::fs::create_dir_all(profile_root.join("inputs/synthetic/case-1")).unwrap();
    std::fs::write(
        profile_root.join("inputs/synthetic/case-1/index.rst"),
        b"hello",
    )
    .unwrap();
    let index_path = profile_root.join("index.json");
    std::fs::write(
        &index_path,
        serde_json::to_vec(&minimal_index(case)).unwrap(),
    )
    .unwrap();
    assert!(IndexDocument::load(&index_path).is_err());
}

#[test]
fn schema_rejects_duplicate_case_key() {
    let first = minimal_case();
    let second = minimal_case();
    let mut index = minimal_index(first);
    index["cases"] = json!([second, minimal_case()]);
    assert!(IndexDocument::from_value(index).is_err());
}

#[test]
fn schema_rejects_null_exit_code_for_built_case() {
    let mut case = minimal_case();
    case["exit_code"] = serde_json::Value::Null;
    assert!(IndexDocument::from_value(minimal_index(case)).is_err());
}

#[test]
fn schema_rejects_files_on_excluded_case() {
    let mut case = minimal_case();
    case["status"] = json!("excluded-network");
    case["exit_code"] = serde_json::Value::Null;
    case["excluded_reason"] = json!("synthetic exclusion");
    case["files"] = json!([{
        "logical_path": "index.html",
        "storage": "ref",
        "storage_path": "refs/synthetic/case-1/index.html",
        "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
        "size": 0
    }]);
    assert!(IndexDocument::from_value(minimal_index(case)).is_err());
}

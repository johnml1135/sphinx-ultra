mod support {
    pub mod html_oracle;
}

use flate2::write::ZlibEncoder;
use flate2::Compression;
use serde_json::json;
use std::collections::BTreeMap;
use std::io::Write;
use std::path::Path;
use support::html_oracle::{
    compare_file, compare_trees, compare_warnings, diagnose_html, diagnose_inventory,
    diagnose_needs_json, diagnose_searchindex, diagnose_warnings, group_first_divergences,
    parse_keep, IndexDocument, InventoryRecord, Policy,
};

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

#[test]
fn comparator_normalizes_crlf_text_but_not_buildinfo() {
    assert!(compare_file("index.html", b"a\r\nb\r\n", b"a\nb\n", None).is_empty());
    assert_eq!(
        compare_file("environment.buildinfo", b"a\r\nb", b"a\nb", None)[0].category,
        "bytes-value"
    );
    assert_eq!(
        support::html_oracle::policy_for_path("index.html"),
        Policy::TextCrlf
    );
    assert_eq!(
        support::html_oracle::policy_for_path("environment.buildinfo"),
        Policy::ExactBytes
    );
}

#[test]
fn comparator_replaces_source_root_only_for_warnings() {
    let expected = b"/oracle/source/index.rst:1: WARNING: issue\r\n";
    let actual = b"C:\\run\\source/index.rst:1: WARNING: issue\n";
    assert!(compare_warnings(
        expected,
        actual,
        Some(Path::new("/oracle/source")),
        Some(Path::new("C:\\run\\source")),
    )
    .is_empty());
    assert_eq!(
        compare_file("index.html", b"/oracle/source", b"C:\\run\\source", None)[0].category,
        "text-value"
    );
}

#[test]
fn comparator_parses_searchindex_and_needs_json_with_key_order_ignored() {
    let search_expected = br#"Search.setIndex({"docnames":["a"],"titles":["A"]});"#;
    let search_actual = br#"Search.setIndex({"titles":["A"],"docnames":["a"]});"#;
    assert!(compare_file("searchindex.js", search_expected, search_actual, None).is_empty());

    let needs_expected = br#"{"versions":[{"needs":{"N-1":{"title":"One","status":"open"}}}]}"#;
    let needs_actual = br#"{"versions":[{"needs":{"N-1":{"status":"open","title":"One"}}}]}"#;
    assert!(compare_file("needs.json", needs_expected, needs_actual, None).is_empty());
}

#[test]
fn comparator_preserves_array_order_and_rejects_malformed_wrappers() {
    let expected = br#"Search.setIndex({"docnames":["a","b"]});"#;
    let reordered = br#"Search.setIndex({"docnames":["b","a"]});"#;
    assert_eq!(
        compare_file("searchindex.js", expected, reordered, None)[0].category,
        "searchindex-value"
    );
    let malformed = br#"{"docnames":[]}"#;
    assert_eq!(
        compare_file("searchindex.js", expected, malformed, None)[0].category,
        "invalid-searchindex"
    );
}

fn inventory_bytes(records: &str) -> Vec<u8> {
    let header = b"# Sphinx inventory version 2\n# Project: synthetic\n# Version: 1\n# The remainder of this file is compressed using zlib.\n";
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder.write_all(records.as_bytes()).unwrap();
    let mut bytes = header.to_vec();
    bytes.extend(encoder.finish().unwrap());
    bytes
}

#[test]
fn comparator_canonicalizes_inventory_records_and_keeps_opaque_bytes_exact() {
    let first = inventory_bytes("alpha py:function 1 a.html -\nbeta py:function 1 b.html Beta\n");
    let second = inventory_bytes("beta py:function 1 b.html Beta\nalpha py:function 1 a.html -\n");
    assert!(compare_file("objects.inv", &first, &second, None).is_empty());
    assert_eq!(
        compare_file("image.png", b"\x00\x01", b"\x00\x02", None)[0].category,
        "bytes-value"
    );
}

#[test]
fn comparator_reports_missing_and_unexpected_files() {
    let expected = BTreeMap::from([(String::from("index.html"), b"same".to_vec())]);
    let actual = BTreeMap::from([
        (String::from("index.html"), b"same".to_vec()),
        (String::from("extra.css"), b"extra".to_vec()),
    ]);
    let diagnostics = compare_trees(&expected, &actual, None, None, None, None);
    assert_eq!(diagnostics.len(), 1);
    assert_eq!(diagnostics[0].category, "unexpected-file");
    assert_eq!(diagnostics[0].logical_path, "extra.css");

    let actual = BTreeMap::new();
    let diagnostics = compare_trees(&expected, &actual, None, None, None, None);
    assert_eq!(diagnostics[0].category, "missing-file");
}

#[test]
fn diagnostic_synthetic_html_localization_classifies_body_and_chrome() {
    let expected = "<html><div class=\"body\" role=\"main\">\nA\n</div><footer>ok</footer></html>";
    let body_changed =
        "<html><div class=\"body\" role=\"main\">\nB\n</div><footer>ok</footer></html>";
    let chrome_changed =
        "<html><div class=\"body\" role=\"main\">\nA\n</div><footer>changed</footer></html>";
    let both_changed =
        "<html><div class=\"body\" role=\"main\">\nB\n</div><footer>changed</footer></html>";
    let unstructured = "<html><main>A</main></html>";
    assert_eq!(diagnose_html(expected, body_changed).category, "html-body");
    assert_eq!(
        diagnose_html(expected, chrome_changed).category,
        "html-chrome"
    );
    assert_eq!(diagnose_html(expected, both_changed).category, "html-both");
    assert_eq!(
        diagnose_html(expected, unstructured).category,
        "html-unstructured"
    );
}

#[test]
fn diagnostic_synthetic_needs_json_is_keyed_by_need_id() {
    let expected = json!({
        "versions": [{
            "version": "1",
            "needs": {
                "N-1": {"title": "One", "status": "open"},
                "N-2": {"title": "Two"}
            }
        }]
    });
    let actual = json!({
        "versions": [{
            "version": "1",
            "needs": {
                "N-1": {"title": "Changed", "status": "open"},
                "N-3": {"title": "Three"}
            }
        }],
        "extra": true
    });
    let diagnostics = diagnose_needs_json(&expected, &actual);
    assert!(diagnostics.iter().any(|item| {
        item.category == "missing-need" && item.logical_path.contains("needs[\"N-2\"]")
    }));
    assert!(diagnostics.iter().any(|item| {
        item.category == "extra-need" && item.logical_path.contains("needs[\"N-3\"]")
    }));
    assert!(diagnostics.iter().any(|item| {
        item.category == "need-field"
            && item.logical_path == "versions[0].needs[\"N-1\"]"
            && item.detail.contains("title")
    }));
    assert!(diagnostics
        .iter()
        .any(|item| item.category == "needs-top-level"));
    assert_eq!(
        diagnose_needs_json(&json!({"versions": "bad"}), &actual)[0].category,
        "needs-json-path"
    );
}

#[test]
fn diagnostic_synthetic_structured_files_report_key_and_record_differences() {
    let expected_search = json!({"docnames": ["a"], "missing": false, "titles": ["A"]});
    let actual_search = json!({"docnames": ["a"], "extra": true, "titles": ["B"]});
    let search = diagnose_searchindex(&expected_search, &actual_search);
    assert!(search
        .iter()
        .any(|item| item.category == "searchindex-missing-key"));
    assert!(search
        .iter()
        .any(|item| item.category == "searchindex-extra-key"));
    assert!(search
        .iter()
        .any(|item| item.category == "searchindex-changed-key"));

    let record = |name: &str, uri: &str, display_name: &str| InventoryRecord {
        name: name.to_string(),
        domain_role: "py:function".to_string(),
        priority: 1,
        uri: uri.to_string(),
        display_name: display_name.to_string(),
    };
    let inventory = diagnose_inventory(
        &[
            record("missing", "missing.html", "-"),
            record("changed", "old.html", "Old"),
        ],
        &[
            record("extra", "extra.html", "-"),
            record("changed", "new.html", "New"),
        ],
    );
    assert!(inventory
        .iter()
        .any(|item| item.category == "inventory-missing-record"));
    assert!(inventory
        .iter()
        .any(|item| item.category == "inventory-extra-record"));
    assert!(inventory
        .iter()
        .any(|item| item.category == "inventory-changed-record"));
}

#[test]
fn diagnostic_synthetic_warnings_and_first_divergence_are_grouped() {
    let warnings = diagnose_warnings("one\r\ntwo\nthree\n", "one\ntwo-changed\n");
    assert!(warnings
        .iter()
        .any(|item| item.category == "warning-changed-line"));
    assert!(warnings
        .iter()
        .any(|item| item.category == "warning-missing-line"));
    let diagnostics = (0..3)
        .map(|index| support::html_oracle::Diagnostic {
            category: "html-body".to_string(),
            logical_path: format!("page-{index}.html"),
            first_expected_line: Some(7),
            expected: String::new(),
            actual: String::new(),
            detail: String::new(),
        })
        .collect::<Vec<_>>();
    let groups = group_first_divergences(&diagnostics);
    assert_eq!(groups.len(), 1);
    assert_eq!(groups[0].expected_line, 7);
    assert_eq!(groups[0].count, 3);
}

#[test]
fn diagnostic_synthetic_keep_and_filter_policy_defaults_and_rejects_invalid_values() {
    assert_eq!(parse_keep(None).unwrap().as_str(), "failed");
    assert_eq!(parse_keep(Some("all")).unwrap().as_str(), "all");
    assert!(parse_keep(Some("passing")).is_err());
    assert_eq!(
        support::html_oracle::rerun_filter("core/synthetic/case-1", Some("synthetic")),
        Some("core/synthetic/case-1".to_string())
    );
    assert_eq!(
        support::html_oracle::rerun_filter("core/synthetic/case-1", Some("other")),
        None
    );
}

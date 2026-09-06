//! Differential test: our RST parser vs the SPHINX ORACLE — the pseudo-XML a
//! real `sphinx-build` 9.1.0 read phase (dummy builder, `extensions = []`,
//! smartquotes off, keep_warnings on) produces for the committed fixture corpus.
//!
//! Regenerate the fixture (manual, never in CI):
//!     PYTHONNOUSERSITE=1 uv run --python 3.12 --with 'sphinx==9.1.0' \
//!         --with 'docutils==0.22.4' \
//!         python tools/gen_sphinx_fixture.py
//!
//! Clones the tests/doctree_differential.rs shape: committed JSON, version
//! assertions (BOTH sphinx and docutils are recorded), floor guard against
//! silent truncation, collect ALL mismatches before asserting, and panics
//! surface as named mismatches, not test aborts. The fixture's source paths
//! are normalized to the "<snippet>" token; ParseOptions.source_path below
//! must use the same token.
//!
//! Per-case config (wave-4.5 task 8): a case may carry a `conf` dict — the
//! confoverrides the generator applied for that case. Every key maps onto
//! `ParseOptions.py` ([`sphinx_ultra::py::PySigConfig`]); an unmapped key is
//! a hard error so a future generator-side conf addition fails HERE instead
//! of silently parsing under defaults (serde ignores unknown struct fields,
//! so without the explicit map a conf case would quietly lose its config).

use std::collections::BTreeMap;

use sphinx_ultra::py::PySigConfig;
use sphinx_ultra::rst::{parse_rst, ParseOptions};

#[derive(serde::Deserialize)]
struct Fixture {
    docutils_version: String,
    sphinx_version: String,
    cases: Vec<Case>,
}

#[derive(serde::Deserialize)]
struct Case {
    name: String,
    rst: String,
    pseudo_xml: String,
    #[serde(default)]
    conf: BTreeMap<String, serde_json::Value>,
}

/// Map a fixture case's `conf` dict onto the [`PySigConfig`] the parse layer
/// consumes. Errors on any key (or value shape) it does not understand.
fn py_config_from_conf(conf: &BTreeMap<String, serde_json::Value>) -> Result<PySigConfig, String> {
    use serde_json::Value;

    fn opt_i64(key: &str, value: &Value) -> Result<Option<i64>, String> {
        match value {
            Value::Null => Ok(None),
            Value::Number(n) => n
                .as_i64()
                .map(Some)
                .ok_or_else(|| format!("conf key {key}: non-integer number {n}")),
            other => Err(format!("conf key {key}: expected integer, got {other}")),
        }
    }
    fn boolean(key: &str, value: &Value) -> Result<bool, String> {
        value
            .as_bool()
            .ok_or_else(|| format!("conf key {key}: expected bool, got {value}"))
    }
    fn string(key: &str, value: &Value) -> Result<String, String> {
        value
            .as_str()
            .map(str::to_owned)
            .ok_or_else(|| format!("conf key {key}: expected string, got {value}"))
    }

    let mut py = PySigConfig::default();
    for (key, value) in conf {
        match key.as_str() {
            "maximum_signature_line_length" => {
                py.maximum_signature_line_length = opt_i64(key, value)?;
            }
            "python_maximum_signature_line_length" => {
                py.python_maximum_signature_line_length = opt_i64(key, value)?;
            }
            "python_trailing_comma_in_multi_line_signatures" => {
                py.python_trailing_comma_in_multi_line_signatures = boolean(key, value)?;
            }
            "python_display_short_literal_types" => {
                py.python_display_short_literal_types = boolean(key, value)?;
            }
            "python_use_unqualified_type_names" => {
                py.python_use_unqualified_type_names = boolean(key, value)?;
            }
            "toc_object_entries" => py.toc_object_entries = boolean(key, value)?,
            "toc_object_entries_show_parents" => {
                py.toc_object_entries_show_parents = string(key, value)?;
            }
            "add_function_parentheses" => py.add_function_parentheses = boolean(key, value)?,
            "add_module_names" => py.add_module_names = boolean(key, value)?,
            "strip_signature_backslash" => py.strip_signature_backslash = boolean(key, value)?,
            other => {
                return Err(format!(
                    "unmapped conf key {other:?}: teach py_config_from_conf about it \
                     (and the parse layer, if it is not a PySigConfig knob)"
                ));
            }
        }
    }
    Ok(py)
}

#[test]
fn an_unmapped_conf_key_fails() {
    let mut conf = BTreeMap::new();
    conf.insert(
        "python_no_such_setting".to_string(),
        serde_json::Value::Bool(true),
    );
    let err = py_config_from_conf(&conf).unwrap_err();
    assert!(
        err.contains("unmapped conf key \"python_no_such_setting\""),
        "unexpected error text: {err}"
    );

    // A mapped key with the wrong value shape fails too.
    let mut conf = BTreeMap::new();
    conf.insert(
        "add_function_parentheses".to_string(),
        serde_json::Value::String("yes".to_string()),
    );
    assert!(py_config_from_conf(&conf).is_err());
}

#[test]
fn a_mapped_conf_translates_onto_py_sig_config() {
    let conf: BTreeMap<String, serde_json::Value> = serde_json::from_str(
        r#"{
            "maximum_signature_line_length": 8,
            "python_maximum_signature_line_length": null,
            "python_trailing_comma_in_multi_line_signatures": false,
            "python_use_unqualified_type_names": true,
            "add_function_parentheses": false
        }"#,
    )
    .unwrap();
    let py = py_config_from_conf(&conf).unwrap();
    assert_eq!(
        py,
        PySigConfig {
            maximum_signature_line_length: Some(8),
            python_maximum_signature_line_length: None,
            python_trailing_comma_in_multi_line_signatures: false,
            python_use_unqualified_type_names: true,
            add_function_parentheses: false,
            ..PySigConfig::default()
        }
    );
}

#[test]
fn matches_sphinx_oracle_pformat() {
    let raw = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/fixtures/sphinx_doctree_differential.json"
    ));
    let fixture: Fixture = serde_json::from_str(raw).expect("fixture parses");
    assert_eq!(fixture.docutils_version, "0.22.4");
    assert_eq!(fixture.sphinx_version, "9.1.0");
    // Consumer-side anti-truncation floor, raised with the generator's in
    // wave-4.5 task 16 (both were set against a much smaller corpus and had
    // gone slack: 300 here against 426 committed cases). The generator
    // carries the matching global floor plus per-family ones.
    assert!(
        fixture.cases.len() >= 400,
        "fixture truncated? only {} cases",
        fixture.cases.len()
    );

    let mut mismatches = Vec::new();
    for case in &fixture.cases {
        let py = match py_config_from_conf(&case.conf) {
            Ok(py) => py,
            Err(err) => {
                mismatches.push(format!("[{}] CONF ERROR: {err}", case.name));
                continue;
            }
        };
        let rst = case.rst.clone();
        let ours = std::panic::catch_unwind(move || {
            parse_rst(
                &rst,
                &ParseOptions {
                    source_path: "<snippet>".into(),
                    sphinx: true,
                    docname: "index".into(),
                    exclude_patterns: Vec::new(),
                    py,
                    found_docs: None,
                    srcdir: None,
                    ..Default::default()
                },
            )
            .root
            .pformat()
        });
        match ours {
            Err(_) => mismatches.push(format!("[{}] PANICKED on:\n{}", case.name, case.rst)),
            Ok(got) if got != case.pseudo_xml => mismatches.push(format!(
                "[{}] MISMATCH\n--- rst ---\n{}\n--- sphinx 9.1.0 ---\n{}\n--- ours ---\n{}",
                case.name, case.rst, case.pseudo_xml, got
            )),
            Ok(_) => {}
        }
    }
    assert!(
        mismatches.is_empty(),
        "{} divergence(s) from the sphinx 9.1.0 oracle:\n\n{}",
        mismatches.len(),
        mismatches.join("\n\n")
    );
}

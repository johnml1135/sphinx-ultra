//! Property tests for the M2 RST parser (ROADMAP §10.6: the parser is total
//! — it never panics on arbitrary input; problems become system_message
//! nodes). First real use of the reserved proptest dev-dependency.
//!
//! Wave 4.5 extended the sweep over the surfaces this milestone added — the
//! py-domain object descriptions and their signature grammar, the std-domain
//! descriptions wave 4 left uncovered, the file-inserting `include` /
//! `literalinclude` family, and the annotation parser reached directly.
//!
//! GENERATOR RULE (learned the expensive way in task 15): a `.`-based regex
//! generator never produces a newline unless it carries `(?s)`, so a sweep
//! written without it silently tests only single-line inputs — exactly the
//! shape these multi-line grammars are least likely to break on. Every
//! free-text generator below is either `(?s)`-flagged or built by joining
//! generated lines with `\n`.

use std::path::Path;
use std::sync::OnceLock;

use proptest::prelude::*;
use sphinx_ultra::py::annotations::{parse_annotation, PyRefContext};
use sphinx_ultra::py::PySigConfig;
use sphinx_ultra::rst::{parse_rst, ParseOptions};

fn opts() -> ParseOptions {
    ParseOptions {
        source_path: "<p>".into(),
        sphinx: true,
        docname: "index".into(),
        exclude_patterns: Vec::new(),
        py: Default::default(),
        srcdir: None,
        found_docs: None,
        ..Default::default()
    }
}

/// The include-argument generator for the arbitrary-path sweep: control
/// characters and newlines (the `(?s)` arms), printable text, and the
/// path shapes a `\\PC` regex can never produce — empty, absolute,
/// `..`-traversing (both into a real file outside the scratch project and
/// into nothing), and a genuine member with a traversal prefix.
fn include_argument() -> impl Strategy<Value = String> {
    prop_oneof![
        4 => "(?s).{0,40}",
        2 => "\\PC{0,40}",
        1 => "(?s)[\\x00-\\x1f\\x7f]{1,8}",
        1 => Just(String::new()),
        1 => Just("/".to_string()),
        1 => Just("/etc/hosts".to_string()),
        1 => Just("/nonexistent/dir/file.rst".to_string()),
        1 => Just("../".repeat(6) + "etc/hosts"),
        1 => Just("../../nope/../member.rst".to_string()),
        1 => Just("./sub/../member.rst".to_string()),
        1 => "(\\.\\./){0,4}[a-z.]{0,10}",
    ]
}

/// Sphinx-mode options rooted at a real source directory, so the
/// file-inserting directives take their real read path (resolution,
/// encoding, filter chain) instead of the srcdir-less short circuit.
fn opts_in(srcdir: &Path) -> ParseOptions {
    ParseOptions {
        source_path: srcdir.join("index.rst").display().to_string(),
        sphinx: true,
        docname: "index".into(),
        exclude_patterns: Vec::new(),
        py: Default::default(),
        srcdir: Some(srcdir.to_path_buf()),
        found_docs: None,
        ..Default::default()
    }
}

/// A scratch project the include sweeps read from: built once and never
/// mutated. It lives in a `static`, and Rust never drops statics, so the
/// `TempDir` guard's cleanup does not run at exit: one directory per test
/// run is deliberately LEAKED in the system temp dir (panel fix round B,
/// minor — an earlier comment claimed it was dropped with the process). It
/// carries the `sphinx-ultra-proptest-` prefix so the leftovers are
/// recognizable and greppable. Its members cover the shapes the filter
/// chain branches on — markers for `:start-after:`/`:end-before:`, python
/// definitions for `:pyobject:`, an empty file (the zero-line
/// `:number-lines:` width edge), a tab-indented file (`:tab-width:` and
/// `:dedent:`), and a non-UTF-8 file (the `:encoding:` failure path).
fn scratch() -> &'static Path {
    static DIR: OnceLock<tempfile::TempDir> = OnceLock::new();
    DIR.get_or_init(|| {
        let dir = tempfile::Builder::new()
            .prefix("sphinx-ultra-proptest-")
            .tempdir()
            .expect("scratch srcdir");
        let p = dir.path();
        std::fs::write(
            p.join("member.rst"),
            "before\n\n.. MARK-START\n\nmember para\n\n- a\n- b\n\n.. MARK-END\n\nafter\n",
        )
        .unwrap();
        std::fs::write(
            p.join("sample.py"),
            "import os\n\n\nclass C:\n    def m(self):\n        return 1\n\n\ndef f(a, b=2):\n    \"\"\"Doc.\"\"\"\n    return a + b\n",
        )
        .unwrap();
        std::fs::write(p.join("empty.txt"), "").unwrap();
        std::fs::write(p.join("tabs.txt"), "\tone\n\t\ttwo\n   three\n").unwrap();
        std::fs::write(p.join("latin1.txt"), [0xE9u8, b'\n']).unwrap();
        std::fs::write(p.join("index.rst"), "placeholder\n").unwrap();
        dir
    })
    .path()
}

/// One indented directive option line, or a blank/garbage line — the
/// option block itself is part of what must never panic.
fn option_line() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("   :literal:\n".to_string()),
        Just("   :code:\n".to_string()),
        Just("   :code: python\n".to_string()),
        Just("   :number-lines:\n".to_string()),
        Just("   :number-lines: 7\n".to_string()),
        Just("   :number-lines: -3\n".to_string()),
        Just("   :encoding: utf-8\n".to_string()),
        Just("   :encoding: latin-1\n".to_string()),
        Just("   :encoding: no-such-codec\n".to_string()),
        Just("   :tab-width: 4\n".to_string()),
        Just("   :tab-width: -1\n".to_string()),
        Just("   :tab-width: 0\n".to_string()),
        Just("   :start-line: 2\n".to_string()),
        Just("   :start-line: -2\n".to_string()),
        Just("   :end-line: 4\n".to_string()),
        Just("   :end-line: 0\n".to_string()),
        Just("   :start-after: MARK-START\n".to_string()),
        Just("   :start-after: nowhere\n".to_string()),
        Just("   :end-before: MARK-END\n".to_string()),
        Just("   :end-before: nowhere\n".to_string()),
        Just("   :parser: rst\n".to_string()),
        Just("   :class: c\n".to_string()),
        Just("   :name: n\n".to_string()),
        Just("   :lines: 1-3\n".to_string()),
        Just("   :lines: 3-1\n".to_string()),
        Just("   :lines: 0\n".to_string()),
        Just("   :lines: 99-\n".to_string()),
        Just("   :lineno-start: 5\n".to_string()),
        Just("   :lineno-start: -5\n".to_string()),
        Just("   :lineno-match:\n".to_string()),
        Just("   :linenos:\n".to_string()),
        Just("   :emphasize-lines: 1,3\n".to_string()),
        Just("   :emphasize-lines: 0\n".to_string()),
        Just("   :emphasize-lines: bogus\n".to_string()),
        Just("   :dedent:\n".to_string()),
        Just("   :dedent: 2\n".to_string()),
        Just("   :dedent: -2\n".to_string()),
        Just("   :prepend: head\n".to_string()),
        Just("   :append: tail\n".to_string()),
        Just("   :language: python\n".to_string()),
        Just("   :language:\n".to_string()),
        Just("   :force:\n".to_string()),
        Just("   :caption:\n".to_string()),
        Just("   :caption: cap\n".to_string()),
        Just("   :diff: sample.py\n".to_string()),
        Just("   :diff: missing.py\n".to_string()),
        Just("   :pyobject: f\n".to_string()),
        Just("   :pyobject: C.m\n".to_string()),
        Just("   :pyobject: nosuch\n".to_string()),
        Just("   :start-at: class C\n".to_string()),
        Just("   :end-at: return 1\n".to_string()),
        Just("   :no-such-option: x\n".to_string()),
        Just("   :literal\n".to_string()),
        Just("\n".to_string()),
        Just("   not an option\n".to_string()),
    ]
}

/// The file arguments the include sweeps use. A fixed set on purpose: a
/// generated path must never be able to name a file outside the scratch
/// project.
fn include_target() -> impl Strategy<Value = String> {
    prop_oneof![
        Just("member.rst".to_string()),
        Just("sample.py".to_string()),
        Just("empty.txt".to_string()),
        Just("tabs.txt".to_string()),
        Just("latin1.txt".to_string()),
        Just("missing.rst".to_string()),
        Just("/member.rst".to_string()),
        Just("index.rst".to_string()),
        Just(String::new()),
    ]
}

proptest! {
    #![proptest_config(ProptestConfig { cases: 512, ..ProptestConfig::default() })]

    #[test]
    fn parse_never_panics_on_arbitrary_input(s in "\\PC*") {
        let _ = parse_rst(&s, &opts());
    }

    #[test]
    fn parse_never_panics_on_rst_shaped_input(
        s in proptest::collection::vec(
            prop_oneof![
                Just("Title\n=====\n".to_string()),
                Just("=====\nOver\n=====\n".to_string()),
                Just("- item\n".to_string()),
                Just("-\n".to_string()),
                Just("1. item\n".to_string()),
                Just("(i) item\n".to_string()),
                Just("#. item\n".to_string()),
                Just("   indented\n".to_string()),
                Just("  half\n".to_string()),
                Just("::\n".to_string()),
                Just("para::\n".to_string()),
                Just(".. _t:\n".to_string()),
                Just(".. _t: uri\n".to_string()),
                Just(".. comment\n".to_string()),
                Just("..\n".to_string()),
                Just("__ uri\n".to_string()),
                Just("| line\n".to_string()),
                Just("|\n".to_string()),
                Just(">>> code\n".to_string()),
                Just("term\n    def\n".to_string()),
                Just("term : c\n    def\n".to_string()),
                Just("*emph* and ``lit`` text\n".to_string()),
                Just("*unclosed here\n".to_string()),
                Just("`phrase ref`_ and word_ and anon__\n".to_string()),
                Just(":role:`text` and `bare`\n".to_string()),
                Just(":bogus:`x` end\n".to_string()),
                Just("[1]_ [#]_ [*]_ [cite]_\n".to_string()),
                Just("|sub| and |sub2|__\n".to_string()),
                Just("https://example.com/ and foo@bar.example\n".to_string()),
                Just(".. [1] footnote\n".to_string()),
                Just(".. [#lbl] auto\n".to_string()),
                Just(":field: value\n".to_string()),
                Just("-a  option desc\n".to_string()),
                Just("+----+----+\n".to_string()),
                Just("| A  | B  |\n".to_string()),
                Just("+====+====+\n".to_string()),
                Just("=====  =====\n".to_string()),
                Just("A      B\n".to_string()),
                Just("_`inline target` here\n".to_string()),
                Just("`text <https://x/>`_ ref\n".to_string()),
                Just("----\n".to_string()),
                Just("---\n".to_string()),
                Just("-- attribution\n".to_string()),
                Just("\n".to_string()),
                Just("\t\ttabs\n".to_string()),
                Just("> quoted\n".to_string()),
                Just("text\n".to_string()),
            ], 0..40).prop_map(|v| v.concat())
    ) {
        let _ = parse_rst(&s, &opts());
    }

    #[test]
    fn parse_handles_multibyte_boundaries(s in "[αβ✓🎉a\\-=\\n \\|•‣⁃ß]{0,200}") {
        let _ = parse_rst(&s, &opts());
    }

    #[test]
    fn pformat_never_panics_after_parse(s in "\\PC{0,300}") {
        let tree = parse_rst(&s, &opts());
        let _ = tree.root.pformat();
    }

    #[test]
    fn parse_never_panics_on_multiline_arbitrary_input(
        v in proptest::collection::vec("\\PC{0,40}", 0..30)
    ) {
        let s = v.join("\n");
        let _ = parse_rst(&s, &opts());
    }

    // ------------------------------------------------------------------
    // wave 4.5: object descriptions (py + std), signatures, annotations
    // ------------------------------------------------------------------

    /// Arbitrary py signatures through every py object directive, with an
    /// arbitrary option block and an arbitrary doc-field body. The
    /// signature generator is `(?s)`-flagged, so a "signature" here really
    /// can be several lines — the multi-line signature path
    /// (`\`-continuation, blank-line termination) is the point.
    #[test]
    fn py_directives_never_panic_on_arbitrary_signatures(
        kind in prop_oneof![
            Just("py:function"), Just("py:data"), Just("py:class"),
            Just("py:exception"), Just("py:method"), Just("py:classmethod"),
            Just("py:staticmethod"), Just("py:attribute"), Just("py:property"),
            Just("py:type"), Just("py:decorator"), Just("py:decoratormethod"),
            Just("py:module"), Just("py:currentmodule"),
        ],
        sig in "(?s).{0,70}",
        options in proptest::collection::vec(
            prop_oneof![
                Just("   :no-index:\n".to_string()),
                Just("   :no-index-entry:\n".to_string()),
                Just("   :no-contents-entry:\n".to_string()),
                Just("   :no-typesetting:\n".to_string()),
                Just("   :module: pkg.mod\n".to_string()),
                Just("   :module:\n".to_string()),
                Just("   :canonical: pkg.mod.Other\n".to_string()),
                Just("   :async:\n".to_string()),
                Just("   :abstractmethod:\n".to_string()),
                Just("   :classmethod:\n".to_string()),
                Just("   :staticmethod:\n".to_string()),
                Just("   :final:\n".to_string()),
                Just("   :type: int\n".to_string()),
                Just("   :value: 3\n".to_string()),
                Just("   :platform: Unix\n".to_string()),
                Just("   :synopsis: s\n".to_string()),
                Just("   :deprecated:\n".to_string()),
                Just("   :single-line-parameter-list:\n".to_string()),
                Just("   :single-line-type-parameter-list:\n".to_string()),
                Just("   :bogus: x\n".to_string()),
            ], 0..5),
        body in proptest::collection::vec(
            prop_oneof![
                Just("   :param x: a value\n".to_string()),
                Just("   :param int x: a typed value\n".to_string()),
                Just("   :type x: int\n".to_string()),
                Just("   :returns: something\n".to_string()),
                Just("   :rtype: int\n".to_string()),
                Just("   :raises ValueError: when bad\n".to_string()),
                Just("   :var y: a variable\n".to_string()),
                Just("   :meta private:\n".to_string()),
                Just("   :unknown field: text\n".to_string()),
                Just("   :param:\n".to_string()),
                Just("   body paragraph\n".to_string()),
                Just("\n".to_string()),
            ], 0..6),
    ) {
        let mut src = format!(".. {kind}:: {sig}\n");
        for line in &options {
            src.push_str(line);
        }
        src.push('\n');
        for line in &body {
            src.push_str(line);
        }
        let tree = parse_rst(&src, &opts());
        let _ = tree.root.pformat();
    }

    /// The wave-4 std-domain description surface, which the totality sweep
    /// never covered either (final-panel backlog item). Same shape as the
    /// py sweep: arbitrary signature, arbitrary options, arbitrary fields.
    ///
    /// GENERATOR RULE, second edition (fix round 1): the fixed body lines
    /// below are all ASCII at three fixed indents, so the one branch that
    /// actually broke totality — `glossary`'s `line[indent_len:]` slicing a
    /// continuation line indented LESS than the entry's first definition
    /// line, with a multi-byte character straddling the offset — was
    /// unreachable by construction. The last arm draws an arbitrary indent
    /// (1..10, so the block's common indent varies and post-dedent lines
    /// land at every relative depth) over text that can carry 2-, 3- and
    /// 4-byte characters.
    #[test]
    fn std_directives_never_panic_on_arbitrary_signatures(
        kind in prop_oneof![
            Just("describe"), Just("object"), Just("envvar"), Just("confval"),
            Just("option"), Just("cmdoption"), Just("glossary"), Just("productionlist"),
        ],
        sig in "(?s).{0,70}",
        body in proptest::collection::vec(
            prop_oneof![
                Just("   :type: int\n".to_string()),
                Just("   :default: 3\n".to_string()),
                Just("   :param x: a value\n".to_string()),
                Just("   :meta private:\n".to_string()),
                Just("   term\n".to_string()),
                Just("      definition\n".to_string()),
                Just("   .. a comment\n".to_string()),
                Just("   body\n".to_string()),
                Just("\n".to_string()),
                (1usize..10, "[a-zé漢🐍 .:]{0,10}")
                    .prop_map(|(n, t)| format!("{}{t}\n", " ".repeat(n))),
            ], 0..8),
    ) {
        let mut src = format!(".. {kind}:: {sig}\n\n");
        for line in &body {
            src.push_str(line);
        }
        let tree = parse_rst(&src, &opts());
        let _ = tree.root.pformat();
    }

    /// The annotation parser reached directly, without a directive around
    /// it: `parse_annotation` runs the py expression parser and falls back
    /// to a plain-text node, and neither path may panic.
    #[test]
    fn parse_annotation_never_panics(s in "(?s).{0,80}") {
        let ctx = PyRefContext { module: Some("m".into()), class_: Some("C".into()), ..Default::default() };
        for cfg in [PySigConfig::default(), PySigConfig {
            python_use_unqualified_type_names: true,
            python_display_short_literal_types: true,
            ..PySigConfig::default()
        }] {
            for node in parse_annotation(&s, &ctx, &cfg) {
                let _ = node.pformat();
            }
        }
    }

    /// Signature text reached through the directive with the two config
    /// families that change the signature grammar's output shape.
    #[test]
    fn py_signature_config_variants_never_panic(sig in "(?s).{0,60}") {
        let mut o = opts();
        o.py = PySigConfig {
            maximum_signature_line_length: Some(1),
            python_maximum_signature_line_length: Some(1),
            add_function_parentheses: false,
            python_use_unqualified_type_names: true,
            python_display_short_literal_types: true,
            ..PySigConfig::default()
        };
        let tree = parse_rst(&format!(".. py:function:: {sig}\n"), &o);
        let _ = tree.root.pformat();
    }

    // ------------------------------------------------------------------
    // wave 4.5: the file-inserting directives
    // ------------------------------------------------------------------

    /// Arbitrary `include` option combinations against the scratch project.
    /// Options are drawn WITH repetition and without filtering, so mutually
    /// contradictory blocks (`:literal:` + `:code:`, `:start-line:` past
    /// `:end-line:`, an unknown codec, a missing file) are all in range.
    #[test]
    fn include_never_panics_on_arbitrary_option_blocks(
        target in include_target(),
        options in proptest::collection::vec(option_line(), 0..6),
        tail in "(?s).{0,40}",
    ) {
        let mut src = format!(".. include:: {target}\n");
        for line in &options {
            src.push_str(line);
        }
        src.push('\n');
        src.push_str(&tail);
        let tree = parse_rst(&src, &opts_in(scratch()));
        let _ = tree.root.pformat();
    }

    /// The same sweep for `literalinclude`, whose filter chain (`:lines:`,
    /// `:pyobject:`, `:start-at:`/`:end-before:`, `:diff:`, `:dedent:`,
    /// `:prepend:`/`:append:`) is the longer one.
    #[test]
    fn literalinclude_never_panics_on_arbitrary_option_blocks(
        target in include_target(),
        options in proptest::collection::vec(option_line(), 0..6),
    ) {
        let mut src = format!(".. literalinclude:: {target}\n");
        for line in &options {
            src.push_str(line);
        }
        let tree = parse_rst(&src, &opts_in(scratch()));
        let _ = tree.root.pformat();
    }

    /// Arbitrary text as the include argument, against a real srcdir. The
    /// property is TOTALITY and nothing more: whatever the argument —
    /// printable Unicode, control characters and newlines (`(?s).` draws
    /// the whole Unicode range, which `\PC` never does), an empty
    /// argument, an absolute path, a `..` traversal that leaves the
    /// project, a real member — path resolution and the read either splice
    /// content or degrade to a `system_message`, never a panic. It does
    /// NOT pin "never reads outside the project": neither sphinx nor this
    /// crate has that property (docutils' `Include` opens whatever path
    /// `relfn2path` produces, `..` and absolute forms included), and a
    /// traversal case is drawn here precisely so the read path past the
    /// srcdir is exercised.
    #[test]
    fn include_never_panics_on_arbitrary_paths(arg in include_argument()) {
        let src = format!(".. include:: {arg}\n");
        let tree = parse_rst(&src, &opts_in(scratch()));
        let _ = tree.root.pformat();
    }
}

#[test]
fn deep_nesting_does_not_overflow_stack() {
    // 6000 levels of nested bullet lists: far past the MAX_NEST_DEPTH guard
    // (200), which drops deeper content with an ERROR message instead of
    // overflowing the stack (docutils crashes with RecursionError here).
    let mut s = String::new();
    for depth in 0..6000 {
        s.push_str(&"  ".repeat(depth));
        s.push_str("- x\n");
    }
    let tree = parse_rst(&s, &opts());
    let out = tree.root.pformat();
    assert_eq!(
        out.matches("Maximum nesting depth exceeded; deeper content skipped.")
            .count(),
        1,
        "depth guard must fire exactly once"
    );
}

#[test]
fn pathological_wide_inputs() {
    // Very long single lines and very many siblings.
    let long_line = "x".repeat(100_000);
    let _ = parse_rst(&long_line, &opts());
    let adornment = "=".repeat(100_000);
    let _ = parse_rst(&format!("{long_line}\n{adornment}\n"), &opts());
    let many_paras = "para\n\n".repeat(20_000);
    let _ = parse_rst(&many_paras, &opts());
    let many_targets = ".. _t:\n".repeat(5_000);
    let _ = parse_rst(&many_targets, &opts());
}

//! `sphinx.pycode` — the `ModuleAnalyzer`/`DefinitionFinder` surface the
//! `literalinclude` `:pyobject:` filter consumes.
//!
//! TODO(T15): today only the interface and an always-erring stub live here,
//! so the literalinclude filter chain carries its `pyobject` slot in the
//! correct position (first, before start/end/lines/dedent — [INC §3.2])
//! and routes a `:pyobject:` use through the standard except→reporter
//! warning channel with an honest message instead of silently ignoring the
//! option. T15 replaces the stub body with the tokenize-stream
//! `DefinitionFinder` port (`SP/pycode/parser.py:514-588`): a
//! `dict[dotted_name, ('class'|'def', start, end)]` with 1-based inclusive
//! line numbers, decorator lines starting the def, one-liners ending at the
//! header, nested defs skipped and methods recorded as `Class.method`.

use std::collections::BTreeMap;

/// The first member of a `find_tags` entry — sphinx's `'class'` / `'def'`
/// tag strings.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TagKind {
    Class,
    Def,
}

/// `sphinx.errors.PycodeError` — every analyzer failure funnels through
/// `LiteralInclude.run()`'s broad `except` into a reporter warning whose
/// text is this error's `Display`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PycodeError(pub String);

impl std::fmt::Display for PycodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for PycodeError {}

/// `ModuleAnalyzer.find_tags()` over already-decoded source text:
/// `dotted_name -> (kind, start_line, end_line)`, both line numbers
/// 1-based inclusive.
///
/// TODO(T15): stub — always errs, with a message that reaches the user
/// verbatim through the reporter-warning channel. (Divergence note for
/// T15: sphinx reads the FILE with `tokenize.open` and PEP 263 encoding
/// detection; this interface takes the reader's already-decoded text, so
/// non-UTF-8 files with a coding cookie will differ — document there.)
pub fn find_tags(_source: &str) -> Result<BTreeMap<String, (TagKind, u32, u32)>, PycodeError> {
    Err(PycodeError(
        "pyobject is not yet supported by sphinx-ultra".to_string(),
    ))
}

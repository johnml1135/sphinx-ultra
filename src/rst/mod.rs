//! Recursive-descent RST parser with docutils-0.22.4 fidelity (M2 wave 1:
//! block grammar only — the inline parser arrives in wave 2).
//!
//! Fidelity contract: output `pformat()` is byte-identical to
//! `docutils.parsers.rst.Parser` parse-layer output for the construct set in
//! `tests/fixtures/doctree_differential.json`. Transforms (doctitle
//! promotion, target propagation, transition hoisting, message filtering)
//! are explicitly NOT applied here; they arrive as separate components in
//! later waves. Behavior sources: the committed differential fixture and the
//! probe notes in docs/superpowers/plans/2026-08-07-m2-wave1-probes.md.

pub(crate) mod block;
pub(crate) mod digits;
pub mod inline;
pub mod lines;
mod punctuation;

use crate::doctree::Doctree;

#[derive(Debug, Clone)]
pub struct ParseOptions {
    /// What `<document source="...">` prints (docutils `new_document` name).
    pub source_path: String,
    /// Sphinx mode: the Sphinx directive/role registries extend the
    /// docutils-native ones (toctree, xref roles, ...). The binary build
    /// path runs with this on; the docutils differential fixture off.
    pub sphinx: bool,
    /// The docname recorded on pending_xref nodes (sphinx `refdoc`).
    pub docname: String,
    /// Every docname the project discovered (sphinx `env.found_docs`).
    /// The `toctree` directive resolves its entries against this set at
    /// parse time, exactly as Sphinx's `TocTree.parse_content` does.
    ///
    /// `None` means "parsed without an environment" — a standalone parse
    /// (the differential harnesses, `parse_rst` callers) where no document
    /// exists, so every toctree entry resolves to nothing and `entries`/
    /// `includefiles` stay empty. Shared by `Arc` because the build clones
    /// these options once per source file.
    pub found_docs: Option<std::sync::Arc<std::collections::BTreeSet<String>>>,
    /// `exclude_patterns`, which `TocTree.parse_content` consults to tell an
    /// *excluded* toctree target from a *nonexisting* one. Empty for a parse
    /// without an environment, where no entry resolves anyway.
    pub exclude_patterns: Vec<String>,
    /// The object-signature / py-domain configuration the read phase
    /// consumes ([`crate::py::PySigConfig`]): today the `fix_parens` roles'
    /// `add_function_parentheses`, and from the py directives onward the
    /// signature-wrapping and TOC-entry keys too. Defaults to sphinx's own
    /// defaults, so a parse without a project behaves like a default one.
    pub py: crate::py::PySigConfig,
    /// The project source directory (sphinx `env.srcdir`), which the
    /// `include` directive's sphinx-mode path rewrite resolves against
    /// (`sphinx/directives/other.py:413-416` runs `env.relfn2path` on every
    /// include argument before docutils sees it) and which the parse-time
    /// `included`/`dependencies` records are spelled relative to.
    ///
    /// `None` — the default — means "parsed without a project": include
    /// arguments then resolve the docutils way, relative to the directory
    /// of the *containing file*, and no records are made. Standalone
    /// parses (the differential harnesses, `parse_rst` callers) keep their
    /// current behavior.
    pub srcdir: Option<std::path::PathBuf>,
}

impl Default for ParseOptions {
    fn default() -> Self {
        ParseOptions {
            source_path: "<string>".to_string(),
            sphinx: false,
            docname: "index".to_string(),
            found_docs: None,
            exclude_patterns: Vec::new(),
            py: crate::py::PySigConfig::default(),
            srcdir: None,
        }
    }
}

/// Pre-conversion directive tuple mirroring the M1 validation scanner's
/// semantics (whitespace-split args, inline-admonition content routing,
/// raw string options) — the feed for `DirectiveValidationSystem`.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct DirectiveRecord {
    pub name: String,
    pub arguments: Vec<String>,
    pub options: Vec<(String, String)>,
    pub content: String,
    /// 1-based marker line.
    pub line: u32,
}

/// A role occurrence (sphinx mode): validation + nitpicky feed.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct RoleRecord {
    /// Final role-name segment, lowercased (`:py:func:` records `func`),
    /// with the full as-written name kept alongside.
    pub name: String,
    pub full_name: String,
    pub target: String,
    pub display: Option<String>,
    /// 1-based line of the enclosing text block's first line.
    pub line: u32,
}

/// A toctree directive occurrence (sphinx mode).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToctreeRecord {
    pub glob: bool,
    pub entries: Vec<ToctreeEntryRecord>,
    pub line: u32,
    /// Diagnostics `TocTree.parse_content` produced while resolving this
    /// directive's entries. They ride the record (and therefore the
    /// document cache) because the parser has no warning sink, and because
    /// a cache hit that skipped the parse must still reproduce them.
    pub warnings: Vec<crate::env::toctree::ToctreeWarning>,
}

#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ToctreeEntryRecord {
    pub title: Option<String>,
    pub target: String,
    /// 1-based line of the entry itself.
    pub line: u32,
}

/// One `Cmdoption.add_target_and_index` call the parse layer made
/// (`sphinx/domains/std/__init__.py:308-315`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProgramOptionRecord {
    /// Source-table index of the registering signature (an option inside
    /// an included file must attribute to that file). Deliberately not
    /// `#[serde(default)]` — the cache-shape rule
    /// [`RegistryExport::program_options`] explains.
    pub source: u16,
    /// The `.. program::` in scope, `None` outside one.
    pub program: Option<String>,
    /// One `desc_signature['allnames']` spelling (`--file`, `-f`, ...).
    pub name: String,
    /// `signode['ids'][0]` — the *first* id of the signature, which is what
    /// Sphinx registers for every spelling in it.
    pub node_id: String,
}

/// One `PythonDomain.note_object` call the parse layer made
/// (`PyObject.add_target_and_index`, `domains/python/_object.py:415-437`,
/// or `PyModule.run`, `__init__.py:522`) — the fullname → `ObjectEntry`
/// registration the env layer replays, plus the provenance the
/// duplicate-description warning needs.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PyObjectRecord {
    /// Module-qualified full object name (`mymod.C.meth`), or the canonical
    /// name for an `aliased` record.
    pub fullname: String,
    /// The desc's objtype AFTER directive-name aliasing (`py:classmethod`
    /// registers `method`, `py:decorator` registers `function`).
    pub objtype: String,
    pub node_id: String,
    /// `:canonical:` alias registrations carry `true` (`_object.py:427-437`)
    /// — resolve-time disambiguation prefers non-aliased entries.
    pub aliased: bool,
    /// Source-table index of the registering signature. Deliberately not
    /// `#[serde(default)]` (cache-shape rule, see
    /// [`RegistryExport::program_options`]).
    pub source: u16,
    /// 1-based line of the signature node (`location=signode`).
    pub lineno: u32,
}

/// One `PythonDomain.note_module` call (`PyModule.run`,
/// `domains/python/__init__.py:515-521`) — the modname → `ModuleEntry`
/// registration feeding the env layer and, later, the py-modindex.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PyModuleRecord {
    pub name: String,
    /// `module-<name>` (or its `module-<n>` collision serial).
    pub node_id: String,
    /// `:synopsis:` option, `''` when absent. (A bare `:synopsis:` with no
    /// value is Python `None` in sphinx's identity-lambda option spec; it is
    /// recorded as `''` here — probe `module_synopsis_bare`.)
    pub synopsis: String,
    /// `:platform:` option, `''` when absent.
    pub platform: String,
    /// `:deprecated:` flag.
    pub deprecated: bool,
    /// Source-table index of the directive. Deliberately not
    /// `#[serde(default)]` (cache-shape rule, see
    /// [`RegistryExport::program_options`]).
    pub source: u16,
    /// 1-based line of the directive marker.
    pub lineno: u32,
}

/// One `StandardDomain.note_object` call the parse layer made
/// (`GenericObject`/`ConfigurationValue.add_target_and_index`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ObjectRegistration {
    /// Source-table index of the registering signature; the duplicate
    /// warning names this source's path. Deliberately not
    /// `#[serde(default)]` (cache-shape rule, see
    /// [`RegistryExport::program_options`]).
    pub source: u16,
    /// `self.objtype` — `envvar`, `confval`, ... `describe`/`object` never
    /// reach here: the base `add_target_and_index` is a no-op.
    pub objtype: String,
    /// The name `handle_signature` returned, which is what the matching
    /// `:envvar:`/`:confval:` role resolves against.
    pub name: String,
    pub node_id: String,
    /// 1-based line of the signature node (`location=signode`), for the
    /// duplicate-description warning.
    pub line: u32,
}

/// What the parse layer hands the environment besides the doctree itself:
/// state that lives in the parser (the docutils id/name registry, Sphinx's
/// `env.ref_context`) and dies with it, but that env collectors need.
///
/// Named for its original single job — the `document.nameids` snapshot
/// harvested from [`crate::doctree::ids::IdRegistry`] right before it drops,
/// which wave 4's std-domain label harvest reads. Intended to eventually
/// ride the document cache, so it stays serde-serializable and cheap to
/// clone.
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct RegistryExport {
    /// `(name, id, explicit)`, one entry per registered name. `id` is
    /// `None` once a name has been duplicated away.
    pub nameids: Vec<(String, Option<String>, bool)>,
    /// sphinx `env.new_serialno('index')` counter value at the end of the
    /// parse (shared by the index directive and index-entry-emitting roles).
    pub index_serial: u32,
    /// The std-domain registrations the object-description directives made
    /// while running, in document order.
    ///
    /// Sphinx performs these from inside `add_target_and_index`, against
    /// state the finished doctree does not carry: the program an option
    /// belongs to comes from `env.ref_context['std:program']` and is
    /// stamped on no node, and a `:no-typesetting:` description registers
    /// itself and then **replaces its whole `desc` node with a bare
    /// target** — so a doctree walk can neither recover the program nor see
    /// that the object existed. Recording the calls keeps the env layer
    /// exact for both.
    ///
    /// Deliberately *not* `#[serde(default)]`, for the reason
    /// [`crate::document::Document::registry`] gives: a cache entry written
    /// before this field existed must FAIL to decode so the document is
    /// re-parsed. Defaulting it to an empty vector would let a pre-desc
    /// cache decode cleanly, and every `:option:`/`:envvar:`/`:confval:`
    /// in the project would then dangle against an empty registry.
    pub program_options: Vec<ProgramOptionRecord>,
    /// See [`Self::program_options`] — including why this is not
    /// `#[serde(default)]` either.
    pub std_objects: Vec<ObjectRegistration>,
    /// The py-domain object registrations (`PythonDomain.note_object`), in
    /// document order. Same rationale and cache-shape rule as
    /// [`Self::program_options`]: the program state analog here is the
    /// parser's `py:module`/`py:class` ref_context, which no doctree node
    /// carries, and a `:no-typesetting:` py object registers and then
    /// vanishes from the tree.
    pub py_objects: Vec<PyObjectRecord>,
    /// The py-domain module registrations (`PythonDomain.note_module`), in
    /// document order. Not `#[serde(default)]` — see
    /// [`Self::program_options`].
    pub py_modules: Vec<PyModuleRecord>,
    /// Diagnostics the parse raised through Sphinx's *logger* rather than
    /// into the tree, which have nowhere else to go: docutils turns a
    /// directive error into a `system_message` node, but a Sphinx directive
    /// calling `logger.warning` produces no node at all. They ride the
    /// export (and therefore the document cache) for the same reason
    /// [`ToctreeRecord::warnings`] does — a cache hit that skipped the parse
    /// must still reproduce them.
    pub log_warnings: Vec<ParseLogWarning>,
    /// Files the document pulls in at parse time (docutils
    /// `settings.record_dependencies`, harvested by sphinx's
    /// `DependenciesCollector`): one srcdir-relative normalized path per
    /// successfully opened `include` target — non-doc files included,
    /// standard includes excluded (§Scope-2b). The env layer replays these
    /// into `env.dependencies`, which drives `get_outdated_files`. Not
    /// `#[serde(default)]` — see [`Self::program_options`]: a pre-include
    /// cache decoding with an empty list would never re-read the document
    /// when an included file changes.
    pub dependencies: Vec<String>,
    /// The docnames this document textually includes (sphinx
    /// `env.note_included`, recorded for every include argument that maps
    /// to a docname — before the file is even opened, like sphinx). The
    /// env layer replays these into `env.included`, whose only consumer is
    /// the orphan check. Not `#[serde(default)]` — see
    /// [`Self::program_options`].
    pub included: Vec<String>,
}

/// One `logger.warning` a directive raised during the parse.
///
/// Today the only producer is `Cmdoption.handle_signature`'s malformed
/// option description (`domains/std/__init__.py:237-245`), which is logged
/// with no `type`/`subtype` and so renders with no `[category]` suffix.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ParseLogWarning {
    /// Source-table index of the `location=` node's line: the replay
    /// renders this source's path, not the document's. Deliberately not
    /// `#[serde(default)]` (cache-shape rule, see
    /// [`RegistryExport::program_options`]).
    pub source: u16,
    /// The warning text, already formatted exactly as Sphinx renders it.
    pub message: String,
    /// 1-based line of the `location=` node Sphinx passes.
    pub line: u32,
}

/// Everything a parse produces: the doctree plus the flat records the
/// build pipeline consumes without re-walking raw source.
pub struct ParseOutput {
    pub doctree: Doctree,
    pub directive_records: Vec<DirectiveRecord>,
    pub role_records: Vec<RoleRecord>,
    pub toctrees: Vec<ToctreeRecord>,
    pub registry: RegistryExport,
}

/// Parse RST source into a doctree. Total: never panics, never errors —
/// problems become `system_message` nodes, exactly like docutils.
pub fn parse_rst(source: &str, opts: &ParseOptions) -> Doctree {
    parse_rst_full(source, opts).doctree
}

pub fn parse_rst_full(source: &str, opts: &ParseOptions) -> ParseOutput {
    let mut parser = block::BlockParser::new(source, &opts.source_path);
    parser.sphinx = opts.sphinx;
    parser.docname = opts.docname.clone();
    parser.found_docs = opts.found_docs.clone();
    parser.exclude_patterns = opts.exclude_patterns.clone();
    parser.py = opts.py.clone();
    parser.srcdir = opts.srcdir.clone();
    parser.parse_document_full()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// [`RegistryExport`]'s newer fields carry state that cannot be recovered
    /// from a cached doctree, so a cache entry written before they existed
    /// must MISS rather than decode with empty vectors — decoding it would
    /// reuse a doctree still full of unknown-directive errors and leave every
    /// `:option:`/`:envvar:`/`:confval:` in the project dangling. Guards the
    /// `#[serde(default)]` off these fields, which nothing else would catch:
    /// the warm-rebuild tests round-trip the current shape only.
    #[test]
    fn a_registry_written_before_the_std_records_existed_fails_to_decode() {
        let complete = r#"{"nameids":[],"index_serial":0,"program_options":[],
            "std_objects":[],"py_objects":[],"py_modules":[],"log_warnings":[],
            "dependencies":[],"included":[]}"#;
        serde_json::from_str::<RegistryExport>(complete).expect("the current shape decodes");

        for missing in [
            // A wave-4.5 pre-include registry (no dependencies/included
            // stream) must MISS: a defaulted empty list would never
            // re-read the document when an included file changes, and
            // would silently un-suppress the orphan warning.
            r#"{"nameids":[],"index_serial":0,"program_options":[],"std_objects":[],
                "py_objects":[],"py_modules":[],"log_warnings":[],"included":[]}"#,
            r#"{"nameids":[],"index_serial":0,"program_options":[],"std_objects":[],
                "py_objects":[],"py_modules":[],"log_warnings":[],"dependencies":[]}"#,
            r#"{"nameids":[],"index_serial":0,"std_objects":[],"py_objects":[],
                "py_modules":[],"log_warnings":[]}"#,
            r#"{"nameids":[],"index_serial":0,"program_options":[],"py_objects":[],
                "py_modules":[],"log_warnings":[]}"#,
            r#"{"nameids":[],"index_serial":0,"program_options":[],"std_objects":[],
                "py_objects":[],"py_modules":[]}"#,
            // A wave-4 registry (no py record streams at all) must MISS: a
            // defaulted empty vector would leave every py xref in the
            // project dangling on a warm rebuild.
            r#"{"nameids":[],"index_serial":0,"program_options":[],
                "std_objects":[],"log_warnings":[]}"#,
            r#"{"nameids":[],"index_serial":0,"program_options":[],
                "std_objects":[],"py_objects":[],"log_warnings":[]}"#,
            r#"{"nameids":[],"index_serial":0,"program_options":[],
                "std_objects":[],"py_modules":[],"log_warnings":[]}"#,
            r#"{"nameids":[],"index_serial":0}"#,
        ] {
            assert!(
                serde_json::from_str::<RegistryExport>(missing).is_err(),
                "a registry missing a record field must fail to decode: {missing}"
            );
        }
    }

    /// The per-record `source` fields added by the provenance wave follow
    /// the same rule: a cache entry whose records predate them must FAIL to
    /// decode (a defaulted 0 would silently mis-attribute nothing today,
    /// but would decode a stale record stream as current).
    #[test]
    fn records_written_before_the_source_field_existed_fail_to_decode() {
        let with_source = r#"{"nameids":[],"index_serial":0,
            "program_options":[{"source":0,"program":null,"name":"-f","node_id":"a"}],
            "std_objects":[{"source":0,"objtype":"envvar","name":"P","node_id":"b","line":1}],
            "py_objects":[{"fullname":"m.f","objtype":"function","node_id":"m.f",
                "aliased":false,"source":0,"lineno":1}],
            "py_modules":[{"name":"m","node_id":"module-m","synopsis":"","platform":"",
                "deprecated":false,"source":0,"lineno":1}],
            "log_warnings":[{"source":0,"message":"m","line":2}],
            "dependencies":["part.rst"],"included":["part"]}"#;
        serde_json::from_str::<RegistryExport>(with_source).expect("the current shape decodes");

        for stale in [
            r#"{"nameids":[],"index_serial":0,
                "program_options":[{"program":null,"name":"-f","node_id":"a"}],
                "std_objects":[],"py_objects":[],"py_modules":[],"log_warnings":[]}"#,
            r#"{"nameids":[],"index_serial":0,"program_options":[],
                "std_objects":[{"objtype":"envvar","name":"P","node_id":"b","line":1}],
                "py_objects":[],"py_modules":[],"log_warnings":[]}"#,
            r#"{"nameids":[],"index_serial":0,"program_options":[],"std_objects":[],
                "py_objects":[{"fullname":"m.f","objtype":"function","node_id":"m.f",
                    "aliased":false,"lineno":1}],
                "py_modules":[],"log_warnings":[]}"#,
            r#"{"nameids":[],"index_serial":0,"program_options":[],"std_objects":[],
                "py_objects":[],
                "py_modules":[{"name":"m","node_id":"module-m","synopsis":"",
                    "platform":"","deprecated":false,"lineno":1}],
                "log_warnings":[]}"#,
            r#"{"nameids":[],"index_serial":0,"program_options":[],"std_objects":[],
                "py_objects":[],"py_modules":[],
                "log_warnings":[{"message":"m","line":2}]}"#,
        ] {
            assert!(
                serde_json::from_str::<RegistryExport>(stale).is_err(),
                "a record missing its source field must fail to decode: {stale}"
            );
        }
    }
}

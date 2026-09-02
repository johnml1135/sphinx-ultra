//! The block-level recursive-descent parser (M2 wave 1).
//!
//! Model: docutils' `RSTStateMachine` re-expressed as recursive descent over
//! dedented line views. Every nested construct materializes a `Vec<LineRec>`
//! dedented to its own base column (docutils `get_indented` does the same),
//! so all productions parse "at column 0". Each `LineRec` carries `(source,
//! lineno)` provenance and a byte range into the parser-owned source-text
//! table — indices, not borrows, so a directive can splice an included
//! file's lines into the running stream (see [`SpliceRequest`]).
//!
//! Dispatch order matches docutils `Body.initial_transitions`: bullet,
//! enumerator, doctest, line_block, explicit markup, anonymous target,
//! adornment line, text (underline-title / definition list / paragraph).
//! Behavior sources: probe notes (2026-08-07-m2-wave1-probes.md) and the
//! committed differential fixture — never memory.

use crate::doctree::ids::{self, IdRegistry};
use crate::doctree::{kinds, messages, AttrValue, Node, Span};

use std::sync::Arc;

use super::lines::{LineRec, Lines};

const ADORNMENT_CHARS: &str = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~";
const BULLET_CHARS: [char; 6] = ['*', '+', '-', '\u{2022}', '\u{2023}', '\u{2043}'];

/// A pending block-quote segment: accumulated body lines plus an optional
/// (attribution node, marker lineno) that closed it.
type QuoteSegment = (Vec<LineRec>, Option<(Node, u32)>);

/// The parser-owned source table `LineRec::source` and `Span::source`
/// index: per source, its path (what messages stamp and
/// `Doctree::sources` publishes) and its *processed* text (what every
/// `LineRec` byte range slices).
///
/// Texts are `Arc<str>` so a caller that must hold line text across an
/// `&mut self` call can clone the handle ([`SourceTable::arc`]) and slice
/// a local instead of borrowing the parser — which is also what keeps a
/// mid-parse push (an included file arriving) free of self-reference.
#[derive(Debug, Default)]
pub(crate) struct SourceTable {
    paths: Vec<Arc<str>>,
    texts: Vec<Arc<str>>,
}

impl SourceTable {
    /// Append a source; returns its id, or `None` once the u16 id space is
    /// exhausted (a totality guard, like `MAX_NEST_DEPTH` — real documents
    /// never approach 65k sources).
    fn push(&mut self, path: Arc<str>, text: Arc<str>) -> Option<u16> {
        let id = u16::try_from(self.paths.len()).ok()?;
        self.paths.push(path);
        self.texts.push(text);
        Some(id)
    }

    fn path(&self, source: u16) -> &str {
        &self.paths[source as usize]
    }

    /// Shared handle on a path, for passing while `self` is mutably
    /// borrowed elsewhere.
    fn arc_path(&self, source: u16) -> Arc<str> {
        Arc::clone(&self.paths[source as usize])
    }

    fn text(&self, source: u16) -> &str {
        &self.texts[source as usize]
    }

    /// Shared handle on a source's processed text: slice a local clone
    /// instead of borrowing the parser when the text must stay usable
    /// across an `&mut self` call.
    fn arc(&self, source: u16) -> Arc<str> {
        Arc::clone(&self.texts[source as usize])
    }

    /// The current view of `rec` — valid until the next dedent/re-wrap of
    /// the record, unaffected by table growth.
    fn line_text(&self, rec: LineRec) -> &str {
        rec.slice(self.text(rec.source))
    }

    fn len(&self) -> usize {
        self.paths.len()
    }

    fn into_paths(self) -> Vec<String> {
        self.paths.iter().map(|p| p.to_string()).collect()
    }
}

/// What a directive hands back besides the nodes it pushed: lines of a new
/// source to insert into the running line stream right after the directive
/// (T12's `include` is the intended producer; only a test directive
/// returns it this wave). Plain data — a directive builds one from its
/// input alone, with no access to parser internals.
#[derive(Debug)]
pub(crate) struct SpliceRequest {
    /// Raw lines of the new source, exactly as read (they are processed —
    /// tab expansion, trailing-whitespace strip — on insertion).
    pub lines: Vec<String>,
    /// The path messages and spans attribute the lines to.
    pub source_path: String,
    /// First line number of the spliced lines; `None` numbers from 1 (an
    /// included file), `Some(n)` keeps a caller-chosen base.
    pub base_lineno_override: Option<u32>,
}

/// What running a directive produced beyond its nodes.
// Only the test directive produces a splice until T12's include lands, so
// outside test builds the channel is currently unconsumed.
#[allow(dead_code)]
enum DirectiveOutcome {
    Done,
    Splice(SpliceRequest),
}

/// A glossary comment line: unindented and opening with `.. `
/// (`domains/std/__init__.py:452`, `line.startswith('.. ')` — the trailing
/// space is part of the test, so a bare `..` is still a term).
fn is_glossary_comment(line: &LineRec, line_text: &str) -> bool {
    line.indent() == 0 && line_text.starts_with(".. ")
}

/// Byte offset of the character `n_chars` into `text` (its length when
/// the text is shorter) — for re-wrapping a `LineRec` view past a marker
/// (char-aware: unicode bullets are multi-byte).
fn rest_after_offset(text: &str, n_chars: usize) -> usize {
    match text.char_indices().nth(n_chars) {
        Some((i, _)) => i,
        None => text.len(),
    }
}

fn adornment_char(text: &str) -> Option<char> {
    let mut chars = text.chars();
    let first = chars.next()?;
    if ADORNMENT_CHARS.contains(first) && chars.all(|c| c == first) {
        Some(first)
    } else {
        None
    }
}

fn char_len(text: &str) -> usize {
    text.chars().count()
}

/// docutils `column_width`: east-asian wide/fullwidth chars count 2.
fn column_width(text: &str) -> usize {
    unicode_width::UnicodeWidthStr::width(text)
}

/// Definition-list term split on the docutils classifier delimiter
/// `' +: +'` (one-or-more spaces, colon, one-or-more spaces).
fn split_classifiers(term: &str) -> Vec<String> {
    lazy_static::lazy_static! {
        static ref CLASSIFIER_RE: regex::Regex = regex::Regex::new(" +: +").unwrap();
    }
    CLASSIFIER_RE.split(term).map(str::to_string).collect()
}

struct SectionStart {
    title: String,
    style: (char, bool),
    /// Raw title + underline lines, for error literals.
    raw_lines: String,
    /// Extra messages inserted right after `<title>` (short-underline
    /// warning); the duplicate-name INFO is added by the caller.
    messages: Vec<Node>,
    title_lineno: u32,
    underline_lineno: u32,
    span: Span,
}

/// Nested-container recursion cap. Real documents nest ~10 deep; docutils
/// itself dies with RecursionError near Python's limit (~1000). We stay
/// total: content beyond this depth is dropped with an ERROR message.
const MAX_NEST_DEPTH: usize = 200;

pub(crate) struct BlockParser {
    top: Vec<LineRec>,
    /// Source table (paths + processed texts). Entry 0 is the document;
    /// sub-parses over lifted text and spliced sources push.
    sources: SourceTable,
    pub(crate) registry: IdRegistry,
    styles: Vec<(char, bool)>,
    depth: usize,
    /// +1 inside table-cell nested parses: docutils' state-machine-derived
    /// line numbers (the unindent/unexpected-indentation family) run one
    /// high there (probe-verified); content-anchored messages stay absolute.
    line_bias: u32,
    /// Innermost container node kind during nested content parses (docutils
    /// `state_machine.node`); None at document/section level. Directives
    /// like topic/sidebar validate their direct parent against this.
    nested_node_kind: Option<&'static str>,
    /// Sphinx mode (see [`super::ParseOptions::sphinx`]).
    pub(crate) sphinx: bool,
    /// The docname stamped on pending_xref nodes (sphinx `refdoc`).
    pub(crate) docname: String,
    /// Every discovered docname (sphinx `env.found_docs`), for toctree entry
    /// resolution; `None` outside a build (see
    /// [`super::ParseOptions::found_docs`]).
    pub(crate) found_docs: Option<std::sync::Arc<std::collections::BTreeSet<String>>>,
    /// `exclude_patterns` (see [`super::ParseOptions::exclude_patterns`]).
    pub(crate) exclude_patterns: Vec<String>,
    /// The py-domain configuration the read phase consumes (see
    /// [`super::ParseOptions::py`]).
    pub(crate) py: crate::py::PySigConfig,
    /// `.. highlight::` state consumed by later code-blocks in the same
    /// document (sphinx env.temp_data\['highlight_language'\]).
    highlight_language: Option<String>,
    /// `.. program::` state consumed by later `.. option::` directives in
    /// the same document (sphinx `env.ref_context['std:program']`).
    program: Option<String>,
    /// sphinx `env.ref_context['py:module']` — set by `py:module`/
    /// `py:currentmodule` and pushed/popped by an object's `:module:`
    /// option (`domains/python/_object.py:477-480`, `498-503`).
    py_module: Option<String>,
    /// sphinx `env.ref_context['py:modules']` — the `:module:` option's
    /// push/pop stack (entries may be `None`: the module in scope when the
    /// option pushed).
    py_modules: Vec<Option<String>>,
    /// sphinx `env.ref_context['py:class']` — the innermost class scope
    /// (`_object.py:449-503`).
    py_class: Option<String>,
    /// sphinx `env.ref_context['py:classes']` — the `allow_nesting`
    /// (class/exception) nesting stack.
    py_classes: Vec<String>,
    /// Sphinx-mode class/rst-class pending classes (the ClassAttribute
    /// transform effect applied inline).
    pending_classes: Option<Vec<String>>,
    /// Per-document equation counter (math domain numbering).
    equation_serial: u32,
    /// Validation-feed records collected during the parse.
    directive_records: Vec<super::DirectiveRecord>,
    role_records: Vec<super::RoleRecord>,
    toctree_records: Vec<super::ToctreeRecord>,
    /// The std-domain registrations the object-description directives made
    /// while running, in document order (see
    /// [`super::RegistryExport::program_options`] for why the finished
    /// doctree cannot carry them).
    program_option_records: Vec<super::ProgramOptionRecord>,
    std_object_records: Vec<super::ObjectRegistration>,
    /// The py-domain registrations (`PythonDomain.note_object` /
    /// `note_module` calls), in document order — see
    /// [`super::RegistryExport::py_objects`].
    py_object_records: Vec<super::PyObjectRecord>,
    py_module_records: Vec<super::PyModuleRecord>,
    /// `logger.warning` diagnostics raised while running directives (see
    /// [`super::ParseLogWarning`]).
    log_warnings: Vec<super::ParseLogWarning>,
    /// Set while running a substitution-embedded directive (docutils
    /// SubstitutionDef state): replace/unicode/date require it, image
    /// flips its align validation, unicode's trim flags land here.
    substitution_ctx: Option<SubstCtx>,
    /// Substitution names seen (whitespace-normalized, case-preserving) —
    /// docutils document.substitution_defs.
    substitution_names_seen: Vec<String>,
    /// Names defined more than once: earlier nodes get names -> dupnames
    /// in a post-parse walk (docutils mutates the old node in place).
    substitution_dupnames: Vec<String>,
    /// A [`SpliceRequest`] a directive just returned, waiting for the
    /// enclosing block-parse loop to insert it at its cursor.
    pending_splice: Option<SpliceRequest>,
}

#[derive(Debug, Default)]
struct SubstCtx {
    ltrim: bool,
    rtrim: bool,
}

impl BlockParser {
    /// Parser over one document: the single-source form (entry 0 = the
    /// document, linenos `1..=n`).
    pub(crate) fn new(source: &str, source_path: &str) -> Self {
        let (text, top) = Lines::new(source).into_parts();
        let mut sources = SourceTable::default();
        sources
            .push(Arc::from(source_path), Arc::from(text))
            .expect("a fresh table accepts entry 0");
        BlockParser::from_parts(top, sources)
    }

    fn from_parts(top: Vec<LineRec>, sources: SourceTable) -> Self {
        BlockParser {
            top,
            sources,
            registry: IdRegistry::new(),
            styles: Vec::new(),
            depth: 0,
            line_bias: 0,
            nested_node_kind: None,
            sphinx: false,
            docname: "index".to_string(),
            found_docs: None,
            exclude_patterns: Vec::new(),
            py: crate::py::PySigConfig::default(),
            highlight_language: None,
            program: None,
            py_module: None,
            py_modules: Vec::new(),
            py_class: None,
            py_classes: Vec::new(),
            pending_classes: None,
            equation_serial: 0,
            directive_records: Vec::new(),
            role_records: Vec::new(),
            toctree_records: Vec::new(),
            program_option_records: Vec::new(),
            std_object_records: Vec::new(),
            py_object_records: Vec::new(),
            py_module_records: Vec::new(),
            log_warnings: Vec::new(),
            substitution_ctx: None,
            substitution_names_seen: Vec::new(),
            substitution_dupnames: Vec::new(),
            pending_splice: None,
        }
    }

    /// parse_document plus the flat build-pipeline records.
    pub(crate) fn parse_document_full(mut self) -> super::ParseOutput {
        let root = self.parse_document_impl();
        // Harvest the id/name registry before it drops with `self`: wave 4's
        // std-domain label harvest needs name -> (id, explicit) data that
        // otherwise dies with the BlockParser.
        let registry = super::RegistryExport {
            nameids: self.registry.nameids_snapshot(),
            index_serial: self.registry.index_serial(),
            program_options: std::mem::take(&mut self.program_option_records),
            std_objects: std::mem::take(&mut self.std_object_records),
            py_objects: std::mem::take(&mut self.py_object_records),
            py_modules: std::mem::take(&mut self.py_module_records),
            log_warnings: std::mem::take(&mut self.log_warnings),
        };
        super::ParseOutput {
            doctree: crate::doctree::Doctree {
                root,
                sources: self.sources.into_paths(),
            },
            directive_records: std::mem::take(&mut self.directive_records),
            role_records: std::mem::take(&mut self.role_records),
            toctrees: std::mem::take(&mut self.toctree_records),
            registry,
        }
    }

    /// Validation-feed record with the M1 validation-scanner's semantics
    /// (spec-INdependent, so registered and unknown directives record the
    /// same way): whitespace-split args, marker-line text routed to
    /// content for the admonition name set, raw string options.
    fn capture_directive_record(
        &mut self,
        name: &str,
        first_line: &LineRec,
        block: &[LineRec],
        lineno: u32,
    ) {
        const INLINE_ADMONITIONS: &[&str] = &[
            "note",
            "warning",
            "tip",
            "hint",
            "important",
            "caution",
            "danger",
            "error",
            "attention",
            "seealso",
        ];
        let mut options: Vec<(String, String)> = Vec::new();
        let mut content_lines: Vec<String> = Vec::new();
        let marker_text = self.sources.line_text(*first_line).trim();
        let lower = name.to_lowercase();
        let is_admonition = INLINE_ADMONITIONS.contains(&lower.as_str());
        if is_admonition && !marker_text.is_empty() {
            content_lines.push(marker_text.to_string());
        }
        // Leading option lines; everything after is content.
        let mut in_options = true;
        for l in block {
            if l.is_blank() {
                if !in_options {
                    content_lines.push(String::new());
                }
                continue;
            }
            let text = self.sources.line_text(*l);
            if in_options {
                if let Some((oname, body_start)) = field_marker(text.trim_start()) {
                    let base = text.len() - text.trim_start().len();
                    let val = text[base + body_start..].trim().to_string();
                    options.push((oname, val));
                    continue;
                }
                in_options = false;
            }
            content_lines.push(text.to_string());
        }
        while content_lines.last().map(|l| l.is_empty()).unwrap_or(false) {
            content_lines.pop();
        }
        let arguments: Vec<String> = if is_admonition {
            Vec::new()
        } else {
            marker_text.split_whitespace().map(str::to_string).collect()
        };
        self.directive_records.push(super::DirectiveRecord {
            name: name.to_string(),
            arguments,
            options,
            content: content_lines.join("\n"),
            line: lineno,
        });
    }

    /// Inline parse through the parser's own registry/mode; collects role
    /// records emitted by the inliner. Messages the inliner raises stamp
    /// the span's own source path.
    fn inline(&mut self, text: &str, span: Span, lineno: u32) -> super::inline::InlineResult {
        let source_path = self.sources.arc_path(span.source);
        let mut result = super::inline::parse_inline_ext(
            text,
            span,
            lineno,
            &mut self.registry,
            &source_path,
            self.sphinx,
            &self.docname,
            self.program.as_deref(),
            self.py_module.as_deref(),
            self.py_class.as_deref(),
            &self.py,
        );
        self.role_records.append(&mut result.roles);
        result
    }

    /// parse_elements with the containing node kind recorded (docutils
    /// nested_parse: `state_machine.node` = the container element).
    fn parse_nested(&mut self, lines: &[LineRec], kind: &'static str) -> Vec<Node> {
        let saved = self.nested_node_kind.replace(kind);
        let nodes = self.parse_elements(lines);
        self.nested_node_kind = saved;
        nodes
    }

    /// Nested parse over OWNED text (csv-table cells and, later,
    /// rst_prolog): a sub-parser over a new source-table entry, sharing
    /// this parser's table and id registry, with linenos starting at
    /// `first_lineno` so absolute line numbers keep working. The entry's
    /// path copies `attribute_to`'s — the source the text was lifted from
    /// — so messages and spans keep attributing to it.
    fn parse_detached(
        &mut self,
        text: &str,
        first_lineno: u32,
        attribute_to: u16,
        kind: &'static str,
    ) -> Vec<Node> {
        let path = self.sources.arc_path(attribute_to);
        let Some((_id, top)) = self.push_source(path, text, first_lineno) else {
            // Source-id space exhausted (totality guard): drop the nested
            // content rather than mis-attribute it.
            return Vec::new();
        };
        let mut sub = BlockParser::from_parts(top, std::mem::take(&mut self.sources));
        sub.registry = std::mem::replace(&mut self.registry, IdRegistry::new());
        sub.nested_node_kind = Some(kind);
        sub.line_bias = self.line_bias;
        sub.depth = self.depth;
        // Mode + records must flow through the detached parse (review
        // finding: csv cells previously parsed in docutils mode and their
        // directive/role records were dropped).
        sub.sphinx = self.sphinx;
        sub.docname = self.docname.clone();
        sub.found_docs = self.found_docs.clone();
        sub.exclude_patterns = self.exclude_patterns.clone();
        sub.py = self.py.clone();
        sub.highlight_language = self.highlight_language.clone();
        sub.program = self.program.clone();
        // The py ref_context flows in like `program` (state changes made
        // inside a detached parse stay local, matching the wave-4
        // convention); the record streams flow back out below.
        sub.py_module = self.py_module.clone();
        sub.py_modules = self.py_modules.clone();
        sub.py_class = self.py_class.clone();
        sub.py_classes = self.py_classes.clone();
        let top = std::mem::take(&mut sub.top);
        let nodes = sub.parse_elements(&top);
        self.sources = sub.sources;
        self.registry = sub.registry;
        self.directive_records.append(&mut sub.directive_records);
        self.role_records.append(&mut sub.role_records);
        self.toctree_records.append(&mut sub.toctree_records);
        self.program_option_records
            .append(&mut sub.program_option_records);
        self.std_object_records.append(&mut sub.std_object_records);
        self.py_object_records.append(&mut sub.py_object_records);
        self.py_module_records.append(&mut sub.py_module_records);
        self.log_warnings.append(&mut sub.log_warnings);
        nodes
    }

    /// Process `text` into a new source-table entry named `path`; returns
    /// the entry's id and record stream (`None` when the id space is
    /// exhausted).
    fn push_source(
        &mut self,
        path: Arc<str>,
        text: &str,
        first_lineno: u32,
    ) -> Option<(u16, Vec<LineRec>)> {
        // The id is only known after the push, but the recs need it up
        // front — take it from the table length the push will use.
        let id = u16::try_from(self.sources.len()).ok()?;
        let (processed, recs) = Lines::for_source(text, id, first_lineno).into_parts();
        self.sources.push(path, Arc::from(processed))?;
        Some((id, recs))
    }

    /// Insert a [`SpliceRequest`]'s lines into `lines` at `at` (the
    /// block-parse loop's cursor, right past the directive that returned
    /// it): the request's text becomes a new source-table entry and its
    /// records join the running stream. On id-space exhaustion the request
    /// is dropped (same totality guard as [`Self::push_source`]).
    fn apply_splice(&mut self, lines: &mut Vec<LineRec>, at: usize, request: SpliceRequest) {
        let SpliceRequest {
            lines: raw_lines,
            source_path,
            base_lineno_override,
        } = request;
        let text = raw_lines.join("\n");
        let first_lineno = base_lineno_override.unwrap_or(1);
        let Some((_id, recs)) = self.push_source(Arc::from(source_path), &text, first_lineno)
        else {
            return;
        };
        let at = at.min(lines.len());
        lines.splice(at..at, recs);
    }

    /// Re-wrap `rec` to the `from..to` byte sub-range of its current view
    /// (a marker consumed, a table-cell column carved): the leading-space
    /// cache is recomputed for the shrunk view.
    fn rewrap_range(&self, rec: LineRec, from: usize, to: usize) -> LineRec {
        let start = rec.start + from as u32;
        let end = rec.start + to as u32;
        debug_assert!(from <= to && end <= rec.end);
        LineRec::new(
            rec.source,
            rec.lineno,
            start,
            end,
            &self.sources.text(rec.source)[start as usize..end as usize],
        )
    }

    /// Re-wrap `rec` past the first `byte_off` bytes of its current view.
    fn rewrap_from(&self, rec: LineRec, byte_off: usize) -> LineRec {
        self.rewrap_range(rec, byte_off, (rec.end - rec.start) as usize)
    }

    /// A zero-width (blank) view at the start of `rec`'s line, keeping its
    /// provenance — the shape table cells use for their blank rows.
    fn blank_at(&self, rec: LineRec) -> LineRec {
        self.rewrap_range(rec, 0, 0)
    }

    /// The lines' current views joined with `\n`.
    fn join_lines(&self, lines: &[LineRec]) -> String {
        let mut joined = String::new();
        for (i, l) in lines.iter().enumerate() {
            if i > 0 {
                joined.push('\n');
            }
            joined.push_str(self.sources.line_text(*l));
        }
        joined
    }

    fn span_of(&self, lines: &[LineRec], first: usize, last: usize) -> Span {
        let first_rec = lines.get(first);
        let start = first_rec.map(|l| l.start).unwrap_or(0);
        let end = lines
            .get(last.min(lines.len().saturating_sub(1)))
            .map(|l| l.end)
            .unwrap_or(start);
        Span {
            source: first_rec.map(|l| l.source).unwrap_or(0),
            line: first_rec.map(|l| l.lineno).unwrap_or(0),
            start,
            end,
        }
    }

    /// A `system_message` anchored at `lineno` of `source` — the message
    /// stamps that source's table path.
    fn msg(&self, level: u8, text: &str, source: u16, lineno: u32) -> Node {
        messages::system_message(level, text, lineno, self.sources.path(source))
    }

    /// For state-machine-position-derived messages (see `line_bias`).
    fn msg_sm(&self, level: u8, text: &str, source: u16, lineno: u32) -> Node {
        messages::system_message(
            level,
            text,
            lineno + self.line_bias,
            self.sources.path(source),
        )
    }

    /// Probe-verified: an explicit-markup element (comment/target) followed
    /// by an ADJACENT non-blank column-0 line that is not itself explicit
    /// markup warns. Consecutive `..`/`__ ` items chain without warning.
    fn warn_explicit_markup_end(&self, lines: &[LineRec], pos: usize, out: &mut Vec<Node>) {
        if let Some(l) = lines.get(pos) {
            let text = self.sources.line_text(*l);
            let explicit_ish = text == ".." || text.starts_with(".. ") || text.starts_with("__ ");
            if !l.is_blank() && l.indent() == 0 && !explicit_ish {
                out.push(self.msg(
                    messages::WARNING,
                    "Explicit markup ends without a blank line; unexpected unindent.",
                    l.source,
                    l.lineno,
                ));
            }
        }
    }

    // ------------------------------------------------------------------
    // document level (the only level where titles match)
    // ------------------------------------------------------------------

    fn parse_document_impl(&mut self) -> Node {
        let mut root = Node::elem(
            kinds::DOCUMENT,
            Span {
                source: 0,
                line: 1,
                start: 0,
                end: self.sources.text(0).len() as u32,
            },
        );
        root.set("source", AttrValue::Str(self.sources.path(0).to_string()));

        // Open sections, deepest last; nodes attach on close.
        let mut stack: Vec<Node> = Vec::new();
        let mut lines = std::mem::take(&mut self.top);
        let mut pos = 0usize;
        while pos < lines.len() {
            if lines[pos].is_blank() {
                pos += 1;
                continue;
            }
            let mut out = Vec::new();
            let section = self.parse_element(&lines, &mut pos, true, &mut out);
            self.apply_pending_classes(&mut out, 0);
            for node in out {
                Self::container(&mut root, &mut stack).children.push(node);
            }
            if let Some(start) = section {
                self.open_section(start, &mut root, &mut stack);
            }
            // A directive just asked for new lines at the cursor (T12's
            // include; only a test directive this wave).
            if let Some(request) = self.pending_splice.take() {
                self.apply_splice(&mut lines, pos, request);
            }
        }
        while !stack.is_empty() {
            Self::close_section(&mut root, &mut stack);
        }

        let fixups = self.registry.take_fixups();
        ids::apply_dupname_fixups(&mut root, &fixups);
        // Duplicate substitution definitions: docutils dupname()s the OLD
        // node in place; we re-walk since the tree is owned (all but the
        // LAST same-name definition lose the name).
        for name in std::mem::take(&mut self.substitution_dupnames) {
            let total = count_subst_defs(&root, &name);
            if total > 1 {
                let mut remaining = total - 1;
                dupname_subst_defs(&mut root, &name, &mut remaining);
            }
        }
        root
    }

    fn container<'r>(root: &'r mut Node, stack: &'r mut [Node]) -> &'r mut Node {
        match stack.last_mut() {
            Some(top) => top,
            None => root,
        }
    }

    fn close_section(root: &mut Node, stack: &mut Vec<Node>) {
        if let Some(mut done) = stack.pop() {
            if let Some(last) = done.children.last() {
                done.span.end = done.span.end.max(last.span.end);
            }
            Self::container(root, stack).children.push(done);
        }
    }

    fn open_section(&mut self, start: SectionStart, root: &mut Node, stack: &mut Vec<Node>) {
        let known = self.styles.iter().position(|s| *s == start.style);
        let level = match known {
            Some(i) => i + 1,
            None => self.styles.len() + 1,
        };
        if level > stack.len() + 1 {
            // Skipped level: ERROR, section dropped, content continues here.
            let text = format!(
                "Inconsistent title style: skip from level {} to {}.",
                stack.len(),
                level
            );
            let mut msg = self.msg(
                messages::ERROR,
                &text,
                start.span.source,
                start.title_lineno,
            );
            msg = messages::with_literal(msg, &start.raw_lines);
            let established: Vec<String> = self
                .styles
                .iter()
                .map(|(c, over)| {
                    if *over {
                        format!("{c}/{c}")
                    } else {
                        c.to_string()
                    }
                })
                .collect();
            msg = messages::with_paragraph(
                msg,
                &format!("Established title styles: {}", established.join(" ")),
            );
            Self::container(root, stack).children.push(msg);
            return;
        }
        if known.is_none() {
            self.styles.push(start.style);
        }
        while stack.len() >= level {
            Self::close_section(root, stack);
        }

        let inline = self.inline(&start.title, start.span, start.title_lineno);
        let mut title = Node::elem(kinds::TITLE, start.span);
        title.children = inline.nodes;
        // Section name from the title's TEXT content (markup stripped).
        // The section's stamped line is one past its span's first line —
        // docutils creates the section only once the state machine has
        // consumed the underline, so it reports the underline line for the
        // plain form and the title line for the overline form.
        let mut section_span = start.span;
        section_span.line += 1;
        let mut section = Node::elem(kinds::SECTION, section_span);
        section
            .attrs
            .names
            .push(ids::fully_normalize_name(&title.astext()));
        let source_path = self.sources.arc_path(start.span.source);
        let dup_info =
            self.registry
                .set_id_implicit(&mut section, start.underline_lineno, &source_path);
        section.children.push(title);
        for m in start.messages {
            section.children.push(m);
        }
        for m in inline.messages {
            section.children.push(m);
        }
        if let Some(info) = dup_info {
            section.children.push(info);
        }
        stack.push(section);
    }

    // ------------------------------------------------------------------
    // element dispatch
    // ------------------------------------------------------------------

    fn parse_elements(&mut self, lines: &[LineRec]) -> Vec<Node> {
        if self.depth >= MAX_NEST_DEPTH {
            // sphinx-ultra-specific totality guard (docutils crashes here).
            let anchor = lines
                .first()
                .map(|l| (l.source, l.lineno))
                .unwrap_or((0, 1));
            return vec![self.msg(
                messages::ERROR,
                "Maximum nesting depth exceeded; deeper content skipped.",
                anchor.0,
                anchor.1,
            )];
        }
        self.depth += 1;
        let out = self.parse_elements_inner(lines);
        self.depth -= 1;
        out
    }

    fn parse_elements_inner(&mut self, lines: &[LineRec]) -> Vec<Node> {
        let mut out = Vec::new();
        let mut pos = 0usize;
        // Once a directive splices new lines in, the loop continues over an
        // owned copy of the stream; until then the borrowed slice serves
        // (records are Copy, so the one-time copy is cheap and rare).
        let mut owned: Option<Vec<LineRec>> = None;
        loop {
            let cur: &[LineRec] = owned.as_deref().unwrap_or(lines);
            if pos >= cur.len() {
                break;
            }
            if cur[pos].is_blank() {
                pos += 1;
                continue;
            }
            let before = out.len();
            let section = self.parse_element(cur, &mut pos, false, &mut out);
            debug_assert!(section.is_none(), "titles never match in nested contexts");
            self.apply_pending_classes(&mut out, before);
            if let Some(request) = self.pending_splice.take() {
                let mut stream = owned.take().unwrap_or_else(|| lines.to_vec());
                self.apply_splice(&mut stream, pos, request);
                owned = Some(stream);
            }
        }
        out
    }

    /// Sphinx mode runs the ClassAttribute transform effect inline: a
    /// class/rst-class directive without content stamps the next
    /// non-invisible sibling element (the pending node itself vanishes).
    fn apply_pending_classes(&mut self, out: &mut [Node], from: usize) {
        if self.pending_classes.is_none() {
            return;
        }
        for node in out[from..].iter_mut() {
            if matches!(
                node.kind,
                kinds::COMMENT | kinds::TARGET | kinds::SYSTEM_MESSAGE | "substitution_definition"
            ) {
                continue;
            }
            if let Some(classes) = self.pending_classes.take() {
                node.attrs.classes.extend(classes);
            }
            break;
        }
    }

    /// Parse one element starting at `lines[*pos]` (non-blank). Returns a
    /// pending section start when `match_titles` and a title was found.
    fn parse_element(
        &mut self,
        lines: &[LineRec],
        pos: &mut usize,
        match_titles: bool,
        out: &mut Vec<Node>,
    ) -> Option<SectionStart> {
        let line = lines[*pos];
        if line.indent() > 0 {
            self.parse_block_quote(lines, pos, out);
            return None;
        }
        // Local handle: the text must stay readable across the `&mut self`
        // dispatch calls below.
        let src = self.sources.arc(line.source);
        let text = line.slice(&src);

        if let Some(bullet) = Self::bullet_marker(text) {
            self.parse_bullet_list(lines, pos, bullet, out);
            return None;
        }
        if let Some(e) = parse_enumerator(text) {
            if self.try_enumerated_list(lines, pos, &e, out) {
                return None;
            }
            // invalid list start: fall through to the text path
        }
        if field_marker(text).is_some() {
            self.parse_field_list(lines, pos, out);
            return None;
        }
        if option_group_marker(text).is_some() && self.option_item_viable(lines, *pos) {
            self.parse_option_list(lines, pos, out);
            return None;
        }
        if text.starts_with(">>> ") || text == ">>>" {
            self.parse_doctest(lines, pos, out);
            return None;
        }
        if text == "|" || text.starts_with("| ") {
            self.parse_line_block(lines, pos, out);
            return None;
        }
        if is_grid_table_top(text) {
            self.parse_grid_table(lines, pos, out);
            return None;
        }
        if is_simple_table_top(text) {
            self.parse_simple_table(lines, pos, out);
            return None;
        }
        if text == ".." || text.starts_with(".. ") {
            self.parse_explicit(lines, pos, out);
            return None;
        }
        if let Some(rest) = text.strip_prefix("__ ") {
            self.parse_anonymous_shortcut(lines, pos, rest, out);
            return None;
        }
        if text == "__" {
            // Bare `__`: anonymous internal target (fixture-verified).
            self.parse_anonymous_shortcut(lines, pos, "", out);
            return None;
        }
        if let Some(c) = adornment_char(text) {
            return self.handle_adornment(lines, pos, c, match_titles, out);
        }
        self.handle_text(lines, pos, match_titles, out)
    }

    fn bullet_marker(text: &str) -> Option<char> {
        let mut chars = text.chars();
        let first = chars.next()?;
        if !BULLET_CHARS.contains(&first) {
            return None;
        }
        match chars.next() {
            None => Some(first),
            Some(' ') => Some(first),
            Some(_) => None,
        }
    }

    // ------------------------------------------------------------------
    // adornment lines ("line" state)
    // ------------------------------------------------------------------

    fn handle_adornment(
        &mut self,
        lines: &[LineRec],
        pos: &mut usize,
        ch: char,
        match_titles: bool,
        out: &mut Vec<Node>,
    ) -> Option<SectionStart> {
        let line = lines[*pos];
        let len = char_len(self.sources.line_text(line));
        let next = lines.get(*pos + 1).copied();
        let next_is_text = next.map(|n| !n.is_blank()).unwrap_or(false);

        if !match_titles {
            if len >= 4 {
                let msg = messages::with_literal(
                    self.msg(
                        messages::ERROR,
                        "Unexpected section title or transition.",
                        line.source,
                        line.lineno,
                    ),
                    self.sources.line_text(line),
                );
                out.push(msg);
                *pos += 1;
            } else {
                // Fixture-verified: short adornments in nested contexts get
                // an INFO, then reprocess through the text state.
                out.push(self.msg(
                    messages::INFO,
                    "Unexpected possible title overline or transition.\nTreating it as ordinary text because it's so short.",
                    line.source,
                    line.lineno,
                ));
                return self.handle_text(lines, pos, match_titles, out);
            }
            return None;
        }

        if !next_is_text {
            if len >= 4 {
                out.push(Node::elem(
                    kinds::TRANSITION,
                    self.span_of(lines, *pos, *pos),
                ));
                *pos += 1;
            } else {
                self.parse_paragraph_like(lines, pos, out);
            }
            return None;
        }

        // Overline candidacy: adornment, then a second line.
        let title_line = next.unwrap();
        if len < 4 {
            // Short overline: INFO, then reprocess through the text state
            // ("--\n--" becomes a section titled "--"; "---\n    x" becomes
            // a definition list).
            out.push(self.msg(
                messages::INFO,
                "Possible incomplete section title.\nTreating the overline as ordinary text because it's so short.",
                line.source,
                line.lineno,
            ));
            return self.handle_text(lines, pos, match_titles, out);
        }
        if adornment_char(self.sources.line_text(title_line)).is_some() {
            let literal = format!(
                "{}\n{}",
                self.sources.line_text(line),
                self.sources.line_text(title_line)
            );
            let msg = messages::with_literal(
                self.msg(
                    messages::ERROR,
                    "Invalid section title or transition marker.",
                    line.source,
                    line.lineno,
                ),
                &literal,
            );
            out.push(msg);
            *pos += 2;
            return None;
        }
        let under = lines.get(*pos + 2).copied();
        // Fixture-verified message split: at EOF the title is "incomplete";
        // with a blank or text third line the underline is "missing".
        let missing_underline = match under {
            None => Some(("Incomplete section title.", 2usize, false)),
            Some(u) if u.is_blank() => Some((
                "Missing matching underline for section title overline.",
                2,
                false,
            )),
            Some(u) if adornment_char(self.sources.line_text(u)).is_none() => Some((
                "Missing matching underline for section title overline.",
                3,
                true,
            )),
            _ => None,
        };
        if let Some((text, consume, third_in_literal)) = missing_underline {
            let literal = if third_in_literal {
                format!(
                    "{}\n{}\n{}",
                    self.sources.line_text(line),
                    self.sources.line_text(title_line),
                    self.sources.line_text(lines[*pos + 2])
                )
            } else {
                format!(
                    "{}\n{}",
                    self.sources.line_text(line),
                    self.sources.line_text(title_line)
                )
            };
            let msg = messages::with_literal(
                self.msg(messages::ERROR, text, line.source, line.lineno),
                &literal,
            );
            out.push(msg);
            *pos += consume;
            return None;
        }
        let under = under.unwrap();
        let under_text = self.sources.line_text(under);
        if adornment_char(under_text) != Some(ch) || char_len(under_text) != len {
            // Different char or different length: both are a mismatch.
            let literal = format!(
                "{}\n{}\n{}",
                self.sources.line_text(line),
                self.sources.line_text(title_line),
                self.sources.line_text(under)
            );
            let msg = messages::with_literal(
                self.msg(
                    messages::ERROR,
                    "Title overline & underline mismatch.",
                    line.source,
                    line.lineno,
                ),
                &literal,
            );
            out.push(msg);
            *pos += 3;
            return None;
        }
        // Title column width (leading spaces included) wider than the
        // adornment: section is still created, WARNING inside.
        let mut msgs = Vec::new();
        let raw = format!(
            "{}\n{}\n{}",
            self.sources.line_text(line),
            self.sources.line_text(title_line),
            self.sources.line_text(under)
        );
        if column_width(self.sources.line_text(title_line)) > len {
            msgs.push(messages::with_literal(
                self.msg(
                    messages::WARNING,
                    "Title overline too short.",
                    line.source,
                    line.lineno,
                ),
                &raw,
            ));
        }
        let span = self.span_of(lines, *pos, *pos + 2);
        let title_lineno = title_line.lineno;
        let underline_lineno = under.lineno;
        let title_text = self.sources.line_text(title_line).trim().to_string();
        *pos += 3;
        Some(SectionStart {
            title: title_text,
            style: (ch, true),
            raw_lines: raw,
            messages: msgs,
            title_lineno,
            underline_lineno,
            span,
        })
    }

    // ------------------------------------------------------------------
    // text state: underline titles, definition lists, paragraphs
    // ------------------------------------------------------------------

    fn handle_text(
        &mut self,
        lines: &[LineRec],
        pos: &mut usize,
        match_titles: bool,
        out: &mut Vec<Node>,
    ) -> Option<SectionStart> {
        let line = lines[*pos];
        let next = lines.get(*pos + 1).copied();

        if let Some(next) = next {
            if !next.is_blank() && next.indent() == 0 {
                if let Some(ch) = adornment_char(self.sources.line_text(next)) {
                    let title_len = column_width(self.sources.line_text(line));
                    let ul_len = char_len(self.sources.line_text(next));
                    if ul_len >= title_len || ul_len >= 4 {
                        let raw = format!(
                            "{}\n{}",
                            self.sources.line_text(line),
                            self.sources.line_text(next)
                        );
                        if !match_titles {
                            let msg = messages::with_literal(
                                self.msg(
                                    messages::ERROR,
                                    "Unexpected section title.",
                                    next.source,
                                    next.lineno,
                                ),
                                &raw,
                            );
                            out.push(msg);
                            *pos += 2;
                            return None;
                        }
                        let mut msgs = Vec::new();
                        if ul_len < title_len {
                            msgs.push(messages::with_literal(
                                self.msg(
                                    messages::WARNING,
                                    "Title underline too short.",
                                    next.source,
                                    next.lineno,
                                ),
                                &raw,
                            ));
                        }
                        let span = self.span_of(lines, *pos, *pos + 1);
                        let title_lineno = line.lineno;
                        let underline_lineno = next.lineno;
                        let title = self.sources.line_text(line).trim().to_string();
                        *pos += 2;
                        return Some(SectionStart {
                            title,
                            style: (ch, false),
                            raw_lines: raw,
                            messages: msgs,
                            title_lineno,
                            underline_lineno,
                            span,
                        });
                    }
                    if match_titles {
                        // Demoted: INFO, then the lines parse as a paragraph.
                        out.push(self.msg(
                            messages::INFO,
                            "Possible title underline, too short for the title.\nTreating it as ordinary text because it's so short.",
                            next.source,
                            next.lineno,
                        ));
                    }
                    // fall through to paragraph (absorbs the underline line)
                }
            }
            if !next.is_blank() && next.indent() > 0 {
                // Single line + immediately indented block: definition list.
                self.parse_definition_list(lines, pos, out);
                return None;
            }
        }
        self.parse_paragraph_like(lines, pos, out);
        None
    }

    /// Paragraph: maximal run of adjacent column-0 non-blank lines, with
    /// docutils `::` literal-block chaining and the multi-line + indent
    /// "Unexpected indentation." recovery.
    fn parse_paragraph_like(&mut self, lines: &[LineRec], pos: &mut usize, out: &mut Vec<Node>) {
        let start = *pos;
        let mut end = *pos;
        while end < lines.len() && !lines[end].is_blank() && lines[end].indent() == 0 {
            end += 1;
        }
        let joined = self.join_lines(&lines[start..end]);
        let (text, expect_literal) = strip_literal_colons(&joined);
        let span = self.span_of(lines, start, end.saturating_sub(1));
        if !text.is_empty() {
            let result = self.inline(&text, span, lines[start].lineno);
            let mut para = Node::elem(kinds::PARAGRAPH, span);
            para.children = result.nodes;
            out.push(para);
            out.extend(result.messages);
        }
        *pos = end;

        // Multi-line paragraph directly followed by an indented line.
        if end < lines.len()
            && !lines[end].is_blank()
            && lines[end].indent() > 0
            && end - start >= 2
        {
            out.push(self.msg_sm(
                messages::ERROR,
                "Unexpected indentation.",
                lines[end].source,
                lines[end].lineno,
            ));
            // With a `::` trigger the indented block is STILL the literal
            // (fixture-verified); otherwise it becomes a block quote via the
            // ordinary element loop.
            if expect_literal {
                self.parse_literal_block(lines, pos, out);
            }
            return;
        }

        if expect_literal {
            self.parse_literal_block(lines, pos, out);
        }
    }

    fn parse_literal_block(&mut self, lines: &[LineRec], pos: &mut usize, out: &mut Vec<Node>) {
        let mut p = *pos;
        while p < lines.len() && lines[p].is_blank() {
            p += 1;
        }
        if p >= lines.len() {
            // Probe-verified: at EOF the warning still fires, anchored to
            // the line after the last one.
            let anchor = lines
                .last()
                .map(|l| (l.source, l.lineno + 1))
                .unwrap_or((0, 1));
            out.push(self.msg(
                messages::WARNING,
                "Literal block expected; none found.",
                anchor.0,
                anchor.1,
            ));
            *pos = p;
            return;
        }
        let first = lines[p];
        if first.indent() > 0 {
            // Indented literal block.
            let (block, consumed, _indent, terminator) = indented_block(lines, p);
            let text = self.join_lines(&block);
            let span = self.span_of(lines, p, p + consumed - 1);
            let mut lb = Node::elem(kinds::LITERAL_BLOCK, span);
            lb.set("xml:space", AttrValue::Str("preserve".to_string()));
            lb.children.push(Node::text_node(text, span));
            out.push(lb);
            *pos = p + consumed;
            if let Some((term_source, term_lineno)) = terminator {
                out.push(self.msg_sm(
                    messages::WARNING,
                    "Literal block ends without a blank line; unexpected unindent.",
                    term_source,
                    term_lineno,
                ));
            }
            return;
        }
        let quote_char = self
            .sources
            .line_text(first)
            .chars()
            .next()
            .filter(|c| ADORNMENT_CHARS.contains(*c));
        if let Some(qc) = quote_char {
            // Quoted literal block: consistent same-char-prefixed run.
            let mut endq = p;
            while endq < lines.len()
                && !lines[endq].is_blank()
                && lines[endq].indent() == 0
                && self.sources.line_text(lines[endq]).starts_with(qc)
            {
                endq += 1;
            }
            let text = self.join_lines(&lines[p..endq]);
            let span = self.span_of(lines, p, endq - 1);
            let mut lb = Node::elem(kinds::LITERAL_BLOCK, span);
            lb.set("xml:space", AttrValue::Str("preserve".to_string()));
            lb.children.push(Node::text_node(text, span));
            out.push(lb);
            if endq < lines.len() && !lines[endq].is_blank() {
                let text = if lines[endq].indent() > 0 {
                    "Unexpected indentation."
                } else {
                    "Inconsistent literal block quoting."
                };
                out.push(self.msg(
                    messages::ERROR,
                    text,
                    lines[endq].source,
                    lines[endq].lineno,
                ));
            }
            *pos = endq;
            return;
        }
        out.push(self.msg(
            messages::WARNING,
            "Literal block expected; none found.",
            first.source,
            first.lineno,
        ));
        *pos = p;
    }

    // ------------------------------------------------------------------
    // lists
    // ------------------------------------------------------------------

    fn parse_bullet_list(
        &mut self,
        lines: &[LineRec],
        pos: &mut usize,
        bullet: char,
        out: &mut Vec<Node>,
    ) {
        let start = *pos;
        let mut list = Node::elem(kinds::BULLET_LIST, Span::ZERO);
        list.set("bullet", AttrValue::Str(bullet.to_string()));
        let mut warn_line: Option<(u16, u32)> = None;
        loop {
            let item = self.parse_list_item(lines, pos, 1);
            list.children.push(item);
            let mut p = *pos;
            let mut saw_blank = false;
            while p < lines.len() && lines[p].is_blank() {
                p += 1;
                saw_blank = true;
            }
            if p >= lines.len() {
                *pos = p;
                break;
            }
            let line = lines[p];
            if line.indent() == 0
                && Self::bullet_marker(self.sources.line_text(line)) == Some(bullet)
            {
                *pos = p;
                continue;
            }
            if !saw_blank {
                warn_line = Some((line.source, line.lineno));
            }
            *pos = p;
            break;
        }
        list.span = self.span_of(lines, start, pos.saturating_sub(1));
        out.push(list);
        if let Some((source, lineno)) = warn_line {
            out.push(self.msg_sm(
                messages::WARNING,
                "Bullet list ends without a blank line; unexpected unindent.",
                source,
                lineno,
            ));
        }
    }

    /// Parse one list item whose marker occupies `marker_chars` characters on
    /// the current line. Content indent per docutils: marker + following
    /// spaces, or the next line's indent when the marker stands alone.
    /// Leaves `*pos` just past the item's content (trailing blank lines are
    /// left for the caller).
    fn parse_list_item(&mut self, lines: &[LineRec], pos: &mut usize, marker_chars: usize) -> Node {
        let marker_line = lines[*pos];
        let marker_text = self.sources.line_text(marker_line);
        let after_off = rest_after_offset(marker_text, marker_chars);
        let after = &marker_text[after_off..];
        let spaces = after.len() - after.trim_start_matches(' ').len();
        let rest_off = after_off + spaces;
        let rest_is_empty = marker_text.len() == rest_off;
        let start = *pos;

        let mut body: Vec<LineRec> = Vec::new();
        let content_indent;
        if rest_is_empty {
            // Fixture-verified: a bare marker's body may follow after blank
            // lines; the first indented line sets the content indent.
            let mut probe = start + 1;
            while probe < lines.len() && lines[probe].is_blank() {
                probe += 1;
            }
            match lines.get(probe) {
                Some(n) if !n.is_blank() && n.indent() > 0 => content_indent = n.indent(),
                _ => {
                    *pos = start + 1;
                    return Node::elem(kinds::LIST_ITEM, self.span_of(lines, start, start));
                }
            }
        } else {
            content_indent = marker_chars + spaces;
            body.push(self.rewrap_from(marker_line, rest_off));
        }

        let mut last_content = start;
        let mut pending_blanks: Vec<LineRec> = Vec::new();
        let mut scan = start + 1;
        while scan < lines.len() {
            let l = lines[scan];
            if l.is_blank() {
                pending_blanks.push(l);
                scan += 1;
                continue;
            }
            if l.indent() >= content_indent {
                body.append(&mut pending_blanks);
                body.push(l.dedented(content_indent));
                last_content = scan;
                scan += 1;
            } else {
                break;
            }
        }
        *pos = last_content + 1;

        let children = self.parse_nested(&body, "list_item");
        let mut item = Node::elem(kinds::LIST_ITEM, self.span_of(lines, start, last_content));
        item.children = children;
        item
    }

    fn try_enumerated_list(
        &mut self,
        lines: &[LineRec],
        pos: &mut usize,
        first: &Enumerator,
        out: &mut Vec<Node>,
    ) -> bool {
        let mut candidates = initial_candidates(&first.literal, first.auto);
        if candidates.is_empty() {
            return false;
        }
        if !self.enum_item_valid(lines, *pos, first, &candidates, first.auto) {
            return false;
        }
        let start = *pos;
        let mut warn_line: Option<(u16, u32)> = None;
        let mut items: Vec<Node> = Vec::new();
        let mut current = first.clone();
        // Fixture-verified: once an item is auto (#), explicit successors
        // invalidate; bare successors ("2." with no text) never continue.
        let mut auto_mode = first.auto;
        loop {
            let item = self.parse_list_item(lines, pos, current.marker_chars);
            items.push(item);
            let mut p = *pos;
            let mut saw_blank = false;
            while p < lines.len() && lines[p].is_blank() {
                p += 1;
                saw_blank = true;
            }
            if p >= lines.len() {
                *pos = p;
                break;
            }
            let line = lines[p];
            let mut accepted = false;
            if line.indent() == 0 {
                if let Some(e) = parse_enumerator(self.sources.line_text(line)) {
                    if e.prefix == first.prefix
                        && e.suffix == first.suffix
                        && !e.rest_empty
                        && !(auto_mode && !e.auto)
                    {
                        let narrowed = advance_candidates(&candidates, &e);
                        if !narrowed.is_empty()
                            && self.enum_item_valid(lines, p, &e, &narrowed, auto_mode || e.auto)
                        {
                            candidates = narrowed;
                            auto_mode |= e.auto;
                            current = e;
                            *pos = p;
                            accepted = true;
                        }
                    }
                }
            }
            if !accepted {
                if !saw_blank {
                    warn_line = Some((line.source, line.lineno));
                }
                *pos = p;
                break;
            }
        }

        let chosen = &candidates[0];
        let mut list = Node::elem(kinds::ENUMERATED_LIST, Span::ZERO);
        list.set("enumtype", AttrValue::Str(chosen.seq.to_string()));
        list.set("prefix", AttrValue::Str(first.prefix.to_string()));
        if chosen.initial != 1 {
            list.set("start", AttrValue::Int(chosen.initial as i64));
        }
        list.set("suffix", AttrValue::Str(first.suffix.to_string()));
        list.children = items;
        list.span = self.span_of(lines, start, pos.saturating_sub(1));
        let first_anchor = (lines[start].source, lines[start].lineno);
        out.push(list);
        if let Some((source, lineno)) = warn_line {
            out.push(self.msg_sm(
                messages::WARNING,
                "Enumerated list ends without a blank line; unexpected unindent.",
                source,
                lineno,
            ));
        }
        if chosen.initial != 1 {
            out.push(self.msg(
                messages::INFO,
                &format!(
                    "Enumerated list start value not ordinal-1: \"{}\" (ordinal {})",
                    first.literal, chosen.initial
                ),
                first_anchor.0,
                first_anchor.1,
            ));
        }
        true
    }

    /// docutils validates an enumerated item by its OWN next line: blank,
    /// EOF, indented continuation, or a valid successor enumerator.
    fn enum_item_valid(
        &self,
        lines: &[LineRec],
        at: usize,
        item: &Enumerator,
        candidates: &[EnumCandidate],
        auto_context: bool,
    ) -> bool {
        let next = match lines.get(at + 1) {
            None => return true,
            Some(n) => n,
        };
        if next.is_blank() || next.indent() > 0 {
            return true;
        }
        match parse_enumerator(self.sources.line_text(*next)) {
            Some(e)
                if e.prefix == item.prefix
                    && e.suffix == item.suffix
                    && !e.rest_empty
                    && !(auto_context && !e.auto) =>
            {
                !advance_candidates(candidates, &e).is_empty()
            }
            _ => false,
        }
    }

    // ------------------------------------------------------------------
    // definition lists
    // ------------------------------------------------------------------

    fn parse_definition_list(&mut self, lines: &[LineRec], pos: &mut usize, out: &mut Vec<Node>) {
        let start = *pos;
        let mut dl = Node::elem(kinds::DEFINITION_LIST, Span::ZERO);
        let mut warn_line: Option<(u16, u32)> = None;
        loop {
            let term_line = lines[*pos];
            let (block, consumed, _indent, terminator) = indented_block(lines, *pos + 1);
            let item_last = *pos + consumed;
            let mut item = Node::elem(
                kinds::DEFINITION_LIST_ITEM,
                self.span_of(lines, *pos, item_last),
            );
            let term_span = self.span_of(lines, *pos, *pos);
            let term_ends_in_colons = self.sources.line_text(term_line).ends_with("::");
            let mut parts = split_classifiers(self.sources.line_text(term_line)).into_iter();
            let term_text = parts.next().unwrap_or_default();
            let mut term_msgs = Vec::new();
            let inline = self.inline(&term_text, term_span, term_line.lineno);
            let mut term = Node::elem(kinds::TERM, term_span);
            term.children = inline.nodes;
            term_msgs.extend(inline.messages);
            item.children.push(term);
            for classifier in parts {
                let inline = self.inline(&classifier, term_span, term_line.lineno);
                let mut c = Node::elem(kinds::CLASSIFIER, term_span);
                c.children = inline.nodes;
                term_msgs.extend(inline.messages);
                item.children.push(c);
            }
            let mut definition =
                Node::elem(kinds::DEFINITION, self.span_of(lines, *pos + 1, item_last));
            // Fixture-verified: term/classifier inline messages land INSIDE
            // the definition, before its content.
            definition.children.append(&mut term_msgs);
            if term_ends_in_colons {
                // Probe-verified: docutils flags a term ending in `::`.
                definition.children.push(self.msg(
                    messages::INFO,
                    "Blank line missing before literal block (after the \"::\")? Interpreted as a definition list item.",
                    term_line.source,
                    term_line.lineno + 1,
                ));
            }
            definition
                .children
                .extend(self.parse_nested(&block, "definition"));
            item.children.push(definition);
            dl.children.push(item);
            *pos += 1 + consumed;

            // Another term? (column-0 text line + immediately indented body)
            let mut p = *pos;
            while p < lines.len() && lines[p].is_blank() {
                p += 1;
            }
            let continues = p < lines.len() && {
                let l = lines[p];
                let text = self.sources.line_text(l);
                let nxt = lines.get(p + 1);
                l.indent() == 0
                    && !l.is_blank()
                    && Self::bullet_marker(text).is_none()
                    && parse_enumerator(text).is_none()
                    && adornment_char(text).is_none()
                    && field_marker(text).is_none()
                    && option_group_marker(text).is_none()
                    && !text.starts_with(".. ")
                    && text != ".."
                    && !text.starts_with("| ")
                    && !text.starts_with(">>> ")
                    && !text.starts_with("__ ")
                    && nxt
                        .map(|n| !n.is_blank() && n.indent() > 0)
                        .unwrap_or(false)
            };
            if continues {
                *pos = p;
                continue;
            }
            if let Some(t) = terminator {
                warn_line = Some(t);
            }
            break;
        }
        dl.span = self.span_of(lines, start, pos.saturating_sub(1));
        out.push(dl);
        if let Some((source, lineno)) = warn_line {
            out.push(self.msg_sm(
                messages::WARNING,
                "Definition list ends without a blank line; unexpected unindent.",
                source,
                lineno,
            ));
        }
    }

    // ------------------------------------------------------------------
    // block quotes
    // ------------------------------------------------------------------

    fn parse_block_quote(&mut self, lines: &[LineRec], pos: &mut usize, out: &mut Vec<Node>) {
        let start = *pos;
        let (block, consumed, _indent, terminator) = indented_block(lines, *pos);
        *pos = start + consumed;
        let span = self.span_of(lines, start, start + consumed - 1);
        out.extend(self.block_quote_elements(&block, span));
        if let Some((source, lineno)) = terminator {
            out.push(self.msg_sm(
                messages::WARNING,
                "Block quote ends without a blank line; unexpected unindent.",
                source,
                lineno,
            ));
        }
    }

    /// docutils `Body.block_quote()`: build block_quote element(s) plus
    /// interleaved attribution messages from an already-extracted block.
    /// Shared by indented block quotes and the epigraph/highlights/
    /// pull-quote directives.
    fn block_quote_elements(&mut self, block: &[LineRec], span: Span) -> Vec<Node> {
        let mut out: Vec<Node> = Vec::new();
        // Split into blank-separated chunks; attribution chunks close quotes.
        let mut quotes: Vec<QuoteSegment> = Vec::new();
        let mut acc: Vec<LineRec> = Vec::new();
        let mut i = 0usize;
        while i < block.len() {
            if block[i].is_blank() {
                acc.push(block[i]);
                i += 1;
                continue;
            }
            let chunk_start = i;
            while i < block.len() && !block[i].is_blank() {
                i += 1;
            }
            let chunk = &block[chunk_start..i];
            // Probe-verified: an attribution needs preceding quote body —
            // a quote whose only content is "-- x" is a plain paragraph.
            let has_body = acc.iter().any(|l| !l.is_blank());
            match attribution_from_chunk(&self.sources, chunk, span) {
                Some(attr) if has_body => quotes.push((std::mem::take(&mut acc), Some(attr))),
                _ => acc.extend_from_slice(chunk),
            }
        }
        if !acc.iter().all(|l| l.is_blank()) || quotes.is_empty() {
            quotes.push((acc, None));
        }
        for (body, attribution) in quotes {
            let mut quote = Node::elem(kinds::BLOCK_QUOTE, span);
            quote.children = self.parse_nested(&body, "block_quote");
            let mut attr_messages = Vec::new();
            if let Some((raw_attr, lineno)) = attribution {
                let raw = raw_attr.astext();
                let inline = self.inline(&raw, raw_attr.span, lineno);
                let mut a = Node::elem(kinds::ATTRIBUTION, raw_attr.span);
                a.children = inline.nodes;
                attr_messages = inline.messages;
                quote.children.push(a);
            }
            if quote.children.is_empty() {
                continue;
            }
            out.push(quote);
            out.append(&mut attr_messages);
        }
        out
    }

    // ------------------------------------------------------------------
    // doctest + line blocks
    // ------------------------------------------------------------------

    fn parse_doctest(&mut self, lines: &[LineRec], pos: &mut usize, out: &mut Vec<Node>) {
        // Fixture-verified: a doctest block runs to the next BLANK line,
        // absorbing indented continuation/output lines verbatim.
        let start = *pos;
        let mut end = *pos;
        while end < lines.len() && !lines[end].is_blank() {
            end += 1;
        }
        let text = self.join_lines(&lines[start..end]);
        let span = self.span_of(lines, start, end - 1);
        let mut dt = Node::elem(kinds::DOCTEST_BLOCK, span);
        dt.set("xml:space", AttrValue::Str("preserve".to_string()));
        dt.children.push(Node::text_node(text, span));
        out.push(dt);
        *pos = end;
    }

    fn parse_line_block(&mut self, lines: &[LineRec], pos: &mut usize, out: &mut Vec<Node>) {
        let start = *pos;
        // (depth, text): depth None on bare `|` lines inherits the previous
        // line's depth (fixture-verified). Continuations dedent by the FIRST
        // continuation line's indent, preserving deeper relative indents.
        let mut items: Vec<(Option<usize>, String)> = Vec::new();
        let mut cont_dedent: Option<usize> = None;
        let mut p = *pos;
        while p < lines.len() && !lines[p].is_blank() {
            let l = lines[p];
            let text = self.sources.line_text(l);
            if l.indent() == 0 && (text == "|" || text.starts_with("| ")) {
                cont_dedent = None;
                if text == "|" {
                    items.push((None, String::new()));
                } else {
                    let content = &text[2..];
                    let depth = content.len() - content.trim_start_matches(' ').len();
                    items.push((Some(depth), content[depth..].to_string()));
                }
                p += 1;
            } else if l.indent() > 0 && !items.is_empty() {
                let dedent = *cont_dedent.get_or_insert(l.indent());
                let dedent = dedent.min(l.indent());
                if let Some(last) = items.last_mut() {
                    if !last.1.is_empty() {
                        last.1.push('\n');
                    }
                    last.1.push_str(&text[dedent..]);
                }
                p += 1;
            } else {
                break;
            }
        }
        // Resolve inherited depths and inline-parse each line's text.
        let span = self.span_of(lines, start, p - 1);
        let first_lineno = lines[start].lineno;
        let mut resolved: Vec<(usize, Vec<Node>)> = Vec::with_capacity(items.len());
        let mut lb_messages: Vec<Node> = Vec::new();
        let mut prev_depth = 0usize;
        for (depth, text) in items {
            let d = depth.unwrap_or(prev_depth);
            prev_depth = d;
            if text.is_empty() {
                resolved.push((d, Vec::new()));
            } else {
                let inline = self.inline(&text, span, first_lineno);
                lb_messages.extend(inline.messages);
                resolved.push((d, inline.nodes));
            }
        }
        out.push(build_line_block(&mut resolved, span, 0));
        out.append(&mut lb_messages);
        // Fixture-verified: warning anchored to the LAST line-block line.
        if p < lines.len() && !lines[p].is_blank() {
            out.push(self.msg_sm(
                messages::WARNING,
                "Line block ends without a blank line.",
                lines[p - 1].source,
                lines[p - 1].lineno,
            ));
        }
        *pos = p;
    }

    // ------------------------------------------------------------------
    // explicit markup: comments + targets
    // ------------------------------------------------------------------

    fn parse_explicit(&mut self, lines: &[LineRec], pos: &mut usize, out: &mut Vec<Node>) {
        let line = lines[*pos];
        // Local handle: `rest` must stay readable across the `&mut self`
        // construct dispatches below.
        let src = self.sources.arc(line.source);
        let line_text = line.slice(&src);
        // docutils consumes ALL whitespace after `..` (fixture-verified for
        // multi-space forms).
        let rest = if line_text == ".." {
            ""
        } else {
            line_text[2..].trim_start()
        };

        if rest.starts_with('[') {
            if let Some(next_pos) = self.try_footnote_def(lines, pos, rest, out) {
                *pos = next_pos;
                self.warn_explicit_markup_end(lines, *pos, out);
                return;
            }
        }
        // docutils explicit_construct(): a construct whose parse raises
        // MarkupError queues a WARNING and falls through to the comment
        // path, which re-absorbs the whole block (through internal blanks).
        let mut construct_error: Option<Node> = None;
        if rest.starts_with('_') {
            // Target attempt: the marker (name + link) may span ADJACENT
            // indented continuation lines; parse the joined form.
            let start = *pos;
            let lineno = line.lineno;
            let mut consumed = 0usize;
            while lines
                .get(start + 1 + consumed)
                .map(|l| !l.is_blank() && l.indent() > 0)
                .unwrap_or(false)
            {
                consumed += 1;
            }
            let cont: Vec<&str> = lines[start + 1..start + 1 + consumed]
                .iter()
                .map(|l| self.sources.line_text(*l).trim())
                .collect();
            let joined = if cont.is_empty() {
                rest.to_string()
            } else {
                format!("{}\n{}", rest, cont.join("\n"))
            };
            *pos = start + 1 + consumed;
            let span = self.span_of(lines, start, start + consumed);
            match parse_target_marker(&joined) {
                Some(marker) => {
                    let mut target = Node::elem(kinds::TARGET, span);
                    let mut internal = false;
                    let mut refuri_val: Option<String> = None;
                    if marker.anonymous {
                        target.set("anonymous", AttrValue::Int(1));
                    } else {
                        target
                            .attrs
                            .names
                            .push(ids::fully_normalize_name(&marker.name));
                    }
                    if marker.link.is_empty() {
                        internal = true;
                    } else if let Some(refname) = reference_name_from_link(&marker.link) {
                        target.set("refname", AttrValue::Str(refname));
                    } else {
                        let uri: String = marker
                            .link
                            .chars()
                            .filter(|c| !c.is_whitespace() && *c != '\\')
                            .collect();
                        refuri_val = Some(uri.clone());
                        target.set("refuri", AttrValue::Str(uri));
                    }
                    let msg = if marker.anonymous {
                        self.registry.set_id_anonymous(&mut target);
                        None
                    } else {
                        let source_path = self.sources.arc_path(line.source);
                        self.registry.set_id_explicit(
                            &mut target,
                            lineno,
                            &source_path,
                            internal,
                            refuri_val.as_deref(),
                        )
                    };
                    if let Some(m) = msg {
                        out.push(m);
                    }
                    out.push(target);
                }
                None => {
                    // Malformed target: queue the WARNING and fall through
                    // to the comment path below (fixture-verified: the
                    // comment re-absorbs the block through blank lines).
                    *pos = start;
                    construct_error = Some(self.msg(
                        messages::WARNING,
                        "malformed hyperlink target.",
                        line.source,
                        lineno,
                    ));
                }
            }
            if construct_error.is_none() {
                self.warn_explicit_markup_end(lines, *pos, out);
                return;
            }
        }

        // Substitution definitions dispatch BEFORE directives
        // (states.py:2441-2483 construct order). The construct pattern
        // requires a non-space char after `|` (`(?![ ])`) — `.. | x` is a
        // plain comment, not a malformed substitution (review finding 19).
        if construct_error.is_none()
            && rest.starts_with('|')
            && !matches!(rest[1..].chars().next(), None | Some(' '))
            && self.parse_substitution_def(lines, pos, rest, out, &mut construct_error)
        {
            return;
        }

        if construct_error.is_none() {
            if let Some((name, first_rest)) = directive_marker(rest) {
                self.parse_directive(lines, pos, &name, first_rest, out);
                self.warn_explicit_markup_end(lines, *pos, out);
                return;
            }
        }

        // Comment. Probe-verified continuation rules: a comment with first-
        // line text absorbs the following indented block THROUGH internal
        // blank lines; a bare `..` takes a body only when the indented block
        // is ADJACENT (`..` + blank + indent leaves an empty comment and a
        // block quote).
        let start = *pos;
        let adjacent_body = lines
            .get(start + 1)
            .map(|l| !l.is_blank() && l.indent() > 0)
            .unwrap_or(false);
        let consume_block = !rest.is_empty() || adjacent_body;
        let (block, consumed) = if consume_block {
            let (block, consumed, _indent, _terminator) = indented_block(lines, start + 1);
            (block, consumed)
        } else {
            (Vec::new(), 0)
        };
        *pos = start + 1 + consumed;
        let span = self.span_of(lines, start, start + consumed);
        let mut text_lines: Vec<String> = Vec::new();
        if !rest.is_empty() {
            text_lines.push(rest.to_string());
        }
        let mut body: &[LineRec] = &block;
        if rest.is_empty() {
            while body.first().map(|l| l.is_blank()).unwrap_or(false) {
                body = &body[1..];
            }
        }
        for l in body {
            text_lines.push(self.sources.line_text(*l).to_string());
        }
        let mut comment = Node::elem(kinds::COMMENT, span);
        comment.set("xml:space", AttrValue::Str("preserve".to_string()));
        if !text_lines.is_empty() {
            comment
                .children
                .push(Node::text_node(text_lines.join("\n"), span));
        }
        out.push(comment);
        if let Some(err) = construct_error {
            out.push(err);
        }
        self.warn_explicit_markup_end(lines, *pos, out);
    }

    /// `.. [label]` footnote and citation definitions. Returns the new
    /// position past the construct, or None when `rest` is not a valid
    /// footnote/citation marker (falls through to comment).
    fn try_footnote_def(
        &mut self,
        lines: &[LineRec],
        pos: &mut usize,
        rest: &str,
        out: &mut Vec<Node>,
    ) -> Option<usize> {
        let chars: Vec<char> = rest.chars().collect();
        let mut j = 1usize; // past '['
        let label_start = j;
        match chars.get(j) {
            Some('#') => {
                j += 1;
                if let Some(len) = match_simplename_chars(&chars, j) {
                    j += len;
                }
            }
            Some('*') => j += 1,
            _ => j += match_simplename_chars(&chars, j)?,
        }
        if chars.get(j) != Some(&']') {
            return None;
        }
        let after = j + 1;
        if !(chars.len() == after || chars.get(after) == Some(&' ')) {
            return None;
        }
        let label: String = chars[label_start..j].iter().collect();
        let start = *pos;
        let lineno = lines[start].lineno;

        // Body: first-line remainder + following indented block (blanks
        // between marker and block allowed; docutils get_first_known_indented).
        // docutils' footnote pattern consumes ALL whitespace after `]`.
        let mut rest_from = after;
        while chars.get(rest_from) == Some(&' ') {
            rest_from += 1;
        }
        let first_rest: String = if rest_from > after {
            chars
                .get(rest_from..)
                .map(|c| c.iter().collect())
                .unwrap_or_default()
        } else {
            String::new()
        };
        let (block, consumed, _indent, _term) = indented_block(lines, start + 1);
        let mut body: Vec<LineRec> = Vec::new();
        if !first_rest.trim().is_empty() {
            // remainder starts at a virtual column; treat as its own line
            let marker_text = self.sources.line_text(lines[start]);
            let off = rest_after_offset(
                marker_text,
                marker_text.chars().count() - first_rest.chars().count(),
            );
            body.push(self.rewrap_from(lines[start], off));
        }
        for l in &block {
            body.push(*l);
        }

        let is_citation =
            !label.starts_with('#') && label != "*" && !label.chars().all(|c| c.is_ascii_digit());
        let kind = if is_citation {
            kinds::CITATION
        } else {
            kinds::FOOTNOTE
        };
        let span = self.span_of(lines, start, start + consumed);
        let mut node = Node::elem(kind, span);
        let mut has_label_child = false;
        if is_citation {
            node.attrs.names.push(ids::fully_normalize_name(&label));
            has_label_child = true;
        } else if label == "*" {
            node.set("auto", AttrValue::Str("*".to_string()));
        } else if let Some(rest_label) = label.strip_prefix('#') {
            node.set("auto", AttrValue::Int(1));
            if !rest_label.is_empty() {
                node.attrs.names.push(ids::fully_normalize_name(rest_label));
            }
        } else {
            node.attrs.names.push(ids::fully_normalize_name(&label));
            has_label_child = true;
        }
        let msg = if node.attrs.names.is_empty() {
            self.registry.set_id_anonymous(&mut node);
            None
        } else {
            let source_path = self.sources.arc_path(lines[start].source);
            self.registry
                .set_id_explicit(&mut node, lineno, &source_path, true, None)
        };
        if has_label_child {
            let mut lab = Node::elem(kinds::LABEL, span);
            lab.children.push(Node::text_node(label.clone(), span));
            node.children.push(lab);
        }
        if let Some(m) = msg {
            node.children.push(m);
        }
        let content = self.parse_nested(&body, if is_citation { "citation" } else { "footnote" });
        if content.is_empty() {
            let text = if is_citation {
                "Citation content expected."
            } else {
                "Footnote content expected."
            };
            node.children
                .push(self.msg(messages::WARNING, text, lines[start].source, lineno));
        } else {
            node.children.extend(content);
        }
        out.push(node);
        Some(start + 1 + consumed)
    }

    /// Field lists: `:name: value` markers (probe-verified regex port).
    fn parse_field_list(&mut self, lines: &[LineRec], pos: &mut usize, out: &mut Vec<Node>) {
        let start = *pos;
        let mut fl = Node::elem(kinds::FIELD_LIST, Span::ZERO);
        let mut warn_line: Option<(u16, u32)> = None;
        loop {
            let line = lines[*pos];
            let line_text = self.sources.line_text(line);
            let (name_raw, body_start) = field_marker(line_text).expect("checked by caller");
            let lineno = line.lineno;
            let field_span = self.span_of(lines, *pos, *pos);
            // body: marker-line remainder + any-indent continuation block
            let first_rest = line_text[body_start..].trim_start();
            let rest_offset = (!first_rest.is_empty()).then(|| line_text.len() - first_rest.len());
            let (block, consumed, _i, terminator) = indented_block(lines, *pos + 1);
            let mut body_lines: Vec<LineRec> = Vec::new();
            if let Some(offset) = rest_offset {
                body_lines.push(self.rewrap_from(line, offset));
            }
            body_lines.extend(block.iter().copied());
            *pos += 1 + consumed;

            let name_inline = self.inline(&name_raw, field_span, lineno);
            let mut field = Node::elem(kinds::FIELD, field_span);
            let mut fname = Node::elem(kinds::FIELD_NAME, field_span);
            fname.children = name_inline.nodes;
            field.children.push(fname);
            let mut fbody = Node::elem(kinds::FIELD_BODY, field_span);
            fbody.children.extend(name_inline.messages);
            fbody
                .children
                .extend(self.parse_nested(&body_lines, "field_body"));
            field.children.push(fbody);
            fl.children.push(field);

            // continue on the next field marker (blanks allowed between)
            let mut p = *pos;
            while p < lines.len() && lines[p].is_blank() {
                p += 1;
            }
            let continues = p < lines.len()
                && lines[p].indent() == 0
                && field_marker(self.sources.line_text(lines[p])).is_some();
            if continues {
                *pos = p;
                continue;
            }
            let _ = terminator;
            // Adjacency: any non-blank line directly after the field body
            // (indented-block terminator OR a col-0 line) warns.
            if let Some(l) = lines.get(*pos) {
                if !l.is_blank() {
                    warn_line = Some((l.source, l.lineno));
                }
            }
            break;
        }
        fl.span = self.span_of(lines, start, pos.saturating_sub(1));
        out.push(fl);
        if let Some((source, lineno)) = warn_line {
            out.push(self.msg_sm(
                messages::WARNING,
                "Field list ends without a blank line; unexpected unindent.",
                source,
                lineno,
            ));
        }
    }

    /// An option marker line is only a list item when it has a two-space
    /// description or an indented following line (else: paragraph).
    fn option_item_viable(&self, lines: &[LineRec], at: usize) -> bool {
        let (_, desc) = match option_group_marker(self.sources.line_text(lines[at])) {
            Some(r) => r,
            None => return false,
        };
        if !desc.is_empty() {
            return true;
        }
        lines
            .get(at + 1)
            .map(|l| !l.is_blank() && l.indent() > 0)
            .unwrap_or(false)
    }

    fn parse_option_list(&mut self, lines: &[LineRec], pos: &mut usize, out: &mut Vec<Node>) {
        let start = *pos;
        let mut ol = Node::elem(kinds::OPTION_LIST, Span::ZERO);
        let mut warn_line: Option<(u16, u32)> = None;
        loop {
            let line = lines[*pos];
            let line_text = self.sources.line_text(line);
            let (specs, desc) = option_group_marker(line_text).expect("checked by caller");
            let desc_offset = (!desc.is_empty()).then(|| line_text.len() - desc.len());
            let span = self.span_of(lines, *pos, *pos);
            let (block, consumed, _i, terminator) = indented_block(lines, *pos + 1);
            let mut body_lines: Vec<LineRec> = Vec::new();
            if let Some(offset) = desc_offset {
                body_lines.push(self.rewrap_from(line, offset));
            }
            body_lines.extend(block.iter().copied());
            *pos += 1 + consumed;

            let mut item = Node::elem(kinds::OPTION_LIST_ITEM, span);
            let mut group = Node::elem(kinds::OPTION_GROUP, span);
            for (opt_string, arg) in specs {
                let mut opt = Node::elem(kinds::OPTION, span);
                let mut os = Node::elem(kinds::OPTION_STRING, span);
                os.children.push(Node::text_node(opt_string, span));
                opt.children.push(os);
                if let Some((delim, argtext)) = arg {
                    let mut oa = Node::elem(kinds::OPTION_ARGUMENT, span);
                    oa.set("delimiter", AttrValue::Str(delim));
                    oa.children.push(Node::text_node(argtext, span));
                    opt.children.push(oa);
                }
                group.children.push(opt);
            }
            item.children.push(group);
            let mut description = Node::elem(kinds::DESCRIPTION, span);
            description.children = self.parse_nested(&body_lines, "description");
            item.children.push(description);
            ol.children.push(item);

            let mut p = *pos;
            while p < lines.len() && lines[p].is_blank() {
                p += 1;
            }
            let continues = p < lines.len()
                && lines[p].indent() == 0
                && option_group_marker(self.sources.line_text(lines[p])).is_some()
                && self.option_item_viable(lines, p);
            if continues {
                *pos = p;
                continue;
            }
            let _ = terminator;
            if let Some(l) = lines.get(*pos) {
                if !l.is_blank() {
                    warn_line = Some((l.source, l.lineno));
                }
            }
            break;
        }
        ol.span = self.span_of(lines, start, pos.saturating_sub(1));
        out.push(ol);
        if let Some((source, lineno)) = warn_line {
            out.push(self.msg_sm(
                messages::WARNING,
                "Option list ends without a blank line; unexpected unindent.",
                source,
                lineno,
            ));
        }
    }

    // ------------------------------------------------------------------
    // grid tables (docutils tableparser.GridTableParser port)
    // ------------------------------------------------------------------

    fn parse_grid_table(&mut self, lines: &[LineRec], pos: &mut usize, out: &mut Vec<Node>) {
        let start = *pos;
        // isolate: consume until blank line
        let mut end = *pos;
        while end < lines.len() && !lines[end].is_blank() {
            end += 1;
        }
        let mut block: Vec<LineRec> = lines[start..end].to_vec();
        *pos = end;
        // docutils left-edge check: trim at the first line not starting
        // with '+' or '|'; the remainder re-parses and a blank-line
        // warning fires. The trim index feeds the stale-line quirk of the
        // bottom-corrupt error.
        let mut trailing_warning = None;
        let mut stale_i = block.len() - 1;
        let mut edge_trim: Option<(usize, u16, u32)> = None;
        for (i, l) in block.iter().enumerate().skip(1) {
            let t = self.sources.line_text(*l).trim_end();
            if !(t.starts_with('+') || t.starts_with('|')) {
                stale_i = i;
                edge_trim = Some((i, l.source, l.lineno));
                break;
            }
        }
        if let Some((i, source, lineno)) = edge_trim {
            trailing_warning = Some(self.msg(
                messages::WARNING,
                "Blank line required after table.",
                source,
                lineno,
            ));
            block.truncate(i);
            *pos = start + i;
        }
        // docutils trims a non-border tail back to the LAST valid border
        // (the remainder re-parses, with a blank-line-required warning),
        // BEFORE any alignment checks.
        if !is_grid_table_top(self.sources.line_text(block[block.len() - 1]).trim_end()) {
            let mut found = None;
            for i in (2..block.len() - 1).rev() {
                if is_grid_table_top(self.sources.line_text(block[i]).trim_end()) {
                    found = Some(i);
                    break;
                }
            }
            if let Some(i) = found {
                let next = block[i + 1];
                block.truncate(i + 1);
                *pos = start + i + 1;
                if trailing_warning.is_none() {
                    trailing_warning = Some(self.msg(
                        messages::WARNING,
                        "Blank line required after table.",
                        next.source,
                        next.lineno,
                    ));
                }
            }
        }
        let raw_block: Vec<String> = block
            .iter()
            .map(|l| self.sources.line_text(*l).to_string())
            .collect();
        let msg_path = self.sources.path(block[0].source).to_string();
        let malformed = |detail: &str, lineno: u32| -> Node {
            messages::with_literal(
                messages::system_message(
                    messages::ERROR,
                    &format!("Malformed table.\n{detail}"),
                    lineno,
                    &msg_path,
                ),
                raw_block.join("\n").trim_end(),
            )
        };
        // right-border alignment (DISPLAY columns: east-asian wide = 2)
        let width = column_width(raw_block[0].trim_end());
        for (l, raw) in block.iter().zip(&raw_block).skip(1) {
            let t = raw.trim_end();
            if column_width(t) != width || !(t.ends_with('+') || t.ends_with('|')) {
                out.push(malformed("Right border not aligned or missing.", l.lineno));
                if let Some(w) = trailing_warning {
                    out.push(w);
                }
                return;
            }
        }
        // bottom border must be a grid border (line anchor reproduces
        // docutils' stale-index quirk: the last line the edge scans reached)
        if !is_grid_table_top(raw_block[raw_block.len() - 1].trim_end()) {
            let lineno = lines[(start + stale_i).min(lines.len() - 1)].lineno;
            out.push(malformed("Bottom border missing or corrupt.", lineno));
            if let Some(w) = trailing_warning {
                out.push(w);
            }
            return;
        }

        // grid as DISPLAY-column matrix (wide chars followed by a filler;
        // head/body sep '=' converted to '-')
        let mut grid: Vec<Vec<char>> = raw_block
            .iter()
            .map(|l| {
                let mut row = Vec::new();
                for c in l.trim_end().chars() {
                    row.push(c);
                    if unicode_width::UnicodeWidthChar::width(c).unwrap_or(1) == 2 {
                        row.push('\u{fffd}');
                    }
                }
                row
            })
            .collect();
        let mut head_sep: Option<usize> = None;
        for (i, row) in grid.iter_mut().enumerate() {
            let s: String = row.iter().collect();
            if is_grid_head_sep(&s) {
                if let Some(first) = head_sep {
                    out.push(malformed(
                        &format!(
                            "Multiple head/body row separators (table lines {} and {}); only one allowed.",
                            first + 1,
                            i + 1
                        ),
                        block[0].lineno,
                    ));
                    return;
                }
                head_sep = Some(i);
                for c in row.iter_mut() {
                    if *c == '=' {
                        *c = '-';
                    }
                }
            }
        }
        let nrows = grid.len();
        let at = |r: usize, c: usize| -> char {
            *grid.get(r).and_then(|row| row.get(c)).unwrap_or(&' ')
        };

        // trace cells from top-left corners
        let mut cells: Vec<(usize, usize, usize, usize)> = Vec::new();
        let mut colseps: Vec<usize> = vec![0];
        let mut rowseps: Vec<usize> = vec![0];
        let mut corners: Vec<(usize, usize)> = vec![(0, 0)];
        let mut done_to: Vec<(usize, usize)> = Vec::new(); // (left, bottom) per traced cell
        while let Some((top, left)) = corners.pop() {
            if cells
                .iter()
                .any(|(t, l, b, r)| *t <= top && top < *b && *l <= left && left < *r)
            {
                continue;
            }
            if at(top, left) != '+' {
                continue;
            }
            if let Some((bottom, right, mut cseps, mut rseps)) = trace_cell(&grid, top, left) {
                cells.push((top, left, bottom, right));
                colseps.append(&mut cseps);
                rowseps.append(&mut rseps);
                corners.push((top, right));
                corners.push((bottom, left));
                done_to.push((left, bottom));
                corners.sort();
                corners.dedup();
            }
        }
        colseps.sort_unstable();
        colseps.dedup();
        rowseps.sort_unstable();
        rowseps.dedup();

        // completeness: every column spanned to the bottom
        let bottom_row = nrows - 1;
        if rowseps.last() != Some(&bottom_row) && !cells.is_empty() {
            out.push(malformed(
                "Malformed table; parse incomplete.",
                block[0].lineno,
            ));
            return;
        }
        let _ = done_to;

        // structure
        let ncols = colseps.len().saturating_sub(1);
        let colwidths: Vec<usize> = colseps.windows(2).map(|w| w[1] - w[0] - 1).collect();
        let row_of = |o: usize| rowseps.iter().position(|r| *r == o);
        let col_of = |o: usize| colseps.iter().position(|c| *c == o);
        let nrows_struct = rowseps.len().saturating_sub(1);
        // rows[r][c] = Option<entry>
        let mut entries: Vec<Vec<Option<Node>>> = vec![];
        for _ in 0..nrows_struct {
            entries.push((0..ncols).map(|_| None).collect());
        }
        let mut covered: Vec<Vec<bool>> = vec![vec![false; ncols]; nrows_struct];
        let mut cell_list = cells.clone();
        cell_list.sort();
        for (top, left, bottom, right) in cell_list {
            let (Some(rn), Some(cn), Some(rb), Some(cr)) =
                (row_of(top), col_of(left), row_of(bottom), col_of(right))
            else {
                continue;
            };
            if covered[rn][cn] {
                continue;
            }
            let morerows = rb - rn - 1;
            let morecols = cr - cn - 1;
            for row in covered.iter_mut().take(rb).skip(rn) {
                for cell in row.iter_mut().take(cr).skip(cn) {
                    *cell = true;
                }
            }
            let span = self.span_of(lines, start + top, start + bottom);
            let mut entry = Node::elem(kinds::ENTRY, span);
            if morecols > 0 {
                entry.set("morecols", AttrValue::Int(morecols as i64));
            }
            if morerows > 0 {
                entry.set("morerows", AttrValue::Int(morerows as i64));
            }
            // cell block: rows top+1..bottom, cols left+1..right
            let mut cell_lines: Vec<LineRec> = Vec::new();
            for l in block.iter().take(bottom).skip(top + 1) {
                let (s, e) = display_range(self.sources.line_text(*l), left + 1, right);
                cell_lines.push(self.rewrap_range(*l, s, e));
            }
            let base = cell_lines
                .iter()
                .filter(|l| !self.sources.line_text(**l).trim().is_empty())
                .map(|l| l.indent())
                .min()
                .unwrap_or(0);
            let dedented: Vec<LineRec> = cell_lines
                .iter()
                .map(|l| {
                    if self.sources.line_text(*l).trim().is_empty() {
                        self.blank_at(*l)
                    } else {
                        let d = l.dedented(base);
                        // strip trailing whitespace inside the cell view
                        let trimmed = self.sources.line_text(d).trim_end().len();
                        self.rewrap_range(d, 0, trimmed)
                    }
                })
                .collect();
            if dedented.iter().any(|l| !l.is_blank()) {
                self.line_bias += 1;
                entry.children = self.parse_nested(&dedented, "entry");
                self.line_bias -= 1;
            }
            entries[rn][cn] = Some(entry);
        }

        let table_span = self.span_of(lines, start, end.saturating_sub(1));
        let mut table = Node::elem(kinds::TABLE, table_span);
        let mut tgroup = Node::elem(kinds::TGROUP, table_span);
        tgroup.set("cols", AttrValue::Int(ncols as i64));
        for w in &colwidths {
            let mut cs = Node::elem(kinds::COLSPEC, table_span);
            cs.set("colwidth", AttrValue::Int(*w as i64));
            tgroup.children.push(cs);
        }
        let head_rows = head_sep.and_then(row_of).unwrap_or(0);
        let build_rows = |range: std::ops::Range<usize>, entries: &mut Vec<Vec<Option<Node>>>| {
            let mut rows = Vec::new();
            for r in range {
                let mut row = Node::elem(kinds::ROW, table_span);
                for slot in entries[r].iter_mut() {
                    if let Some(e) = slot.take() {
                        row.children.push(e);
                    }
                }
                rows.push(row);
            }
            rows
        };
        if head_sep.is_some() && head_rows > 0 {
            let mut thead = Node::elem(kinds::THEAD, table_span);
            thead.children = build_rows(0..head_rows, &mut entries);
            tgroup.children.push(thead);
        } else if head_sep.is_some() {
            let mut thead = Node::elem(kinds::THEAD, table_span);
            thead.children = build_rows(0..0, &mut entries);
            let _ = &mut thead;
            tgroup.children.push(thead);
        }
        let mut tbody = Node::elem(kinds::TBODY, table_span);
        tbody.children = build_rows(head_rows..nrows_struct, &mut entries);
        tgroup.children.push(tbody);
        table.children.push(tgroup);
        out.push(table);
        if let Some(w) = trailing_warning {
            out.push(w);
        }
    }

    fn parse_simple_table(&mut self, lines: &[LineRec], pos: &mut usize, out: &mut Vec<Node>) {
        let start = *pos;
        let toplen = char_len(self.sources.line_text(lines[start]).trim_end());
        // isolate: find border candidates (=-runs line, same stripped length)
        let mut found = 0usize;
        let mut found_at = None;
        let mut end = None;
        let mut i = start + 1;
        while i < lines.len() {
            let t = self.sources.line_text(lines[i]).trim_end();
            if is_simple_table_border(t) {
                if char_len(t) != toplen {
                    let raw = self.join_lines(&lines[start..=i]);
                    out.push(messages::with_literal(
                        self.msg(
                            messages::ERROR,
                            "Malformed table.\nBottom border or header rule does not match top border.",
                            lines[i].source,
                            lines[i].lineno,
                        ),
                        raw.trim_end(),
                    ));
                    *pos = i + 1;
                    return;
                }
                found += 1;
                found_at = Some(i);
                if found == 2
                    || i + 1 >= lines.len()
                    || lines.get(i + 1).map(|l| l.is_blank()).unwrap_or(true)
                {
                    end = Some(i);
                    break;
                }
            }
            i += 1;
        }
        let Some(end) = end else {
            // no bottom border
            let (block_end, extra) = match found_at {
                Some(f) => (f, " or no blank line after table bottom"),
                None => (i.saturating_sub(1).max(start), ""),
            };
            let raw = self.join_lines(&lines[start..=block_end.min(lines.len() - 1)]);
            out.push(messages::with_literal(
                self.msg(
                    messages::ERROR,
                    &format!("Malformed table.\nNo bottom table border found{extra}."),
                    lines[start].source,
                    lines[start].lineno,
                ),
                raw.trim_end(),
            ));
            *pos = block_end + 1;
            if !extra.is_empty() {
                if let Some(l) = lines.get(*pos).filter(|l| !l.is_blank()) {
                    out.push(self.msg(
                        messages::WARNING,
                        "Blank line required after table.",
                        l.source,
                        l.lineno,
                    ));
                }
            }
            return;
        };
        *pos = end + 1;
        let blank_after_ok = lines.get(*pos).map(|l| l.is_blank()).unwrap_or(true);

        let block: Vec<LineRec> = lines[start..=end].to_vec();
        let raw_block: Vec<String> = block
            .iter()
            .map(|l| self.sources.line_text(*l).to_string())
            .collect();
        let msg_path = self.sources.path(block[0].source).to_string();
        let malformed = |detail: &str, lineno: u32| -> Node {
            messages::with_literal(
                messages::system_message(
                    messages::ERROR,
                    &format!("Malformed table.\n{detail}"),
                    lineno,
                    &msg_path,
                ),
                raw_block.join("\n").trim_end(),
            )
        };

        // columns from the top border '=' runs
        let top_chars: Vec<char> = raw_block[0].trim_end().chars().collect();
        let mut columns: Vec<(usize, usize)> = Vec::new();
        let mut run_start = None;
        for (ci, c) in top_chars.iter().enumerate() {
            if *c == '=' {
                if run_start.is_none() {
                    run_start = Some(ci);
                }
            } else if let Some(s) = run_start.take() {
                columns.push((s, ci));
            }
        }
        if let Some(s) = run_start {
            columns.push((s, top_chars.len()));
        }
        let border_end = columns.last().map(|(_, e)| *e).unwrap_or(0);

        // interior head/body sep: full-'='-runs line converted to span line
        let mut head_sep_row: Option<usize> = None; // index into block
        let mut work: Vec<String> = raw_block.iter().map(|l| l.trim_end().to_string()).collect();
        let n = work.len();
        for (bi, w) in work.iter_mut().enumerate() {
            if bi > 0 && bi < n - 1 && is_simple_table_border(w) {
                head_sep_row = Some(bi);
                *w = w.replace('=', "-");
            }
        }
        let bottom = work.len() - 1;
        work[0] = work[0].replace('=', "-");
        work[bottom] = work[bottom].replace('=', "-");

        // rows: (start_line_idx, end_line_idx_exclusive, colspec)
        struct RawRow {
            start: usize,
            end: usize,
            cols: Vec<(usize, usize)>,
        }
        let parse_span_cols =
            |line: &str, table_line: usize| -> Result<Vec<(usize, usize)>, Box<Node>> {
                let chars: Vec<char> = line.chars().collect();
                let mut cols = Vec::new();
                let mut rs = None;
                for (ci, c) in chars.iter().enumerate() {
                    if *c == '-' {
                        if rs.is_none() {
                            rs = Some(ci);
                        }
                    } else if let Some(s) = rs.take() {
                        cols.push((s, ci));
                    }
                }
                if let Some(s) = rs {
                    cols.push((s, chars.len()));
                }
                if cols.last().map(|(_, e)| *e) != Some(border_end) {
                    return Err(Box::new(malformed(
                        &format!("Column span incomplete in table line {}.", table_line + 1),
                        block[0].lineno,
                    )));
                }
                Ok(cols)
            };

        let is_span_line = |s: &str| {
            let t = s.trim_end();
            !t.is_empty() && t.starts_with('-') && t.chars().all(|c| matches!(c, '-' | ' '))
        };
        let first_col = columns.first().copied().unwrap_or((0, 0));
        let mut rows: Vec<RawRow> = Vec::new();
        let mut open: Option<usize> = None;
        #[allow(clippy::needless_range_loop)]
        for bi in 1..work.len() {
            let line = &work[bi];
            let at_bottom = bi == bottom;
            if is_span_line(line) || at_bottom {
                let span_cols = match parse_span_cols(&work[bi], bi) {
                    Ok(c) => c,
                    Err(m) => {
                        out.push(*m);
                        return;
                    }
                };
                if let Some(s) = open.take() {
                    rows.push(RawRow {
                        start: s,
                        end: bi,
                        cols: span_cols,
                    });
                } else if !at_bottom || rows.is_empty() {
                    // span line with no open row: empty row
                    rows.push(RawRow {
                        start: bi,
                        end: bi,
                        cols: span_cols,
                    });
                }
                continue;
            }
            let fc_text = display_slice(line, first_col.0, first_col.1);
            if !fc_text.trim().is_empty() {
                if let Some(s) = open.take() {
                    rows.push(RawRow {
                        start: s,
                        end: bi,
                        cols: columns.clone(),
                    });
                }
                open = Some(bi);
            } else if open.is_none() {
                // blank first column with no open row: dropped silently
            }
        }
        if let Some(s) = open {
            rows.push(RawRow {
                start: s,
                end: bottom,
                cols: columns.clone(),
            });
        }

        // margin check + last-column extension, per ROW using the row's own
        // colspec (span rows have merged columns — docutils check_columns).
        let mut last_col_end = border_end;
        for row in &rows {
            for bi in row.start..row.end.min(bottom) {
                let line = &work[bi];
                for w2 in row.cols.windows(2) {
                    let (_, e1) = w2[0];
                    let (s2, _) = w2[1];
                    if !display_slice(line, e1, s2).trim().is_empty() {
                        out.push(malformed(
                            &format!("Text in column margin in table line {}.", bi + 1),
                            block[bi].lineno,
                        ));
                        return;
                    }
                }
                let row_border_end = row.cols.last().map(|(_, e)| *e).unwrap_or(border_end);
                let tail = display_slice(line, row_border_end, column_width(line));
                if !tail.trim().is_empty() {
                    let last_start = row.cols.last().map(|(s, _)| *s).unwrap_or(0);
                    let extent = last_start
                        + column_width(
                            display_slice(line, last_start, column_width(line)).trim_end(),
                        );
                    last_col_end = last_col_end.max(extent);
                }
            }
        }

        // map span cols -> column indices for morecols; validate alignment
        let col_starts: Vec<usize> = columns.iter().map(|(s, _)| *s).collect();
        let col_ends: Vec<usize> = columns.iter().map(|(_, e)| *e).collect();
        let mut built_rows: Vec<(usize, Node)> = Vec::new(); // (start_line, row)
        for row in &rows {
            let mut r = Node::elem(kinds::ROW, self.span_of(lines, start, end));
            for (ci, (cs, ce)) in row.cols.iter().enumerate() {
                let ce_eff = if ci == row.cols.len() - 1 {
                    last_col_end.max(*ce)
                } else {
                    *ce
                };
                let Some(ci_start) = col_starts.iter().position(|s| s == cs) else {
                    out.push(malformed(
                        &format!(
                            "Column span alignment problem in table line {}.",
                            row.start + 2
                        ),
                        block[0].lineno,
                    ));
                    return;
                };
                let span_end_col = if ci == row.cols.len() - 1 {
                    columns.len() - 1
                } else {
                    match col_ends.iter().position(|e| e == ce) {
                        Some(p) => p,
                        None => {
                            out.push(malformed(
                                &format!(
                                    "Column span alignment problem in table line {}.",
                                    row.start + 2
                                ),
                                block[0].lineno,
                            ));
                            return;
                        }
                    }
                };
                let morecols = span_end_col - ci_start;
                let mut entry = Node::elem(kinds::ENTRY, self.span_of(lines, start, end));
                if morecols > 0 {
                    entry.set("morecols", AttrValue::Int(morecols as i64));
                }
                // cell block
                let mut cell_lines: Vec<LineRec> = Vec::new();
                #[allow(clippy::needless_range_loop)]
                for bi in row.start..row.end.min(bottom) {
                    let l = block[bi];
                    let text = self.sources.line_text(l);
                    let (s, e) = display_range(text, *cs, ce_eff.min(column_width(text)));
                    // strip trailing whitespace inside the cell view
                    let e = s + text[s..e].trim_end().len();
                    cell_lines.push(self.rewrap_range(l, s, e));
                }
                let base = cell_lines
                    .iter()
                    .filter(|l| !self.sources.line_text(**l).trim().is_empty())
                    .map(|l| l.indent())
                    .min()
                    .unwrap_or(0);
                let dedented: Vec<LineRec> = cell_lines
                    .iter()
                    .map(|l| {
                        if self.sources.line_text(*l).trim().is_empty() {
                            self.blank_at(*l)
                        } else {
                            l.dedented(base)
                        }
                    })
                    .collect();
                if dedented.iter().any(|l| !l.is_blank()) {
                    self.line_bias += 1;
                    entry.children = self.parse_nested(&dedented, "entry");
                    self.line_bias -= 1;
                }
                r.children.push(entry);
            }
            built_rows.push((row.start, r));
        }

        // widened last column affects colwidths
        let mut colwidths: Vec<usize> = columns.iter().map(|(s, e)| e - s).collect();
        if let (Some(last), Some((s, _))) = (colwidths.last_mut(), columns.last()) {
            *last = (*last).max(last_col_end.saturating_sub(*s));
        }

        let table_span = self.span_of(lines, start, end);
        let mut table = Node::elem(kinds::TABLE, table_span);
        let mut tgroup = Node::elem(kinds::TGROUP, table_span);
        tgroup.set("cols", AttrValue::Int(columns.len() as i64));
        for w in &colwidths {
            let mut cs = Node::elem(kinds::COLSPEC, table_span);
            cs.set("colwidth", AttrValue::Int(*w as i64));
            tgroup.children.push(cs);
        }
        if let Some(sep) = head_sep_row {
            let mut thead = Node::elem(kinds::THEAD, table_span);
            let mut tbody_rows = Vec::new();
            for (rs, r) in built_rows {
                if rs < sep {
                    thead.children.push(r);
                } else {
                    tbody_rows.push(r);
                }
            }
            tgroup.children.push(thead);
            let mut tbody = Node::elem(kinds::TBODY, table_span);
            tbody.children = tbody_rows;
            tgroup.children.push(tbody);
        } else {
            let mut tbody = Node::elem(kinds::TBODY, table_span);
            tbody.children = built_rows.into_iter().map(|(_, r)| r).collect();
            tgroup.children.push(tbody);
        }
        table.children.push(tgroup);
        out.push(table);

        if !blank_after_ok {
            if let Some(l) = lines.get(*pos) {
                out.push(self.msg(
                    messages::WARNING,
                    "Blank line required after table.",
                    l.source,
                    l.lineno,
                ));
            }
        }
    }

    /// `.. name:: …` directives: the docutils machinery (probe-verified;
    /// see 2026-08-13-m2-wave3-probes.md). Wave-3 registry: admonitions +
    /// generic admonition; more directives arrive in later tasks.
    fn parse_directive(
        &mut self,
        lines: &[LineRec],
        pos: &mut usize,
        name: &str,
        first_rest: &str,
        out: &mut Vec<Node>,
    ) {
        let start = *pos;
        let lineno = lines[start].lineno;
        let (block, consumed, _indent, _term) = indented_block(lines, start + 1);
        *pos = start + 1 + consumed;
        let span = self.span_of(lines, start, start + consumed);
        // Full raw source (original indentation preserved) — reproduced in
        // EVERY directive error literal. Fixture-verified: docutils'
        // block_text spans the marker through ALL trailing blank lines
        // (the final newline then disappears in line-splitting, so exactly
        // one trailing blank renders in the literal).
        let mut raw_end = start + 1 + consumed;
        while lines.get(raw_end).map(|l| l.is_blank()).unwrap_or(false) {
            raw_end += 1;
        }
        let mut rawsource = self.sources.line_text(lines[start]).to_string();
        for l in &lines[start + 1..raw_end] {
            rawsource.push('\n');
            rawsource.push_str(self.sources.line_text(*l));
        }

        let first_line = {
            let t = first_rest.trim_start_matches(' ');
            let offset = self.sources.line_text(lines[start]).len() - t.len();
            self.rewrap_from(lines[start], offset)
        };
        self.run_directive_core(
            name,
            first_line,
            &block,
            &rawsource,
            lineno,
            span,
            Vec::new(),
            out,
        );
    }

    /// The name-lookup + parse_directive_block + run dispatch shared by
    /// body-level directives and substitution-embedded ones.
    #[allow(clippy::too_many_arguments)]
    fn run_directive_core(
        &mut self,
        name: &str,
        first_line: LineRec,
        block: &[LineRec],
        rawsource: &str,
        lineno: u32,
        span: Span,
        presets: Vec<(String, OptVal)>,
        out: &mut Vec<Node>,
    ) {
        self.capture_directive_record(name, &first_line, block, lineno);
        let lower = name.to_lowercase();
        let Some(spec) = directive_spec_mode(&lower, self.sphinx) else {
            // Unknown: INFO (language-resolution narrative) + ERROR.
            out.push(self.msg(
                messages::INFO,
                &format!(
                    "No directive entry for \"{name}\" in module \"docutils.parsers.rst.languages.en\".\nTrying \"{name}\" as canonical directive name."
                ),
                span.source,
                lineno,
            ));
            out.push(messages::with_literal(
                self.msg(
                    messages::ERROR,
                    &format!("Unknown directive type \"{name}\"."),
                    span.source,
                    lineno,
                ),
                rawsource,
            ));
            return;
        };

        // MarkupError wrapper (states.py:2274-2281): uses the directive
        // name AS WRITTEN (`.. NOTE::` errors say "NOTE").
        let dir_error = |me: &Self, detail: &str| -> Node {
            messages::with_literal(
                me.msg(
                    messages::ERROR,
                    &format!("Error in \"{name}\" directive:\n{detail}."),
                    span.source,
                    lineno,
                ),
                rawsource,
            )
        };

        // ---- parse_directive_block (states.py:2301-2345), exact order ----
        // `indented` mirrors get_first_known_indented(match.end(),
        // strip_top=0): the marker-line remainder after `::` and ALL
        // following spaces, then the (already dedented) indented block.
        let mut indented: Vec<LineRec> = vec![first_line];
        indented.extend(block.iter().copied());
        // Exactly ONE leading blank line is trimmed, then all trailing.
        if indented.first().map(|l| l.is_blank()).unwrap_or(false) {
            indented.remove(0);
        }
        while indented.last().map(|l| l.is_blank()).unwrap_or(false) {
            indented.pop();
        }

        // Split arg block vs content at the first blank line — only when
        // the directive declares arguments or options.
        let declares_specs = spec.required_arguments > 0
            || spec.optional_arguments > 0
            || !spec.option_spec.is_empty();
        let mut arg_block: Vec<LineRec>;
        let mut content: Vec<LineRec>;
        let blank_idx;
        if !indented.is_empty() && declares_specs {
            blank_idx = indented
                .iter()
                .position(|l| l.is_blank())
                .unwrap_or(indented.len());
            arg_block = indented[..blank_idx].to_vec();
            content = indented
                .get(blank_idx + 1..)
                .map(|s| s.to_vec())
                .unwrap_or_default();
        } else {
            blank_idx = 0;
            arg_block = Vec::new();
            content = indented.clone();
        }

        // Options before arguments (parse_directive_options,
        // states.py:2347-2363): the arg block splits at the FIRST
        // field-marker line. Presets (the substitution alt=) seed the
        // dict and are overridden by parsed options.
        let mut options: Vec<(String, OptVal)> = presets;
        if !spec.option_spec.is_empty() {
            if let Some(k) = arg_block
                .iter()
                .position(|l| field_marker(self.sources.line_text(*l)).is_some())
            {
                let opt_block = arg_block.split_off(k);
                match parse_extension_options(&self.sources, &opt_block, spec.option_spec) {
                    Ok(opts) => {
                        for (k2, v) in opts {
                            match options.iter_mut().find(|(n, _)| *n == k2) {
                                Some(slot) => slot.1 = v,
                                None => options.push((k2, v)),
                            }
                        }
                    }
                    Err(detail) => {
                        out.push(dir_error(self, &detail));
                        return;
                    }
                }
            }
        }

        // Leftover argument lines become content for argument-less
        // directives (probe X6), re-joined with the blank separator and
        // everything after it (states.py:2330-2334).
        if !arg_block.is_empty() && spec.required_arguments == 0 && spec.optional_arguments == 0 {
            let mut rejoined = arg_block.clone();
            rejoined.extend(indented[blank_idx.min(indented.len())..].iter().copied());
            content = rejoined;
            arg_block.clear();
        }
        while content.first().map(|l| l.is_blank()).unwrap_or(false) {
            content.remove(0);
        }

        // Arguments (parse_directive_arguments, states.py:2365-2380).
        let mut arguments: Vec<String> = Vec::new();
        if spec.required_arguments + spec.optional_arguments > 0 {
            let arg_text = self.join_lines(&arg_block);
            match parse_directive_arguments(&arg_text, &spec) {
                Ok(a) => arguments = a,
                Err(detail) => {
                    out.push(dir_error(self, &detail));
                    return;
                }
            }
        }

        // The content-permission check runs LAST (states.py:2343-2344).
        if !content.is_empty() && !spec.has_content {
            out.push(dir_error(self, "no content permitted"));
            return;
        }

        let input = DirectiveInput {
            name,
            arguments,
            options,
            content,
            span,
            lineno,
            rawsource,
        };
        match spec.kind {
            DirectiveKind::Admonition(kind) => self.run_admonition(kind, input, out),
            DirectiveKind::GenericAdmonition => self.run_generic_admonition(input, out),
            DirectiveKind::Image => self.run_image(input, out),
            DirectiveKind::PseudoSection(kind) => self.run_pseudo_section(kind, input, out),
            DirectiveKind::Rubric => self.run_rubric(input, out),
            DirectiveKind::QuoteClass(class) => self.run_quote_class(class, input, out),
            DirectiveKind::Compound => self.run_compound(input, out),
            DirectiveKind::Container => self.run_container(input, out),
            DirectiveKind::ParsedLiteral => self.run_parsed_literal(input, out),
            DirectiveKind::Figure => self.run_figure(input, out),
            DirectiveKind::Code => self.run_code(input, out),
            DirectiveKind::MathBlock => self.run_math(input, out),
            DirectiveKind::Raw => self.run_raw(input, out),
            DirectiveKind::LineBlockDir => self.run_line_block(input, out),
            DirectiveKind::ClassDir => self.run_class(input, out),
            DirectiveKind::RstTable => self.run_rst_table(input, out),
            DirectiveKind::CsvTable => self.run_csv_table(input, out),
            DirectiveKind::ListTable => self.run_list_table(input, out),
            DirectiveKind::Replace => self.run_replace(input, out),
            DirectiveKind::UnicodeDir => self.run_unicode(input, out),
            DirectiveKind::DateDir => self.run_date(input, out),
            DirectiveKind::Toctree => self.run_toctree(input, out),
            DirectiveKind::VersionChange(info) => self.run_version_change(info, input, out),
            DirectiveKind::SeeAlso => self.run_seealso(input, out),
            DirectiveKind::SphinxCodeBlock => self.run_sphinx_code_block(input, out),
            DirectiveKind::Highlight => self.run_highlight(input, out),
            DirectiveKind::Only => self.run_only(input, out),
            DirectiveKind::SphinxMath => self.run_sphinx_math(input, out),
            DirectiveKind::IndexDir => self.run_index(input, out),
            DirectiveKind::HList => self.run_hlist(input, out),
            DirectiveKind::Glossary => self.run_glossary(input, out),
            DirectiveKind::ObjectDesc(kind) => {
                self.run_object_description(DescDispatch::Std(kind), input, out)
            }
            DirectiveKind::PyObjectDesc(py) => {
                self.run_object_description(DescDispatch::Py(py), input, out)
            }
            DirectiveKind::PyModule => self.run_py_module(input, out),
            DirectiveKind::PyCurrentModule => self.run_py_currentmodule(input),
            DirectiveKind::ProgramDir => self.run_program(input),
            // `DefaultDomain.run` sets `env.current_document.default_domain`
            // and returns []. This crate implements no domain whose
            // directives/roles the default would route to (the std domain is
            // always consulted last anyway), so the state has nothing to
            // steer — the node-level effect, an empty return, is all of it.
            DirectiveKind::DefaultDomainDir => {}
            #[cfg(test)]
            DirectiveKind::TestSplice => {
                let outcome = self.run_test_splice(input);
                self.finish_directive(outcome);
            }
        }
    }

    /// Bank a directive's [`DirectiveOutcome`] for the enclosing
    /// block-parse loop: a splice waits in `pending_splice` until the loop
    /// reaches its cursor. Splice-producing directive arms (T12's include)
    /// route their return value through here.
    #[allow(dead_code)] // see DirectiveOutcome
    fn finish_directive(&mut self, outcome: DirectiveOutcome) {
        if let DirectiveOutcome::Splice(request) = outcome {
            self.pending_splice = Some(request);
        }
    }

    /// The test-only splice producer: content lines become a spliced
    /// source named by the argument. Built from the input alone — the
    /// shape T12's include takes: no parser internals needed to return a
    /// splice.
    #[cfg(test)]
    fn run_test_splice(&mut self, input: DirectiveInput<'_>) -> DirectiveOutcome {
        let lines = input
            .content
            .iter()
            .map(|l| self.sources.line_text(*l).to_string())
            .collect();
        DirectiveOutcome::Splice(SpliceRequest {
            lines,
            source_path: input.arguments[0].clone(),
            base_lineno_override: None,
        })
    }

    /// `.. program::` (`domains/std/__init__.py:333-348`): pure
    /// `env.ref_context` state, no nodes. The literal argument `None` pops
    /// the scope rather than naming a program called "None".
    fn run_program(&mut self, input: DirectiveInput<'_>) {
        let Some(argument) = input.arguments.first() else {
            return;
        };
        let program = ws_collapse(argument.trim(), "-");
        if program == "None" {
            self.program = None;
        } else {
            self.program = Some(program);
        }
    }

    /// sphinx math (patches.py MathDirective + math-domain numbering).
    /// Absent label/number are Python None -> pformat "True".
    fn run_sphinx_math(&mut self, input: DirectiveInput<'_>, out: &mut Vec<Node>) {
        let mut latex = self.join_lines(&input.content);
        if let Some(arg) = input.arguments.first() {
            latex = if latex.is_empty() {
                format!("{arg}\n\n")
            } else {
                format!("{arg}\n\n{latex}")
            };
        }
        let label =
            match opt_get(&input.options, "label").or_else(|| opt_get(&input.options, "name")) {
                Some(OptVal::Str(s)) if !s.is_empty() => Some(s.clone()),
                _ => None,
            };
        let nowrap = opt_get(&input.options, "nowrap").is_some()
            || opt_get(&input.options, "no-wrap").is_some();
        let mut node = Node::elem("math_block", input.span);
        if let Some(OptVal::StrList(classes)) = opt_get(&input.options, "class") {
            node.attrs.classes.extend(classes.iter().cloned());
        }
        node.set("docname", AttrValue::Str(self.docname.clone()));
        node.set("no-wrap", AttrValue::Int(i64::from(nowrap)));
        node.set("nowrap", AttrValue::Int(i64::from(nowrap)));
        node.set("xml:space", AttrValue::Str("preserve".to_string()));
        node.children.push(Node::text_node(latex, input.span));
        match label {
            Some(label) => {
                self.equation_serial += 1;
                let id = ids::make_id(&format!("equation-{label}"));
                node.attrs.ids.push(id.clone());
                node.set("label", AttrValue::Str(label));
                node.set("number", AttrValue::Int(i64::from(self.equation_serial)));
                let mut target = Node::elem(kinds::TARGET, input.span);
                target.set("refid", AttrValue::Str(id));
                out.push(target);
                out.push(node);
            }
            None => {
                node.set("label", AttrValue::Str("True".to_string()));
                node.set("number", AttrValue::Str("True".to_string()));
                out.push(node);
            }
        }
    }

    /// sphinx index directive (sphinx/domains/index.py IndexDirective).
    fn run_index(&mut self, input: DirectiveInput<'_>, out: &mut Vec<Node>) {
        let target_id = format!("index-{}", self.registry.new_index_serialno());
        let mut entries: Vec<String> = Vec::new();
        for line in input.arguments[0].split('\n') {
            let line = line.trim();
            if line.is_empty() {
                continue;
            }
            entries.extend(process_index_entry(line, &target_id));
        }
        let mut index = Node::elem("index", input.span);
        index.set("entries", AttrValue::List(entries));
        index.set("inline", AttrValue::Int(0));
        let mut target = Node::elem(kinds::TARGET, input.span);
        match opt_get(&input.options, "name") {
            Some(OptVal::Str(n)) => {
                target.attrs.names.push(ids::fully_normalize_name(n));
            }
            _ => target.attrs.ids.push(target_id),
        }
        out.push(index);
        out.push(target);
    }

    /// sphinx hlist (other.py HList): content must be exactly one bullet
    /// list; distributed into ncolumns hlistcol children.
    fn run_hlist(&mut self, input: DirectiveInput<'_>, out: &mut Vec<Node>) {
        let ncolumns = match opt_get(&input.options, "columns") {
            Some(OptVal::Int(n)) if *n > 0 => *n as usize,
            _ => 2,
        };
        let children = self.parse_nested(&input.content, "element");
        let one_list = children.len() == 1 && children[0].kind == kinds::BULLET_LIST;
        if !one_list {
            // logger.warning('.. hlist content is not a list') goes to the
            // log stream, not the tree.
            return;
        }
        let list = children.into_iter().next().expect("length checked");
        let items = list.children;
        let npercol = items.len() / ncolumns;
        let nmore = items.len() % ncolumns;
        let mut hlist = Node::elem("hlist", input.span);
        hlist.set("ncolumns", AttrValue::Str(ncolumns.to_string()));
        let mut it = items.into_iter();
        for col in 0..ncolumns {
            let take = npercol + usize::from(col < nmore);
            let mut bl = Node::elem(kinds::BULLET_LIST, input.span);
            for _ in 0..take {
                match it.next() {
                    Some(item) => bl.children.push(item),
                    None => break,
                }
            }
            let mut colnode = Node::elem("hlistcol", input.span);
            colnode.children.push(bl);
            hlist.children.push(colnode);
        }
        out.push(hlist);
    }

    /// sphinx glossary (std domain): term lines + indented definitions;
    /// each term gets a term-<id> target and an embedded index entry.
    ///
    /// Comments are honoured: `Glossary.run` treats an unindented `.. `
    /// line as a comment rather than a term
    /// (`domains/std/__init__.py:452-455`) and swallows its indented
    /// continuation lines with it (`:493-494`, `elif in_comment: pass`).
    /// Sphinx's explicit `in_comment` flag has no counterpart here because
    /// this loop already skips every indented line it meets outside a
    /// definition block, whether or not a comment preceded it.
    fn run_glossary(&mut self, input: DirectiveInput<'_>, out: &mut Vec<Node>) {
        let mut glossary = Node::elem("glossary", input.span);
        glossary.set(
            "sorted",
            AttrValue::Int(i64::from(opt_get(&input.options, "sorted").is_some())),
        );
        let mut dl = Node::elem(kinds::DEFINITION_LIST, input.span);
        dl.attrs.classes.push("glossary".to_string());
        // Entry split: unindented term line(s) followed by an indented
        // definition block (misformat warnings go to the log, not the
        // tree; the corpus pins well-formed input).
        let mut i = 0usize;
        let content = &input.content;
        while i < content.len() {
            if content[i].is_blank() {
                i += 1;
                continue;
            }
            if content[i].indent() > 0 {
                // A comment's continuation lines, or a stray indented line
                // without a term (log-warned in Sphinx); skipped either way.
                i += 1;
                continue;
            }
            if is_glossary_comment(&content[i], self.sources.line_text(content[i])) {
                i += 1;
                continue;
            }
            let mut term_lines: Vec<LineRec> = Vec::new();
            while i < content.len()
                && !content[i].is_blank()
                && content[i].indent() == 0
                && !is_glossary_comment(&content[i], self.sources.line_text(content[i]))
            {
                term_lines.push(content[i]);
                i += 1;
            }
            let mut def_lines: Vec<LineRec> = Vec::new();
            while i < content.len() && (content[i].is_blank() || content[i].indent() > 0) {
                if content[i].is_blank()
                    && content
                        .get(i + 1)
                        .map(|l| !l.is_blank() && l.indent() == 0)
                        .unwrap_or(true)
                {
                    i += 1;
                    break;
                }
                def_lines.push(content[i]);
                i += 1;
            }
            while def_lines.last().map(|l| l.is_blank()).unwrap_or(false) {
                def_lines.pop();
            }
            let mut item = Node::elem(kinds::DEFINITION_LIST_ITEM, input.span);
            let mut term_messages: Vec<Node> = Vec::new();
            for tl in &term_lines {
                let raw_term = self.sources.line_text(*tl).trim().to_string();
                let raw_term = raw_term.as_str();
                // split_term_classifiers: ' +: +' — first classifier is
                // the index key.
                let mut parts = raw_term.splitn(2, " : ");
                let term_text = parts.next().unwrap_or(raw_term).trim_end().to_string();
                let index_key = parts
                    .next()
                    .map(|c| c.split(" : ").next().unwrap_or(c).trim().to_string());
                // Sphinx's `make_glossary_term` stamps the term node with
                // the *term line's* own source info, not the directive's
                // (`domains/std/__init__.py:386-388`), and the index node it
                // appends inherits it. Everything that reports a term's
                // location — the duplicate-object warning, above all — reads
                // that, so each term carries its own span here.
                let term_span = Span {
                    source: tl.source,
                    line: tl.lineno,
                    start: tl.start,
                    end: tl.end,
                };
                let inline = self.inline(&term_text, term_span, tl.lineno);
                let mut term = Node::elem(kinds::TERM, term_span);
                term.children = inline.nodes;
                term_messages.extend(inline.messages);
                // `termtext = term.astext()` (`domains/std:389`), taken from
                // the PARSED term and before the index node is appended: a
                // term written with markup registers, indexes and ids itself
                // under its rendered text, not its source text.
                let term_text = term.astext();
                // `make_glossary_term` (`domains/std:375-407`):
                // `make_id(env, document, 'term', termtext)` — sphinx's
                // case-preserving `_make_id` fork, with a `term`-keyed
                // serial fallback of its own, then `note_explicit_target`.
                let node_id = self.registry.sphinx_make_id("term", &term_text);
                self.registry.note_explicit_id(&node_id);
                term.attrs.ids.push(node_id.clone());
                let mut index = Node::elem("index", term_span);
                index.set(
                    "entries",
                    AttrValue::List(vec![index_entry_tuple(
                        "single",
                        &term_text,
                        &node_id,
                        "main",
                        index_key.as_deref(),
                    )]),
                );
                term.children.push(index);
                item.children.push(term);
            }
            item.children.extend(term_messages);
            let dedented = dedent_by_min(&def_lines);
            let mut definition = Node::elem(kinds::DEFINITION, input.span);
            definition.children = self.parse_nested(&dedented, "definition");
            item.children.push(definition);
            dl.children.push(item);
        }
        glossary.children.push(dl);
        out.push(glossary);
    }

    /// sphinx `ObjectDescription.run` (`directives/__init__.py:183-314`):
    /// the `index` + `desc` anatomy every object-describing directive
    /// shares, with each subclass's `handle_signature` /
    /// `add_target_and_index` / `transform_content` inlined by
    /// [`ObjectDescKind`].
    fn run_object_description(
        &mut self,
        kind: DescDispatch,
        input: DirectiveInput<'_>,
        out: &mut Vec<Node>,
    ) {
        let Some(argument) = input.arguments.first() else {
            return;
        };
        // `self.name` is the directive name as written for the bare
        // docutils registration, but `'{domain}:{name}'` for a domain
        // directive (`Domain.directive`'s adapter, `domains/__init__.py`),
        // which is why `describe` reports `domain=""` and `option` reports
        // `domain="std"` / `objtype="option"`. For the aliasing py
        // directives, `run()` rewrites `self.name` BEFORE the base run
        // partitions it, so the objtype is the ALIASED kind's (trap 13).
        let (domain, objtype) = match kind {
            DescDispatch::Std(ObjectDescKind::Describe) => ("", input.name.to_string()),
            DescDispatch::Std(_) => ("std", input.name.to_lowercase()),
            DescDispatch::Py(py) => ("py", py.kind.objtype().to_string()),
        };
        let span = input.span;

        // Deprecated-alias merge (`:226-241`): the old spelling feeds the
        // new one, and BOTH attributes end up carrying the merged value.
        let has = |name: &'static str| opt_get(&input.options, name).is_some();
        let no_index = has("no-index") || has("noindex");
        let no_index_entry = has("no-index-entry") || has("noindexentry");
        let no_contents_entry = has("no-contents-entry") || has("nocontentsentry");
        let no_typesetting = has("no-typesetting");

        let mut desc = Node::elem("desc", span);
        desc.set("domain", AttrValue::Str(domain.to_string()));
        desc.set("objtype", AttrValue::Str(objtype.clone()));
        // 'desctype' is sphinx's backwards-compatible alias of 'objtype'.
        desc.set("desctype", AttrValue::Str(objtype.clone()));
        desc.set("no-index", AttrValue::Int(i64::from(no_index)));
        desc.set("noindex", AttrValue::Int(i64::from(no_index)));
        desc.set("no-index-entry", AttrValue::Int(i64::from(no_index_entry)));
        desc.set("noindexentry", AttrValue::Int(i64::from(no_index_entry)));
        desc.set(
            "no-contents-entry",
            AttrValue::Int(i64::from(no_contents_entry)),
        );
        desc.set(
            "nocontentsentry",
            AttrValue::Int(i64::from(no_contents_entry)),
        );
        desc.set("no-typesetting", AttrValue::Int(i64::from(no_typesetting)));
        if !domain.is_empty() {
            desc.attrs.classes.push(domain.to_string());
        }
        desc.attrs.classes.push(objtype.clone());

        let mut index_entries: Vec<String> = Vec::new();
        // Names are the `(fullname, name_prefix)` tuples py's
        // `handle_signature` returns (`_object.py:397`); std kinds carry an
        // empty prefix. Dedup is on the whole tuple, exactly like the base
        // run's `if name not in self.names` (`directives/__init__.py:273`).
        let mut names: Vec<(String, String)> = Vec::new();
        for sig in object_signatures(argument, self.py.strip_signature_backslash) {
            let mut signode = Node::elem("desc_signature", span);
            signode
                .attrs
                .classes
                .extend(["sig".to_string(), "sig-object".to_string()]);
            let name = self.handle_object_signature(kind, &sig, &input, &mut signode);
            if let DescDispatch::Std(std_kind) = kind {
                // `_toc_parts`/`_toc_name` are assigned in a `finally`
                // (`:264-272`), so the ValueError path carries them too.
                // Only `ConfigurationValue` overrides the two empty
                // defaults; the py arm stamps its own inside
                // `handle_py_signature`.
                let (toc_parts, toc_name) = match (std_kind, &name) {
                    (ObjectDescKind::Confval, Some((n, _))) => {
                        (format!("({},)", py_repr(Some(n))), n.clone())
                    }
                    _ => ("()".to_string(), String::new()),
                };
                signode.set("_toc_parts", AttrValue::Str(toc_parts));
                signode.set("_toc_name", AttrValue::Str(toc_name));
            }
            // "only add target and index entry if this is the first
            // description of the object with this name in this desc block".
            if let Some(name) = name {
                if !names.contains(&name) {
                    names.push(name.clone());
                    if !no_index {
                        self.object_target_and_index(
                            kind,
                            &objtype,
                            &name,
                            &input,
                            &mut signode,
                            &mut index_entries,
                        );
                    }
                }
            }
            desc.children.push(signode);
        }

        // py `before_content` (`_object.py:449-480`): the class/module
        // ref_context pushes the nested content parses under.
        if let DescDispatch::Py(py) = kind {
            self.py_before_content(py, &names, &input);
        }
        let mut content = Node::elem("desc_content", span);
        content.children = self.parse_nested(&input.content, "desc_content");
        if kind == DescDispatch::Std(ObjectDescKind::Confval) {
            self.confval_transform_content(&input, &mut content);
        }
        // Base-run tail order (`directives/__init__.py`): the
        // `object-description-transform` event fires FIRST (its only
        // handler, `filter_meta_fields`, guards `domain == 'py'` —
        // `domains/python/__init__.py:610-611` — so a std `:meta:` field
        // survives and renders renamed), then `DocFieldTransformer`
        // rewrites the doc fields UNCONDITIONALLY for every object
        // description (`directives/__init__.py:295`) — std kinds get an
        // empty typemap, so their fields all take the unknown
        // rename-and-pass-through branch — then `after_content` pops the
        // ref_context the py field xrefs just read.
        match kind {
            DescDispatch::Py(py) => {
                filter_meta_fields(&mut content);
                self.transform_doc_fields(&mut content, py_field_type_map);
                self.py_after_content(py, &input);
            }
            DescDispatch::Std(_) => {
                self.transform_doc_fields(&mut content, std_field_type_map);
            }
        }
        desc.children.push(content);

        let mut index = Node::elem("index", span);
        index.set("entries", AttrValue::List(index_entries));
        out.push(index);

        if no_typesetting {
            // `:299-313`: the description is replaced by a bare target
            // carrying every id it and its children had — and dropped
            // entirely when there are none (docutils rejects an id-less
            // target).
            let mut ids = Vec::new();
            collect_element_ids(&desc, &mut ids);
            if !ids.is_empty() {
                let mut target = Node::elem(kinds::TARGET, span);
                target.attrs.ids = ids;
                out.push(target);
            }
            return;
        }
        out.push(desc);
    }

    /// The per-subclass `handle_signature`. Returns the object name — a
    /// `(name, prefix)` tuple, prefix empty for std kinds — or `None` for
    /// the ValueError path, where `run` clears the signature node and drops
    /// the whole signature into one `desc_name` (`:259-263`), which each
    /// arm does itself.
    fn handle_object_signature(
        &mut self,
        kind: DescDispatch,
        sig: &str,
        input: &DirectiveInput<'_>,
        signode: &mut Node,
    ) -> Option<(String, String)> {
        let span = signode.span;
        let std_name = |name: String| Some((name, String::new()));
        match kind {
            // The base `handle_signature` raises unconditionally (`:100-111`).
            DescDispatch::Std(ObjectDescKind::Describe) => {
                signode.children.clear();
                signode.children.push(desc_name_node(sig, span));
                None
            }
            // `GenericObject.handle_signature` (`domains/std:56-64`).
            DescDispatch::Std(ObjectDescKind::EnvVar) => {
                signode.children.clear();
                signode.children.push(desc_name_node(sig, span));
                std_name(ws_collapse(sig, " "))
            }
            // `ConfigurationValue.handle_signature` (`domains/std:126-131`).
            DescDispatch::Std(ObjectDescKind::Confval) => {
                signode.children.clear();
                signode.children.push(desc_name_node(sig, span));
                let name = ws_collapse(sig, " ");
                signode.set("fullname", AttrValue::Str(name.clone()));
                std_name(name)
            }
            DescDispatch::Std(ObjectDescKind::Cmdoption) => self
                .handle_option_signature(sig, input.lineno, signode)
                .and_then(std_name),
            DescDispatch::Py(py) => self.handle_py_signature(py, sig, input, signode),
        }
    }

    /// `Cmdoption.handle_signature` (`domains/std/__init__.py:229-290`) with
    /// `option_emphasise_placeholders` at its default False, which is the
    /// plain `desc_name` + `desc_addname` pair per spelling.
    fn handle_option_signature(
        &mut self,
        sig: &str,
        lineno: u32,
        signode: &mut Node,
    ) -> Option<String> {
        let span = signode.span;
        let mut firstname: Option<String> = None;
        let mut allnames: Vec<String> = Vec::new();
        for potential in sig.split(", ") {
            let potential = potential.trim();
            let Some((optname, args)) = option_desc_match(potential) else {
                // This diagnostic goes to the logger, not the tree
                // (`domains/std/__init__.py:237-245`), located on the
                // signature node — which carries the directive's own line.
                // The spelling contributes nothing either way.
                self.log_warnings.push(super::ParseLogWarning {
                    source: span.source,
                    message: format!(
                        "Malformed option description {}, should look like \"opt\", \
                         \"-opt args\", \"--opt args\", \"/opt args\" or \"+opt args\"",
                        py_repr(Some(potential))
                    ),
                    line: lineno,
                });
                continue;
            };
            // "optional value surrounded by brackets (ex. foo[=bar])".
            // Sphinx tests `args[-1] == ']'` unguarded, so `.. option:: foo[`
            // raises IndexError out of the whole parse; leaving the
            // signature unchanged is the hardening deviation (a crash is not
            // a contract, and there is no tree to be byte-identical to).
            let (optname, args) = match (optname.strip_suffix('['), args.strip_suffix(']')) {
                (Some(trimmed), Some(_)) => (trimmed.to_string(), format!("[{args}")),
                _ => (optname, args),
            };
            if firstname.is_some() {
                signode.children.push(desc_addname_node(", ", span));
            }
            signode.children.push(desc_name_node(&optname, span));
            signode.children.push(desc_addname_node(&args, span));
            firstname.get_or_insert_with(|| optname.clone());
            allnames.push(optname);
        }
        let firstname = match firstname {
            Some(name) => name,
            None => {
                signode.children.clear();
                signode.children.push(desc_name_node(sig, span));
                return None;
            }
        };
        signode.set("allnames", AttrValue::List(allnames));
        Some(firstname)
    }

    /// The per-subclass `add_target_and_index`: node ids through sphinx's
    /// `make_id`, the index entries, and the domain registration records
    /// the env layer replays.
    fn object_target_and_index(
        &mut self,
        kind: DescDispatch,
        objtype: &str,
        name_cls: &(String, String),
        input: &DirectiveInput<'_>,
        signode: &mut Node,
        entries: &mut Vec<String>,
    ) {
        let name: &str = &name_cls.0;
        let line = input.lineno;
        match kind {
            DescDispatch::Py(py) => {
                self.py_target_and_index(py, objtype, name_cls, input, signode, entries)
            }
            // `ObjectDescription.add_target_and_index` is `pass` (`:113-120`)
            // — no id, no index entry, no std object. (Unreachable in
            // practice: `Describe`'s handle_signature never returns a name.)
            DescDispatch::Std(ObjectDescKind::Describe) => {}
            // `GenericObject.add_target_and_index` (`domains/std:66-84`).
            // `EnvVar.indextemplate` has no ':' separator, so the whole
            // template is a 'single' entry value.
            DescDispatch::Std(ObjectDescKind::EnvVar) => {
                let node_id = self.note_object_id(objtype, name, line, signode);
                entries.push(index_entry_tuple(
                    "single",
                    &format!("environment variable; {name}"),
                    &node_id,
                    "",
                    None,
                ));
            }
            // `ConfigurationValue.add_target_and_index` (`domains/std:142-151`).
            DescDispatch::Std(ObjectDescKind::Confval) => {
                let node_id = self.note_object_id(objtype, name, line, signode);
                entries.push(index_entry_tuple(
                    "pair",
                    &format!("{name}; configuration value"),
                    &node_id,
                    "",
                    None,
                ));
            }
            // `Cmdoption.add_target_and_index` (`domains/std:292-330`).
            DescDispatch::Std(ObjectDescKind::Cmdoption) => {
                let program = self.program.clone();
                let allnames = match signode.get("allnames") {
                    Some(AttrValue::List(names)) => names.clone(),
                    _ => Vec::new(),
                };
                for optname in &allnames {
                    let mut prefix = String::from("cmdoption");
                    if let Some(program) = &program {
                        prefix.push('-');
                        prefix.push_str(program);
                    }
                    if !optname.starts_with(['-', '/']) {
                        prefix.push_str("-arg");
                    }
                    let node_id = self.registry.sphinx_make_id(&prefix, optname);
                    signode.attrs.ids.push(node_id);
                }
                // `note_explicit_target` runs once, AFTER every id is
                // chosen, so the ids of one signature never see each other
                // in `document.ids`.
                for node_id in signode.attrs.ids.clone() {
                    self.registry.note_explicit_id(&node_id);
                }
                // Every spelling registers against `signode['ids'][0]`.
                let first_id = signode.attrs.ids.first().cloned().unwrap_or_default();
                for optname in &allnames {
                    self.program_option_records
                        .push(super::ProgramOptionRecord {
                            source: signode.span.source,
                            program: program.clone(),
                            name: optname.clone(),
                            node_id: first_id.clone(),
                        });
                }
                let descr = match &program {
                    Some(program) => format!("{program} command line option"),
                    None => "command line option".to_string(),
                };
                for optname in &allnames {
                    entries.push(index_entry_tuple(
                        "pair",
                        &format!("{descr}; {optname}"),
                        &first_id,
                        "",
                        None,
                    ));
                }
            }
        }
    }

    /// The `make_id` + `note_explicit_target` + `note_object` trio the
    /// single-id `add_target_and_index` implementations share.
    fn note_object_id(
        &mut self,
        objtype: &str,
        name: &str,
        line: u32,
        signode: &mut Node,
    ) -> String {
        // Both callers pass `self.objtype` as the make_id prefix.
        let node_id = self.registry.sphinx_make_id(objtype, name);
        signode.attrs.ids.push(node_id.clone());
        self.registry.note_explicit_id(&node_id);
        self.std_object_records.push(super::ObjectRegistration {
            source: signode.span.source,
            objtype: objtype.to_string(),
            name: name.to_string(),
            node_id: node_id.clone(),
            line,
        });
        node_id
    }

    /// `PyObject.handle_signature` with every subclass override inlined
    /// (`domains/python/_object.py:248-397`, `__init__.py`, [PY §1.3]).
    /// Returns `(fullname, name_prefix)`, or `None` for the no-match
    /// ValueError path — silent, whole sig in one `desc_name`, empty toc
    /// attrs, no registration (trap 7).
    fn handle_py_signature(
        &mut self,
        py: PyDirective,
        sig: &str,
        input: &DirectiveInput<'_>,
        signode: &mut Node,
    ) -> Option<(String, String)> {
        let span = signode.span;
        let Some(m) = py_sig_match(sig) else {
            signode.children.clear();
            signode.children.push(desc_name_node(sig, span));
            signode.set("_toc_parts", AttrValue::Str("()".to_string()));
            signode.set("_toc_name", AttrValue::Str(String::new()));
            return None;
        };

        // Python-truthy option access: a present-but-empty value is falsy
        // everywhere handle_signature consults these.
        let opt_str = |name: &'static str| match opt_get(&input.options, name) {
            Some(OptVal::Str(s)) => Some(s.clone()),
            _ => None,
        };
        let opt_truthy = |name: &'static str| opt_str(name).filter(|s| !s.is_empty());

        // `modname = self.options.get('module', ref_context['py:module'])`
        // (`_object.py:263`): option PRESENCE wins, even with an empty value.
        let modname: Option<String> = match opt_str("module") {
            Some(module) => Some(module),
            None => self.py_module.clone(),
        };
        let ref_class = self.py_class.clone().filter(|c| !c.is_empty());

        // Module/class resolution (`_object.py:262-285`).
        let mut prefix = m.prefix.clone().unwrap_or_default();
        let name = m.name.clone();
        let fullname: String;
        let classname_attr: String;
        let add_module: bool;
        match &ref_class {
            Some(classname) => {
                add_module = false;
                if !prefix.is_empty()
                    && (prefix == *classname || prefix.starts_with(&format!("{classname}.")))
                {
                    // Class name given again in the signature: stripped
                    // from display, kept in the fullname.
                    fullname = format!("{prefix}{name}");
                    prefix = prefix[classname.len()..]
                        .trim_start_matches('.')
                        .to_string();
                } else if !prefix.is_empty() {
                    // A DIFFERENT prefix inside a class nests under it:
                    // `D.meth` inside `C` → `C.D.meth` (probe
                    // method_other_prefix).
                    fullname = format!("{classname}.{prefix}{name}");
                } else {
                    fullname = format!("{classname}.{name}");
                }
                classname_attr = classname.clone();
            }
            None => {
                add_module = true;
                if !prefix.is_empty() {
                    // A dotted prefix at top level becomes the signature
                    // CLASS name, not a module (trap 12).
                    classname_attr = prefix.trim_end_matches('.').to_string();
                    fullname = format!("{prefix}{name}");
                } else {
                    classname_attr = String::new();
                    fullname = name.clone();
                }
            }
        }

        // Stamped on every successful signature (`_object.py:287-289`);
        // a None modname pformats as the `"True"` sentinel (trap 2).
        signode.set(
            "module",
            AttrValue::Str(modname.clone().unwrap_or_else(|| "True".to_string())),
        );
        signode.set("class", AttrValue::Str(classname_attr.clone()));
        signode.set("fullname", AttrValue::Str(fullname.clone()));

        let single_line = crate::py::arglist::SingleLineOpts {
            parameter_list: opt_get(&input.options, "single-line-parameter-list").is_some(),
            type_parameter_list: opt_get(&input.options, "single-line-type-parameter-list")
                .is_some(),
        };
        let (multi_line_params, multi_line_tp) =
            crate::py::arglist::multi_line_flags(sig, &m, single_line, &self.py);

        // Annotation xrefs read the RAW ref_context, not the option-
        // modified modname: the `:module:` option only touches
        // `env.ref_context` in before_content (`_annotations.py:62-66`).
        let ctx = crate::py::annotations::PyRefContext {
            module: self.py_module.clone(),
            class_: self.py_class.clone(),
        };

        // 1. Signature prefix keywords (`get_signature_prefix`).
        let prefix_nodes = py_signature_prefix(py, input);
        if !prefix_nodes.is_empty() {
            let mut anno = desc_annotation_node(span);
            anno.children = prefix_nodes;
            signode.children.push(anno);
        }

        // 2. Written prefix, else `{modname}.` under add_module_names
        // (`_object.py:326-330`).
        if !prefix.is_empty() {
            signode.children.push(desc_addname_node(&prefix, span));
        } else if let Some(modname) = modname.as_deref().filter(|s| !s.is_empty()) {
            if add_module && self.py.add_module_names {
                signode
                    .children
                    .push(desc_addname_node(&format!("{modname}."), span));
            }
        }

        // 3. Object name.
        signode.children.push(desc_name_node(&name, span));

        // 4. Type parameter list; any failure is a WARNING (`_object.py:
        // 342-345`), interpolating the exception text (probes
        // tp_list_warning / tp_list_tokerror).
        if let Some(tp_list) = m.tp_list.as_deref().filter(|t| !t.is_empty()) {
            match crate::py::arglist::parse_type_list(tp_list, multi_line_tp, &ctx, &self.py) {
                Ok(node) => signode.children.push(node),
                Err(err) => self.log_warnings.push(super::ParseLogWarning {
                    source: span.source,
                    message: format!(
                        "could not parse tp_list ({}): {err}",
                        py_repr(Some(tp_list))
                    ),
                    line: input.lineno,
                }),
            }
        }

        // 5. Parameter list. An EMPTY written `()` is falsy and routes to
        // the needs_arglist branch, exactly like no parens at all (see
        // arglist_empty_still_carries_attrs in src/py/arglist.rs); the
        // bare paramlist carries NO multi_line attrs (trap 1).
        match m.arglist.as_deref().filter(|a| !a.is_empty()) {
            Some(arglist) => {
                match crate::py::arglist::parse_arglist(arglist, multi_line_params, &ctx, &self.py)
                {
                    Ok(node) => signode.children.push(node),
                    Err(crate::py::arglist::SigParseError::Syntax(_)) => {
                        // `logger.debug` — invisible (`_object.py:355-369`).
                        signode
                            .children
                            .push(crate::py::arglist::pseudo_parse_arglist(
                                arglist,
                                multi_line_params,
                                &ctx,
                                &self.py,
                            ));
                    }
                    Err(err) => {
                        // Duplicate parameter names: WARNING + pseudo
                        // fallback (`_object.py:370-381`, probe
                        // arglist_dup_warning).
                        self.log_warnings.push(super::ParseLogWarning {
                            source: span.source,
                            message: format!(
                                "could not parse arglist ({}): {err}",
                                py_repr(Some(arglist))
                            ),
                            line: input.lineno,
                        });
                        signode
                            .children
                            .push(crate::py::arglist::pseudo_parse_arglist(
                                arglist,
                                multi_line_params,
                                &ctx,
                                &self.py,
                            ));
                    }
                }
            }
            None => {
                if py.needs_arglist() {
                    let mut params = Node::elem("desc_parameterlist", span);
                    params.set("xml:space", AttrValue::Str("preserve".to_string()));
                    signode.children.push(params);
                }
            }
        }

        // 6. Return annotation (`_object.py:387-389`).
        if let Some(retann) = m.retann.as_deref().filter(|r| !r.is_empty()) {
            let mut returns = Node::elem("desc_returns", span);
            returns.set("xml:space", AttrValue::Str("preserve".to_string()));
            returns.children = crate::py::annotations::parse_annotation(retann, &ctx, &self.py);
            signode.children.push(returns);
        }

        // 7. `:annotation:` option tail (`_object.py:391-395`).
        if let Some(anno) = opt_truthy("annotation") {
            let mut node = desc_annotation_node(span);
            node.children.push(crate::py::annotations::desc_sig_space());
            node.children.push(Node::text_node(anno, span));
            signode.children.push(node);
        }

        // Subclass tails run AFTER the base handle_signature returns:
        // `:type:`/`:value:` for data/attribute (the `:`/`=` here are
        // desc_sig_punctuation, unlike parameter defaults — trap 3),
        // `:type:` only for property, display-only `:canonical:` for
        // py:type.
        match py.kind {
            PyObjectKind::Data | PyObjectKind::Attribute => {
                if let Some(typ) = opt_truthy("type") {
                    let mut node = desc_annotation_node(span);
                    node.children
                        .push(crate::py::annotations::desc_sig_punctuation(":"));
                    node.children.push(crate::py::annotations::desc_sig_space());
                    node.children
                        .extend(crate::py::annotations::parse_annotation(
                            &typ, &ctx, &self.py,
                        ));
                    signode.children.push(node);
                }
                if let Some(value) = opt_truthy("value") {
                    let mut node = desc_annotation_node(span);
                    node.children.push(crate::py::annotations::desc_sig_space());
                    node.children
                        .push(crate::py::annotations::desc_sig_punctuation("="));
                    node.children.push(crate::py::annotations::desc_sig_space());
                    node.children.push(Node::text_node(value, span));
                    signode.children.push(node);
                }
            }
            PyObjectKind::Property => {
                if let Some(typ) = opt_truthy("type") {
                    let mut node = desc_annotation_node(span);
                    node.children
                        .push(crate::py::annotations::desc_sig_punctuation(":"));
                    node.children.push(crate::py::annotations::desc_sig_space());
                    node.children
                        .extend(crate::py::annotations::parse_annotation(
                            &typ, &ctx, &self.py,
                        ));
                    signode.children.push(node);
                }
            }
            PyObjectKind::TypeAlias => {
                if let Some(canonical) = opt_truthy("canonical") {
                    let mut node = desc_annotation_node(span);
                    node.children.push(crate::py::annotations::desc_sig_space());
                    node.children
                        .push(crate::py::annotations::desc_sig_punctuation("="));
                    node.children.push(crate::py::annotations::desc_sig_space());
                    node.children
                        .extend(crate::py::annotations::parse_annotation(
                            &canonical, &ctx, &self.py,
                        ));
                    signode.children.push(node);
                }
            }
            _ => {}
        }

        // Decorators insert the `@` addname FIRST, after everything else
        // ran (`__init__.py:124-127`, `313-316`).
        if py.decorator {
            signode.children.insert(0, desc_addname_node("@", span));
        }

        // `_toc_parts`/`_toc_name` — `_object_hierarchy_parts` +
        // `_toc_entry_name` (`_object.py:399-408`, `505-522`), gated on
        // `toc_object_entries` by the base run's finally (`:264-272`).
        if self.py.toc_object_entries {
            let mut parts: Vec<String> = Vec::new();
            if let Some(modname) = modname.as_deref().filter(|s| !s.is_empty()) {
                parts.push(modname.to_string());
            }
            parts.extend(fullname.split('.').map(str::to_string));
            let toc_name = py_toc_entry_name(&parts, &fullname, py.kind, &self.py);
            signode.set("_toc_parts", AttrValue::Str(py_tuple_repr(&parts)));
            signode.set("_toc_name", AttrValue::Str(toc_name));
        } else {
            signode.set("_toc_parts", AttrValue::Str("()".to_string()));
            signode.set("_toc_name", AttrValue::Str(String::new()));
        }

        Some((fullname, prefix))
    }

    /// `PyObject.add_target_and_index` + the PyFunction extension
    /// (`_object.py:415-447`, `__init__.py:95-109`, [PY §1.4/1.5]).
    fn py_target_and_index(
        &mut self,
        py: PyDirective,
        objtype: &str,
        name_cls: &(String, String),
        input: &DirectiveInput<'_>,
        signode: &mut Node,
        entries: &mut Vec<String>,
    ) {
        let opt_str = |name: &'static str| match opt_get(&input.options, name) {
            Some(OptVal::Str(s)) => Some(s.clone()),
            _ => None,
        };
        let modname: Option<String> = match opt_str("module") {
            Some(module) => Some(module),
            None => self.py_module.clone(),
        };
        let modname = modname.filter(|m| !m.is_empty());
        let name = &name_cls.0;
        let fullname = match &modname {
            Some(modname) => format!("{modname}.{name}"),
            None => name.clone(),
        };
        // Empty prefix: the id IS the fullname, `id{n}` on collision
        // ([PY §1.5], the empty-prefix branch of `sphinx_make_id`).
        let node_id = self.registry.sphinx_make_id("", &fullname);
        signode.attrs.ids.push(node_id.clone());
        self.registry.note_explicit_id(&node_id);
        self.py_object_records.push(super::PyObjectRecord {
            fullname: fullname.clone(),
            objtype: objtype.to_string(),
            node_id: node_id.clone(),
            aliased: false,
            source: signode.span.source,
            lineno: input.lineno,
        });
        // `:canonical:` registers an alias — except on py:type, where the
        // option is display-only (`_object.py:427-437`, §6).
        if py.kind != PyObjectKind::TypeAlias {
            if let Some(canonical) = opt_str("canonical").filter(|c| !c.is_empty()) {
                self.py_object_records.push(super::PyObjectRecord {
                    fullname: canonical,
                    objtype: objtype.to_string(),
                    node_id: node_id.clone(),
                    aliased: true,
                    source: signode.span.source,
                    lineno: input.lineno,
                });
            }
        }
        let has = |n: &'static str| opt_get(&input.options, n).is_some();
        if has("no-index-entry") || has("noindexentry") {
            return;
        }
        let index_text = py_index_text(
            py,
            input,
            modname.as_deref(),
            name,
            self.py.add_module_names,
        );
        if !index_text.is_empty() {
            entries.push(index_entry_tuple("single", &index_text, &node_id, "", None));
        }
        // PyFunction adds its entry in its own add_target_and_index
        // (`__init__.py:95-109`): module-less functions are a PAIR entry
        // (trap 10).
        if py.kind == PyObjectKind::Function {
            match &modname {
                Some(modname) => entries.push(index_entry_tuple(
                    "single",
                    &format!("{name}() (in module {modname})"),
                    &node_id,
                    "",
                    None,
                )),
                None => entries.push(index_entry_tuple(
                    "pair",
                    &format!("built-in function; {name}()"),
                    &node_id,
                    "",
                    None,
                )),
            }
        }
    }

    /// `PyObject.before_content` (`_object.py:449-480`): class scope from
    /// the LAST signature's name — the fullname for nesting kinds, the
    /// written prefix otherwise — plus the `:module:` option push.
    fn py_before_content(
        &mut self,
        py: PyDirective,
        names: &[(String, String)],
        input: &DirectiveInput<'_>,
    ) {
        let mut prefix: Option<String> = None;
        if let Some((fullname, name_prefix)) = names.last() {
            if py.allow_nesting() {
                prefix = Some(fullname.clone());
            } else if !name_prefix.is_empty() {
                prefix = Some(name_prefix.trim_matches('.').to_string());
            }
        }
        if let Some(prefix) = prefix.filter(|p| !p.is_empty()) {
            self.py_class = Some(prefix.clone());
            if py.allow_nesting() {
                self.py_classes.push(prefix);
            }
        }
        if let Some(OptVal::Str(module)) = opt_get(&input.options, "module") {
            self.py_modules.push(self.py_module.take());
            self.py_module = Some(module.clone());
        }
    }

    /// `PyObject.after_content` (`_object.py:482-503`): pop the nesting
    /// stack (nesting kinds only), always reassign `py:class` from the
    /// stack top, and undo the `:module:` push.
    fn py_after_content(&mut self, py: PyDirective, input: &DirectiveInput<'_>) {
        if py.allow_nesting() {
            self.py_classes.pop();
        }
        self.py_class = self.py_classes.last().cloned();
        if opt_get(&input.options, "module").is_some() {
            // `modules.pop()` when the stack has entries, else the
            // ref_context key is removed — both read back as None here.
            self.py_module = self.py_modules.pop().flatten();
        }
    }

    /// `DocFieldTransformer(self).transform_all(content_node)` — the base
    /// `run` applies it to EVERY object description AFTER the
    /// `object-description-transform` event and BEFORE `after_content`,
    /// so the ref_context the field xrefs read is still the object's own
    /// scope. Only immediate `field_list` children are transformed
    /// (`docfields.py:354-359`). The `map` is the directive's
    /// `get_field_type_map()`: the py table for py kinds, empty for the
    /// std kinds (none of them declare `doc_field_types`), whose fields
    /// therefore all take the unknown rename-and-pass-through branch.
    fn transform_doc_fields(&mut self, content: &mut Node, map: DocFieldTypeMap) {
        let ctx = crate::py::annotations::PyRefContext {
            module: self.py_module.clone(),
            class_: self.py_class.clone(),
        };
        for child in &mut content.children {
            if child.kind == kinds::FIELD_LIST {
                transform_doc_field_list(child, map, &ctx, &self.py);
            }
        }
    }

    /// `PyModule.run` (`domains/python/__init__.py:492-536`, [PY §1.5]):
    /// node order `[index?, target, *content]`, always-set ref_context
    /// (trap 6), registration unless `:no-index:`.
    fn run_py_module(&mut self, input: DirectiveInput<'_>, out: &mut Vec<Node>) {
        let Some(argument) = input.arguments.first() else {
            return;
        };
        let modname = argument.trim().to_string();
        let has = |n: &'static str| opt_get(&input.options, n).is_some();
        let no_index = has("no-index") || has("noindex");
        // ALWAYS sets the module scope, even under `:no-index:` (trap 6).
        self.py_module = Some(modname.clone());
        // Content parses BEFORE the module's own id is allocated
        // (`__init__.py:505-510`), so ids taken by content come first.
        // sphinx parses it with allow_section_headings=True; sections
        // inside nested content are not representable in this parser (a
        // pre-existing wave-4 limitation shared by every nested parse),
        // and the T8 corpus excludes section-bearing module content.
        let content = self.parse_nested(&input.content, "py_module");
        if !no_index {
            let node_id = self.registry.sphinx_make_id("module", &modname);
            self.registry.note_explicit_id(&node_id);
            let mut target = Node::elem(kinds::TARGET, input.span);
            target.attrs.ids.push(node_id.clone());
            target.set("ismod", AttrValue::Int(1));
            let opt = |name: &'static str| match opt_get(&input.options, name) {
                Some(OptVal::Str(s)) => s.clone(),
                _ => String::new(),
            };
            self.py_module_records.push(super::PyModuleRecord {
                name: modname.clone(),
                node_id: node_id.clone(),
                synopsis: opt("synopsis"),
                platform: opt("platform"),
                deprecated: has("deprecated"),
                source: input.span.source,
                lineno: input.lineno,
            });
            // `note_object(modname, 'module', node_id)` (`__init__.py:522`)
            // — modules also join the objects table.
            self.py_object_records.push(super::PyObjectRecord {
                fullname: modname.clone(),
                objtype: "module".to_string(),
                node_id: node_id.clone(),
                aliased: false,
                source: input.span.source,
                lineno: input.lineno,
            });
            if !has("no-index-entry") {
                let mut index = Node::elem("index", input.span);
                index.set(
                    "entries",
                    AttrValue::List(vec![index_entry_tuple(
                        "pair",
                        &format!("module; {modname}"),
                        &node_id,
                        "",
                        None,
                    )]),
                );
                out.push(index);
            }
            // NOTE §Scope-3: this is the PRE-propagation shape — the target
            // keeps its ids; docutils PropagateTargets (a transform this
            // parse layer deliberately does not run) is what turns it into
            // `refid` and moves the id onto the next body node (trap 5).
            out.push(target);
        }
        out.extend(content);
    }

    /// `PyCurrentModule.run` (`__init__.py:550-556`): pure ref_context
    /// state, no nodes; the literal argument `None` pops the scope.
    fn run_py_currentmodule(&mut self, input: DirectiveInput<'_>) {
        let Some(argument) = input.arguments.first() else {
            return;
        };
        let modname = argument.trim();
        if modname == "None" {
            self.py_module = None;
        } else {
            self.py_module = Some(modname.to_string());
        }
    }

    /// `ConfigurationValue.transform_content` (`domains/std:153-185`):
    /// `:type:` and `:default:` render as a field list prepended to the
    /// description content, each field followed by its own inline messages.
    fn confval_transform_content(&mut self, input: &DirectiveInput<'_>, content: &mut Node) {
        let mut field_list = Node::elem(kinds::FIELD_LIST, input.span);
        for (option, label) in [("type", "Type"), ("default", "Default")] {
            let Some(OptVal::Str(value)) = opt_get(&input.options, option) else {
                continue;
            };
            let parsed = self.inline(&value.clone(), input.span, input.lineno);
            let mut field_name = Node::elem(kinds::FIELD_NAME, input.span);
            field_name.children.push(Node::text_node(label, input.span));
            let mut field_body = Node::elem(kinds::FIELD_BODY, input.span);
            field_body.children = parsed.nodes;
            let mut field = Node::elem(kinds::FIELD, input.span);
            field.children.push(field_name);
            field.children.push(field_body);
            field_list.children.push(field);
            field_list.children.extend(parsed.messages);
        }
        if !field_list.children.is_empty() {
            content.children.insert(0, field_list);
        }
    }

    /// versionadded family (sphinx/domains/changeset.py VersionChange):
    /// a versionmodified node holding ONE translatable="0" paragraph whose
    /// lead-in inline ends with '.' (no text) or ': ' (text follows as
    /// siblings in the same paragraph).
    fn run_version_change(
        &mut self,
        info: &'static (&'static str, &'static str, &'static str),
        input: DirectiveInput<'_>,
        out: &mut Vec<Node>,
    ) {
        let (type_name, label, lead_fmt) = *info;
        let version = &input.arguments[0];
        // Inline messages from the explanation must anchor on the text's
        // own line, not the directive marker (review finding 41).
        let mut text_lineno = input.lineno;
        let text: Option<String> = input
            .arguments
            .get(1)
            .cloned()
            .or_else(|| {
                if input.content.is_empty() {
                    None
                } else {
                    text_lineno = input.content[0].lineno;
                    Some(self.join_lines(&input.content))
                }
            })
            .filter(|t| !t.is_empty());
        let mut node = Node::elem("versionmodified", input.span);
        node.set("type", AttrValue::Str(type_name.to_string()));
        node.set("version", AttrValue::Str(version.clone()));
        let lead_base = lead_fmt.replace("{}", version);
        let lead = match &text {
            Some(_) => format!("{lead_base}: "),
            None => format!("{lead_base}."),
        };
        let mut para = Node::elem(kinds::PARAGRAPH, input.span);
        para.set("translatable", AttrValue::Int(0));
        let mut inner = Node::elem("inline", input.span);
        inner
            .attrs
            .classes
            .extend(["versionmodified".to_string(), label.to_string()]);
        inner.children.push(Node::text_node(lead, input.span));
        para.children.push(inner);
        let mut messages = Vec::new();
        if let Some(t) = text {
            let inline = self.inline(&t, input.span, text_lineno);
            para.children.extend(inline.nodes);
            messages = inline.messages;
        }
        node.children.push(para);
        out.push(node);
        out.extend(messages);
    }

    /// seealso (sphinx/directives/other.py): admonition-shaped custom
    /// node with no attributes.
    fn run_seealso(&mut self, input: DirectiveInput<'_>, out: &mut Vec<Node>) {
        if input.content.is_empty() {
            out.push(self.directive_content_error(
                input.name,
                input.span.source,
                input.lineno,
                input.rawsource,
            ));
            return;
        }
        let mut node = Node::elem("seealso", input.span);
        if let Some(OptVal::StrList(classes)) = opt_get(&input.options, "class") {
            node.attrs.classes.extend(classes.iter().cloned());
        }
        self.directive_add_name(
            &mut node,
            &input.options,
            input.span.source,
            input.lineno,
            out,
        );
        let content = self.parse_nested(&input.content, "seealso");
        node.children.extend(content);
        out.push(node);
    }

    /// sphinx code-block (sphinx/directives/code.py CodeBlock): language
    /// falls back to the `.. highlight::` state then the 'default'
    /// sentinel; :caption: wraps in a literal-block-wrapper container
    /// that takes the ids/names.
    fn run_sphinx_code_block(&mut self, input: DirectiveInput<'_>, out: &mut Vec<Node>) {
        let language = input
            .arguments
            .first()
            .cloned()
            .or_else(|| self.highlight_language.clone())
            .unwrap_or_else(|| "default".to_string());
        // sphinx util.parselinenos + CodeBlock.run: an invalid spec
        // REPLACES the whole block with a WARNING system_message;
        // out-of-range lines are filtered (review findings 31/43/45/46).
        let nlines = input.content.len() as i64;
        let mut hl_lines: Vec<i64> = Vec::new();
        if let Some(OptVal::Str(spec)) = opt_get(&input.options, "emphasize-lines") {
            match parse_linenos(spec, nlines) {
                Ok(lines_list) => hl_lines = lines_list,
                Err(msg) => {
                    out.push(self.msg(messages::WARNING, &msg, input.span.source, input.lineno));
                    return;
                }
            }
        }
        let highlight_args = if hl_lines.is_empty() {
            "{}".to_string()
        } else {
            format!(
                "{{'hl_lines': [{}]}}",
                hl_lines
                    .iter()
                    .map(|n| n.to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        let mut lb = Node::elem(kinds::LITERAL_BLOCK, input.span);
        lb.set(
            "force",
            AttrValue::Int(i64::from(opt_get(&input.options, "force").is_some())),
        );
        lb.set("highlight_args", AttrValue::Str(highlight_args));
        lb.set("language", AttrValue::Str(language));
        if opt_get(&input.options, "linenos").is_some() {
            lb.set("linenos", AttrValue::Int(1));
        }
        lb.set("xml:space", AttrValue::Str("preserve".to_string()));
        if let Some(OptVal::StrList(classes)) = opt_get(&input.options, "class") {
            lb.attrs.classes.extend(classes.iter().cloned());
        }
        let code = self.join_lines(&input.content);
        lb.children.push(Node::text_node(code, input.span));
        match opt_get(&input.options, "caption") {
            Some(OptVal::Str(caption_text)) => {
                let mut container = Node::elem("container", input.span);
                container
                    .attrs
                    .classes
                    .push("literal-block-wrapper".to_string());
                container.set("literal_block", AttrValue::Int(1));
                self.directive_add_name(
                    &mut container,
                    &input.options,
                    input.span.source,
                    input.lineno,
                    out,
                );
                let inline = self.inline(&caption_text.clone(), input.span, input.lineno);
                let mut caption = Node::elem("caption", input.span);
                caption.children = inline.nodes;
                container.children.push(caption);
                container.children.push(lb);
                out.push(container);
                out.extend(inline.messages);
            }
            _ => {
                self.directive_add_name(
                    &mut lb,
                    &input.options,
                    input.span.source,
                    input.lineno,
                    out,
                );
                out.push(lb);
            }
        }
    }

    /// sphinx highlight: emits a highlightlang node AND sets the state
    /// later code-blocks read (env.temp_data parity).
    fn run_highlight(&mut self, input: DirectiveInput<'_>, out: &mut Vec<Node>) {
        let lang = input.arguments[0].clone();
        self.highlight_language = Some(lang.clone());
        let mut node = Node::elem("highlightlang", input.span);
        node.set(
            "force",
            AttrValue::Int(i64::from(opt_get(&input.options, "force").is_some())),
        );
        node.set("lang", AttrValue::Str(lang));
        let threshold = match opt_get(&input.options, "linenothreshold") {
            Some(OptVal::Int(n)) => *n,
            _ => i64::MAX,
        };
        node.set("linenothreshold", AttrValue::Int(threshold));
        out.push(node);
    }

    /// sphinx only: expr stored verbatim; evaluation is a later build
    /// phase.
    fn run_only(&mut self, input: DirectiveInput<'_>, out: &mut Vec<Node>) {
        let mut node = Node::elem("only", input.span);
        node.set("expr", AttrValue::Str(input.arguments[0].clone()));
        let content = self.parse_nested(&input.content, "only");
        node.children.extend(content);
        out.push(node);
    }

    /// sphinx toctree: entries recorded (as authored, with per-entry
    /// lines) for the build pipeline; the node is a best-effort
    /// `compound.toctree-wrapper > toctree` (probe shape; exact attr
    /// parity lands with the sphinx-fixture toctree cases).
    fn run_toctree(&mut self, input: DirectiveInput<'_>, out: &mut Vec<Node>) {
        let glob = matches!(opt_get(&input.options, "glob"), Some(OptVal::Null));
        let mut entries: Vec<super::ToctreeEntryRecord> = Vec::new();
        let mut raw_entries: Vec<String> = Vec::new();
        for l in &input.content {
            if l.is_blank() {
                continue;
            }
            let t = self.sources.line_text(*l).trim().to_string();
            let t = t.as_str();
            raw_entries.push(t.to_string());
            // sphinx explicit_title_re `^(.+?)\s*<(.*?)>$`: the TITLE part
            // must be nonempty — a bare `<foo>` entry is a literal target
            // named '<foo>' (review finding 40).
            let (title, target) = match crate::env::toctree::split_explicit_title(t) {
                Some((title, target)) => (Some(title.to_string()), target.to_string()),
                None => (None, t.to_string()),
            };
            entries.push(super::ToctreeEntryRecord {
                title,
                target,
                line: l.lineno,
            });
        }
        // Full sphinx attr set (oracle-pinned). entries/includefiles are
        // resolved against the environment's document set the way
        // `TocTree.parse_content` does — including its warnings, which ride
        // the record to the builder; a parse with no environment
        // (`found_docs: None`) resolves nothing and leaves both empty.
        let resolved = match &self.found_docs {
            Some(found) => {
                crate::env::toctree::resolve_entries(&crate::env::toctree::ToctreeContent {
                    content: &raw_entries,
                    docname: &self.docname,
                    glob,
                    reversed: opt_get(&input.options, "reversed").is_some(),
                    line: input.lineno,
                    found_docs: found,
                    source_suffixes: SOURCE_SUFFIXES,
                    exclude_patterns: &self.exclude_patterns,
                })
            }
            None => crate::env::toctree::ResolvedEntries::default(),
        };
        self.toctree_records.push(super::ToctreeRecord {
            glob,
            entries: entries.clone(),
            line: input.lineno,
            warnings: resolved.warnings.clone(),
        });
        let mut toctree = Node::elem("toctree", input.span);
        match opt_get(&input.options, "caption") {
            // pformat renders a Python None attr value as "True".
            Some(OptVal::Str(c)) => toctree.set("caption", AttrValue::Str(c.clone())),
            _ => toctree.set("caption", AttrValue::Str("True".to_string())),
        }
        toctree.set("entries", resolved.entries_attr());
        toctree.set("glob", AttrValue::Int(i64::from(glob)));
        toctree.set(
            "hidden",
            AttrValue::Int(i64::from(opt_get(&input.options, "hidden").is_some())),
        );
        toctree.set("includefiles", resolved.includefiles_attr());
        toctree.set(
            "includehidden",
            AttrValue::Int(i64::from(
                opt_get(&input.options, "includehidden").is_some(),
            )),
        );
        let maxdepth = match opt_get(&input.options, "maxdepth") {
            Some(OptVal::Int(d)) => *d,
            _ => -1,
        };
        toctree.set("maxdepth", AttrValue::Int(maxdepth));
        // sphinx `int_or_nothing` (directives/other.py:36): a bare
        // `:numbered:` is depth 999, not 999_999.
        let numbered = match opt_get(&input.options, "numbered") {
            Some(OptVal::Str(s)) if s.is_empty() => 999,
            Some(OptVal::Str(s)) => py_int(s).unwrap_or(0),
            _ => 0,
        };
        toctree.set("numbered", AttrValue::Int(numbered));
        toctree.set("parent", AttrValue::Str(self.docname.clone()));
        toctree.set("rawentries", AttrValue::Str(String::new()));
        toctree.set(
            "titlesonly",
            AttrValue::Int(i64::from(opt_get(&input.options, "titlesonly").is_some())),
        );
        let mut compound = Node::elem("compound", input.span);
        compound.attrs.classes.push("toctree-wrapper".to_string());
        compound.children.push(toctree);
        out.push(compound);
    }

    fn substitution_context_error(&self, input: &DirectiveInput<'_>, out: &mut Vec<Node>) -> bool {
        if self.substitution_ctx.is_some() {
            return false;
        }
        out.push(self.directive_run_error(
            &format!(
                "Invalid context: the \"{}\" directive can only be used within a substitution definition.",
                input.name
            ),
            input.span.source, input.lineno,
            input.rawsource,
        ));
        true
    }

    /// replace (misc.py:357-387).
    fn run_replace(&mut self, input: DirectiveInput<'_>, out: &mut Vec<Node>) {
        if self.substitution_context_error(&input, out) {
            return;
        }
        if input.content.is_empty() {
            out.push(self.directive_content_error(
                input.name,
                input.span.source,
                input.lineno,
                input.rawsource,
            ));
            return;
        }
        // The nested parse runs OUTSIDE the SubstitutionDef state: an
        // embedded `.. date::` inside replace content must context-error
        // exactly like at body level (review finding 22).
        let saved_ctx = self.substitution_ctx.take();
        let children = self.parse_nested(&input.content, "substitution_definition");
        self.substitution_ctx = saved_ctx;
        let mut msgs: Vec<Node> = Vec::new();
        let mut paragraphs: Vec<Node> = Vec::new();
        let mut others = false;
        for c in children {
            if c.kind == kinds::SYSTEM_MESSAGE {
                let mut m = c;
                m.attrs.backrefs.clear();
                msgs.push(m);
            } else if c.kind == kinds::PARAGRAPH {
                paragraphs.push(c);
            } else {
                others = true;
            }
        }
        if paragraphs.len() == 1 && !others {
            out.extend(msgs);
            out.extend(paragraphs.remove(0).children);
        } else {
            // reporter.error without a literal child (misc.py:378-383).
            out.push(self.msg(
                messages::ERROR,
                &format!(
                    "Error in \"{}\" directive: may contain a single paragraph only.",
                    input.name
                ),
                input.span.source,
                input.lineno,
            ));
        }
    }

    /// unicode (misc.py:390-431).
    fn run_unicode(&mut self, input: DirectiveInput<'_>, out: &mut Vec<Node>) {
        if self.substitution_context_error(&input, out) {
            return;
        }
        let trim = opt_get(&input.options, "trim").is_some();
        let ltrim = opt_get(&input.options, "ltrim").is_some();
        let rtrim = opt_get(&input.options, "rtrim").is_some();
        if let Some(ctx) = self.substitution_ctx.as_mut() {
            ctx.ltrim |= trim || ltrim;
            ctx.rtrim |= trim || rtrim;
        }
        let arg = &input.arguments[0];
        let codes_text = &arg[..unicode_comment_cut(arg)];
        for code in codes_text.split_whitespace() {
            match unicode_code(code) {
                Ok(s) => out.push(Node::text_node(s, input.span)),
                Err(detail) => {
                    out.push(self.directive_run_error(
                        &format!("Invalid character code: {code}\nValueError: {detail}"),
                        input.span.source,
                        input.lineno,
                        input.rawsource,
                    ));
                    return;
                }
            }
        }
    }

    /// date (misc.py:639-666): strftime at PARSE time (deliberately
    /// non-deterministic output; the fixture corpus avoids success cases).
    fn run_date(&mut self, input: DirectiveInput<'_>, out: &mut Vec<Node>) {
        if self.substitution_context_error(&input, out) {
            return;
        }
        let format = if input.content.is_empty() {
            "%Y-%m-%d".to_string()
        } else {
            self.join_lines(&input.content)
        };
        out.push(Node::text_node(strftime_now(&format), input.span));
    }

    /// Table.make_title (tables.py:46-57).
    fn table_make_title(&mut self, input: &DirectiveInput<'_>) -> (Option<Node>, Vec<Node>) {
        match input.arguments.first() {
            Some(text) => {
                let inline = self.inline(text, input.span, input.lineno);
                let mut title = Node::elem(kinds::TITLE, input.span);
                title.children = inline.nodes;
                (Some(title), inline.messages)
            }
            None => (None, Vec::new()),
        }
    }

    /// Shared tail of the three table directives: user classes, width,
    /// align, the colwidths marker class, :name:, then the title at
    /// index 0 (tables.py:141-171).
    #[allow(clippy::too_many_arguments)]
    fn finish_table(
        &mut self,
        mut table: Node,
        input: &DirectiveInput<'_>,
        title: Option<Node>,
        title_messages: Vec<Node>,
        out: &mut Vec<Node>,
    ) {
        if let Some(OptVal::Str(w)) = opt_get(&input.options, "width") {
            table.set("width", AttrValue::Str(w.clone()));
        }
        if let Some(OptVal::Str(a)) = opt_get(&input.options, "align") {
            table.set("align", AttrValue::Str(a.clone()));
        }
        self.directive_add_name(
            &mut table,
            &input.options,
            input.span.source,
            input.lineno,
            out,
        );
        if let Some(t) = title {
            table.children.insert(0, t);
        }
        out.push(table);
        out.extend(title_messages);
    }

    /// table (tables.py RSTTable:127-172).
    fn run_rst_table(&mut self, input: DirectiveInput<'_>, out: &mut Vec<Node>) {
        if input.content.is_empty() {
            // RSTTable's missing-content diagnostic is a WARNING, unlike
            // the assert_has_content ERROR family (tables.py:135-139).
            out.push(self.directive_run_message(
                messages::WARNING,
                &format!(
                    "Content block expected for the \"{}\" directive; none found.",
                    input.name
                ),
                input.span.source,
                input.lineno,
                input.rawsource,
            ));
            return;
        }
        let (title, title_messages) = self.table_make_title(&input);
        let children = self.parse_nested(&input.content, "element");
        if children.len() != 1 || children[0].kind != kinds::TABLE {
            out.push(self.directive_run_error(
                &format!(
                    "Error parsing content block for the \"{}\" directive: exactly one table expected.",
                    input.name
                ),
                input.span.source, input.lineno,
                input.rawsource,
            ));
            return;
        }
        let mut table = children.into_iter().next().expect("length checked");
        // User classes precede the colwidths marker class here (RSTTable
        // run order); csv/list get theirs appended AFTER the build-time
        // marker instead — both orders fixture-pinned.
        if let Some(OptVal::StrList(cls)) = opt_get(&input.options, "class") {
            table.attrs.classes.extend(cls.iter().cloned());
        }
        match opt_get(&input.options, "widths") {
            Some(OptVal::Str(kw)) if kw == "auto" => {
                table.attrs.classes.push("colwidths-auto".to_string());
            }
            Some(OptVal::Str(_)) => {
                // 'grid': keep the syntax-derived colwidths.
                table.attrs.classes.push("colwidths-given".to_string());
            }
            Some(OptVal::IntList(list)) => {
                let n_cols = table
                    .children
                    .first()
                    .map(|tg| {
                        tg.children
                            .iter()
                            .filter(|c| c.kind == kinds::COLSPEC)
                            .count()
                    })
                    .unwrap_or(0);
                if list.len() != n_cols {
                    out.push(self.directive_run_error(
                        &format!(
                            "\"{}\" widths do not match the number of columns in table ({}).",
                            input.name, n_cols
                        ),
                        input.span.source,
                        input.lineno,
                        input.rawsource,
                    ));
                    return;
                }
                if let Some(tg) = table.children.first_mut() {
                    let mut i = 0usize;
                    for c in &mut tg.children {
                        if c.kind == kinds::COLSPEC {
                            c.set("colwidth", AttrValue::Int(list[i]));
                            i += 1;
                        }
                    }
                }
                table.attrs.classes.push("colwidths-given".to_string());
            }
            _ => {}
        }
        self.finish_table(table, &input, title, title_messages, out);
    }

    /// csv-table (tables.py CSVTable:175-403).
    fn run_csv_table(&mut self, input: DirectiveInput<'_>, out: &mut Vec<Node>) {
        let has_file = opt_get(&input.options, "file").is_some();
        let has_url = opt_get(&input.options, "url").is_some();
        // get_csv_data (tables.py:321-388).
        let csv_text: String;
        if !input.content.is_empty() {
            if has_file || has_url {
                out.push(self.directive_run_error(
                    &format!(
                        "\"{}\" directive may not both specify an external file and have content.",
                        input.name
                    ),
                    input.span.source,
                    input.lineno,
                    input.rawsource,
                ));
                return;
            }
            csv_text = self.join_lines(&input.content);
        } else if has_file {
            if has_url {
                out.push(self.directive_run_error(
                    &format!(
                        "The \"file\" and \"url\" options may not be simultaneously specified for the \"{}\" directive.",
                        input.name
                    ),
                    input.span.source, input.lineno,
                    input.rawsource,
                ));
                return;
            }
            let Some(OptVal::Str(path)) = opt_get(&input.options, "file") else {
                unreachable!("file option is Path-converted");
            };
            let base = std::path::Path::new(self.sources.path(0))
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_default();
            match std::fs::read_to_string(base.join(path)) {
                Ok(t) => csv_text = t,
                Err(_) => {
                    // Unlike raw's io.error_string (InputError: prefix),
                    // the csv path formats the bare OSError.
                    out.push(self.directive_run_message(
                        messages::SEVERE,
                        &format!(
                            "Problems with \"{}\" directive path:\n[Errno 2] No such file or directory: {}.",
                            input.name,
                            py_repr(Some(path))
                        ),
                        input.span.source, input.lineno,
                        input.rawsource,
                    ));
                    return;
                }
            }
        } else {
            out.push(self.directive_run_message(
                messages::WARNING,
                &format!(
                    "The \"{}\" directive requires content; none supplied.",
                    input.name
                ),
                input.span.source,
                input.lineno,
                input.rawsource,
            ));
            return;
        }
        let (title, title_messages) = self.table_make_title(&input);
        // Dialect (tables.py DocutilsDialect:198-220).
        let delim = match opt_get(&input.options, "delim") {
            Some(OptVal::Str(s)) => s.chars().next().unwrap_or(','),
            _ => ',',
        };
        let quote = match opt_get(&input.options, "quote") {
            Some(OptVal::Str(s)) => s.chars().next().unwrap_or('"'),
            _ => '"',
        };
        let escape = match opt_get(&input.options, "escape") {
            Some(OptVal::Str(s)) => s.chars().next(),
            _ => None,
        };
        let skipinitialspace = opt_get(&input.options, "keepspace").is_none();
        let doublequote = escape.is_none();
        let header_rows = match opt_get(&input.options, "header-rows") {
            Some(OptVal::Int(n)) => *n as usize,
            _ => 0,
        };
        let stub_columns = match opt_get(&input.options, "stub-columns") {
            Some(OptVal::Int(n)) => *n as usize,
            _ => 0,
        };
        let header_option_rows: Vec<Vec<String>> = match opt_get(&input.options, "header") {
            Some(OptVal::Str(h)) => {
                parse_csv_text(h, delim, quote, escape, doublequote, skipinitialspace)
            }
            _ => Vec::new(),
        };
        let rows = parse_csv_text(
            &csv_text,
            delim,
            quote,
            escape,
            doublequote,
            skipinitialspace,
        );
        let max_header_cols = header_option_rows.iter().map(Vec::len).max().unwrap_or(0);
        let max_cols = rows
            .iter()
            .map(Vec::len)
            .max()
            .unwrap_or(0)
            .max(max_header_cols);
        let row_lens: Vec<usize> = rows.iter().map(Vec::len).collect();
        if let Err(msg) = Self::check_table_dimensions(
            input.name,
            rows.len(),
            &row_lens,
            header_rows,
            stub_columns,
        ) {
            out.push(self.directive_run_error(
                &msg,
                input.span.source,
                input.lineno,
                input.rawsource,
            ));
            return;
        }
        // Column widths (tables.py:101-118).
        let widths_opt = opt_get(&input.options, "widths").cloned();
        let col_widths: Vec<i64> = match &widths_opt {
            Some(OptVal::IntList(list)) => {
                if list.len() != max_cols {
                    out.push(self.directive_run_error(
                        &format!(
                            "\"{}\" widths do not match the number of columns in table ({}).",
                            input.name, max_cols
                        ),
                        input.span.source,
                        input.lineno,
                        input.rawsource,
                    ));
                    return;
                }
                list.clone()
            }
            _ => {
                if max_cols == 0 {
                    out.push(self.directive_run_error(
                        "No table data detected in CSV file.",
                        input.span.source,
                        input.lineno,
                        input.rawsource,
                    ));
                    return;
                }
                vec![(100 / max_cols) as i64; max_cols]
            }
        };
        // Cells -> entry nodes; short rows extend with empty cells.
        let mut make_row = |cells: &[String]| -> Vec<Node> {
            let mut entries = Vec::with_capacity(max_cols);
            for i in 0..max_cols {
                let mut entry = Node::elem(kinds::ENTRY, input.span);
                if let Some(cell) = cells.get(i) {
                    if !cell.is_empty() {
                        entry.children =
                            self.parse_detached(cell, input.lineno, input.span.source, "entry");
                    }
                }
                entries.push(entry);
            }
            entries
        };
        let mut head: Vec<Vec<Node>> = Vec::new();
        let mut body: Vec<Vec<Node>> = Vec::new();
        for cells in &header_option_rows {
            head.push(make_row(cells));
        }
        for (i, cells) in rows.iter().enumerate() {
            if i < header_rows {
                head.push(make_row(cells));
            } else {
                body.push(make_row(cells));
            }
        }
        let mut table = Self::build_directive_table(
            &col_widths,
            stub_columns,
            widths_opt.as_ref(),
            head,
            body,
            input.span,
        );
        if let Some(OptVal::StrList(cls)) = opt_get(&input.options, "class") {
            table.attrs.classes.extend(cls.iter().cloned());
        }
        self.finish_table(table, &input, title, title_messages, out);
    }

    /// list-table (tables.py ListTable:406-523).
    fn run_list_table(&mut self, input: DirectiveInput<'_>, out: &mut Vec<Node>) {
        if input.content.is_empty() {
            out.push(self.directive_run_error(
                &format!(
                    "The \"{}\" directive is empty; content required.",
                    input.name
                ),
                input.span.source,
                input.lineno,
                input.rawsource,
            ));
            return;
        }
        let (title, title_messages) = self.table_make_title(&input);
        let children = self.parse_nested(&input.content, "element");
        let content_error = |me: &Self, detail: &str| -> Node {
            me.directive_run_error(
                &format!(
                    "Error parsing content block for the \"{}\" directive: {detail}",
                    input.name
                ),
                input.span.source,
                input.lineno,
                input.rawsource,
            )
        };
        if children.len() != 1 || children[0].kind != kinds::BULLET_LIST {
            out.push(content_error(self, "exactly one bullet list expected."));
            return;
        }
        let outer = children.into_iter().next().expect("length checked");
        let mut table_data: Vec<Vec<Vec<Node>>> = Vec::new();
        let mut first_len: Option<usize> = None;
        for (i, item) in outer.children.into_iter().enumerate() {
            let one_inner_list =
                item.children.len() == 1 && item.children[0].kind == kinds::BULLET_LIST;
            if !one_inner_list {
                out.push(content_error(
                    self,
                    &format!(
                        "two-level bullet list expected, but row {} does not contain a second-level bullet list.",
                        i + 1
                    ),
                ));
                return;
            }
            let inner = item.children.into_iter().next().expect("length checked");
            let row: Vec<Vec<Node>> = inner.children.into_iter().map(|it| it.children).collect();
            if let Some(f) = first_len {
                if row.len() != f {
                    out.push(content_error(
                        self,
                        &format!(
                            "uniform two-level bullet list expected, but row {} does not contain the same number of items as row 1 ({} vs {}).",
                            i + 1,
                            row.len(),
                            f
                        ),
                    ));
                    return;
                }
            } else {
                first_len = Some(row.len());
            }
            table_data.push(row);
        }
        let header_rows = match opt_get(&input.options, "header-rows") {
            Some(OptVal::Int(n)) => *n as usize,
            _ => 0,
        };
        let stub_columns = match opt_get(&input.options, "stub-columns") {
            Some(OptVal::Int(n)) => *n as usize,
            _ => 0,
        };
        let row_lens: Vec<usize> = table_data.iter().map(Vec::len).collect();
        if let Err(msg) = Self::check_table_dimensions(
            input.name,
            table_data.len(),
            &row_lens,
            header_rows,
            stub_columns,
        ) {
            out.push(self.directive_run_error(
                &msg,
                input.span.source,
                input.lineno,
                input.rawsource,
            ));
            return;
        }
        let n_cols = first_len.unwrap_or(0);
        let widths_opt = opt_get(&input.options, "widths").cloned();
        let col_widths: Vec<i64> = match &widths_opt {
            Some(OptVal::IntList(list)) => {
                if list.len() != n_cols {
                    out.push(self.directive_run_error(
                        &format!(
                            "\"{}\" widths do not match the number of columns in table ({}).",
                            input.name, n_cols
                        ),
                        input.span.source,
                        input.lineno,
                        input.rawsource,
                    ));
                    return;
                }
                list.clone()
            }
            _ => {
                if n_cols == 0 {
                    out.push(content_error(self, "exactly one bullet list expected."));
                    return;
                }
                vec![(100 / n_cols) as i64; n_cols]
            }
        };
        let mut all_rows: Vec<Vec<Node>> = Vec::new();
        for row in table_data {
            let entries: Vec<Node> = row
                .into_iter()
                .map(|cell_children| {
                    let mut entry = Node::elem(kinds::ENTRY, input.span);
                    entry.children = cell_children;
                    entry
                })
                .collect();
            all_rows.push(entries);
        }
        let body = all_rows.split_off(header_rows.min(all_rows.len()));
        let head = all_rows;
        let mut table = Self::build_directive_table(
            &col_widths,
            stub_columns,
            widths_opt.as_ref(),
            head,
            body,
            input.span,
        );
        if let Some(OptVal::StrList(cls)) = opt_get(&input.options, "class") {
            table.attrs.classes.extend(cls.iter().cloned());
        }
        self.finish_table(table, &input, title, title_messages, out);
    }

    /// check_table_dimensions (tables.py:59-91). Err = the message text.
    fn check_table_dimensions(
        name: &str,
        rows: usize,
        row_lens: &[usize],
        header_rows: usize,
        stub_columns: usize,
    ) -> Result<(), String> {
        if rows < header_rows {
            return Err(format!(
                "{header_rows} header row(s) specified but only {rows} row(s) of data supplied (\"{name}\" directive)."
            ));
        }
        if rows == header_rows && header_rows > 0 {
            return Err(format!(
                "Insufficient data supplied ({rows} row(s)); no data remaining for table body, required by \"{name}\" directive."
            ));
        }
        for len in row_lens {
            if *len < stub_columns {
                return Err(format!(
                    "{stub_columns} stub column(s) specified but only {len} columns(s) of data supplied (\"{name}\" directive)."
                ));
            }
            if *len == stub_columns && stub_columns > 0 {
                return Err(format!(
                    "Insufficient data supplied ({len} columns(s)); no data remaining for table body, required by \"{name}\" directive."
                ));
            }
        }
        Ok(())
    }

    /// build_table (states.py:1911-1953) for the csv/list table paths.
    fn build_directive_table(
        col_widths: &[i64],
        stub_columns: usize,
        widths_opt: Option<&OptVal>,
        head: Vec<Vec<Node>>,
        body: Vec<Vec<Node>>,
        span: Span,
    ) -> Node {
        let mut table = Node::elem(kinds::TABLE, span);
        match widths_opt {
            Some(OptVal::Str(kw)) if kw == "auto" => {
                table.attrs.classes.push("colwidths-auto".to_string());
            }
            Some(OptVal::IntList(_)) => {
                table.attrs.classes.push("colwidths-given".to_string());
            }
            _ => {}
        }
        let mut tgroup = Node::elem(kinds::TGROUP, span);
        tgroup.set("cols", AttrValue::Int(col_widths.len() as i64));
        for (i, w) in col_widths.iter().enumerate() {
            let mut colspec = Node::elem(kinds::COLSPEC, span);
            colspec.set("colwidth", AttrValue::Int(*w));
            if i < stub_columns {
                colspec.set("stub", AttrValue::Int(1));
            }
            tgroup.children.push(colspec);
        }
        let build_rows = |rows: Vec<Vec<Node>>| -> Vec<Node> {
            rows.into_iter()
                .map(|entries| {
                    let mut row = Node::elem(kinds::ROW, span);
                    row.children = entries;
                    row
                })
                .collect()
        };
        if !head.is_empty() {
            let mut thead = Node::elem(kinds::THEAD, span);
            thead.children = build_rows(head);
            tgroup.children.push(thead);
        }
        let mut tbody = Node::elem(kinds::TBODY, span);
        tbody.children = build_rows(body);
        tgroup.children.push(tbody);
        table.children.push(tgroup);
        table
    }

    /// topic + sidebar (body.py BasePseudoSection:21-96).
    fn run_pseudo_section(
        &mut self,
        kind: &'static str,
        input: DirectiveInput<'_>,
        out: &mut Vec<Node>,
    ) {
        // Sidebar's own pre-checks run before the shared context check
        // (body.py:88-96).
        if kind == "sidebar" {
            if self.nested_node_kind == Some("sidebar") {
                out.push(self.directive_run_error(
                    &format!(
                        "The \"{}\" directive may not be used within a sidebar element.",
                        input.name
                    ),
                    input.span.source,
                    input.lineno,
                    input.rawsource,
                ));
                return;
            }
            if opt_get(&input.options, "subtitle").is_some() && input.arguments.is_empty() {
                out.push(self.directive_run_error(
                    "The \"subtitle\" option may not be used without a title.",
                    input.span.source,
                    input.lineno,
                    input.rawsource,
                ));
                return;
            }
        }
        // BasePseudoSection context check: allowed parents are the document
        // root, sections, and sidebars (body.py:33-40).
        if let Some(parent) = self.nested_node_kind {
            if parent != "sidebar" {
                out.push(self.directive_run_error(
                    &format!(
                        "The \"{}\" directive may not be used within topics or body elements.",
                        input.name
                    ),
                    input.span.source,
                    input.lineno,
                    input.rawsource,
                ));
                return;
            }
        }
        if input.content.is_empty() {
            out.push(self.directive_content_error(
                input.name,
                input.span.source,
                input.lineno,
                input.rawsource,
            ));
            return;
        }
        let mut node = Node::elem(kind, input.span);
        if let Some(OptVal::StrList(classes)) = opt_get(&input.options, "class") {
            node.attrs.classes.extend(classes.iter().cloned());
        }
        let mut title_messages: Vec<Node> = Vec::new();
        if let Some(title_text) = input.arguments.first() {
            let inline = self.inline(title_text, input.span, input.lineno);
            let mut title = Node::elem(kinds::TITLE, input.span);
            title.children = inline.nodes;
            node.children.push(title);
            title_messages.extend(inline.messages);
            if let Some(OptVal::Str(subtitle_text)) = opt_get(&input.options, "subtitle") {
                let sub_inline = self.inline(subtitle_text, input.span, input.lineno);
                let mut subtitle = Node::elem(kinds::SUBTITLE, input.span);
                subtitle.children = sub_inline.nodes;
                node.children.push(subtitle);
                title_messages.extend(sub_inline.messages);
            }
        }
        node.children.append(&mut title_messages);
        self.directive_add_name(
            &mut node,
            &input.options,
            input.span.source,
            input.lineno,
            out,
        );
        let content = self.parse_nested(&input.content, kind);
        node.children.extend(content);
        out.push(node);
    }

    /// rubric (body.py:240-254): inline children, no paragraph wrapper,
    /// inline messages as siblings after the node.
    fn run_rubric(&mut self, input: DirectiveInput<'_>, out: &mut Vec<Node>) {
        let inline = self.inline(&input.arguments[0], input.span, input.lineno);
        let mut node = Node::elem("rubric", input.span);
        node.children = inline.nodes;
        if let Some(OptVal::StrList(classes)) = opt_get(&input.options, "class") {
            node.attrs.classes.extend(classes.iter().cloned());
        }
        self.directive_add_name(
            &mut node,
            &input.options,
            input.span.source,
            input.lineno,
            out,
        );
        out.push(node);
        out.extend(inline.messages);
    }

    /// epigraph / highlights / pull-quote (body.py:257-283): standard
    /// block-quote elements, each block_quote stamped with the class.
    fn run_quote_class(
        &mut self,
        class: &'static str,
        input: DirectiveInput<'_>,
        out: &mut Vec<Node>,
    ) {
        if input.content.is_empty() {
            out.push(self.directive_content_error(
                input.name,
                input.span.source,
                input.lineno,
                input.rawsource,
            ));
            return;
        }
        let mut elements = self.block_quote_elements(&input.content, input.span);
        for el in &mut elements {
            if el.kind == kinds::BLOCK_QUOTE {
                el.attrs.classes.push(class.to_string());
            }
        }
        out.extend(elements);
    }

    /// compound (body.py:286-301).
    fn run_compound(&mut self, input: DirectiveInput<'_>, out: &mut Vec<Node>) {
        if input.content.is_empty() {
            out.push(self.directive_content_error(
                input.name,
                input.span.source,
                input.lineno,
                input.rawsource,
            ));
            return;
        }
        let mut node = Node::elem("compound", input.span);
        if let Some(OptVal::StrList(classes)) = opt_get(&input.options, "class") {
            node.attrs.classes.extend(classes.iter().cloned());
        }
        self.directive_add_name(
            &mut node,
            &input.options,
            input.span.source,
            input.lineno,
            out,
        );
        let content = self.parse_nested(&input.content, "compound");
        node.children.extend(content);
        out.push(node);
    }

    /// container (body.py:304-329): classes come from the ARGUMENT.
    fn run_container(&mut self, input: DirectiveInput<'_>, out: &mut Vec<Node>) {
        if input.content.is_empty() {
            out.push(self.directive_content_error(
                input.name,
                input.span.source,
                input.lineno,
                input.rawsource,
            ));
            return;
        }
        let mut classes: Vec<String> = Vec::new();
        if let Some(arg) = input.arguments.first() {
            match convert_option(Conv::ClassOption, Some(arg)) {
                Ok(OptVal::StrList(list)) => classes = list,
                _ => {
                    out.push(self.directive_run_error(
                        &format!(
                            "Invalid class attribute value for \"{}\" directive: \"{}\".",
                            input.name, arg
                        ),
                        input.span.source,
                        input.lineno,
                        input.rawsource,
                    ));
                    return;
                }
            }
        }
        let mut node = Node::elem("container", input.span);
        node.attrs.classes.extend(classes);
        self.directive_add_name(
            &mut node,
            &input.options,
            input.span.source,
            input.lineno,
            out,
        );
        let content = self.parse_nested(&input.content, "container");
        node.children.extend(content);
        out.push(node);
    }

    /// parsed-literal (body.py:132-146): full inline parse inside a
    /// whitespace-preserving literal_block; messages follow the node.
    fn run_parsed_literal(&mut self, input: DirectiveInput<'_>, out: &mut Vec<Node>) {
        if input.content.is_empty() {
            out.push(self.directive_content_error(
                input.name,
                input.span.source,
                input.lineno,
                input.rawsource,
            ));
            return;
        }
        let text = self.join_lines(&input.content);
        let inline = self.inline(&text, input.span, input.lineno);
        let mut node = Node::elem(kinds::LITERAL_BLOCK, input.span);
        node.set("xml:space", AttrValue::Str("preserve".to_string()));
        node.children = inline.nodes;
        if let Some(OptVal::StrList(classes)) = opt_get(&input.options, "class") {
            node.attrs.classes.extend(classes.iter().cloned());
        }
        self.directive_add_name(
            &mut node,
            &input.options,
            input.span.source,
            input.lineno,
            out,
        );
        out.push(node);
        out.extend(inline.messages);
    }

    /// DirectiveError-style message (raised by a directive's own run()):
    /// message text VERBATIM — no 'Error in "X" directive:' prefix — plus
    /// the raw block as a literal_block child (states.py:2287-2291).
    fn directive_run_message(
        &self,
        level: u8,
        text: &str,
        source: u16,
        lineno: u32,
        rawsource: &str,
    ) -> Node {
        messages::with_literal(self.msg(level, text, source, lineno), rawsource)
    }

    fn directive_run_error(&self, text: &str, source: u16, lineno: u32, rawsource: &str) -> Node {
        self.directive_run_message(messages::ERROR, text, source, lineno, rawsource)
    }

    /// assert_has_content() (rst/__init__.py:370-377).
    fn directive_content_error(
        &self,
        name: &str,
        source: u16,
        lineno: u32,
        rawsource: &str,
    ) -> Node {
        self.directive_run_error(
            &format!("Content block expected for the \"{name}\" directive; none found."),
            source,
            lineno,
            rawsource,
        )
    }

    /// add_name() (rst/__init__.py:379-389): the :name: option registers an
    /// explicit target on the node.
    fn directive_add_name(
        &mut self,
        node: &mut Node,
        options: &[(String, OptVal)],
        source: u16,
        lineno: u32,
        out: &mut Vec<Node>,
    ) {
        if let Some(OptVal::Str(n)) = opt_get(options, "name") {
            node.attrs.names.push(ids::fully_normalize_name(n));
            let source_path = self.sources.arc_path(source);
            let msg = self
                .registry
                .set_id_explicit(node, lineno, &source_path, true, None);
            if let Some(m) = msg {
                out.push(m);
            }
        }
    }

    fn run_admonition(
        &mut self,
        kind: &'static str,
        input: DirectiveInput<'_>,
        out: &mut Vec<Node>,
    ) {
        if input.content.is_empty() {
            out.push(self.directive_content_error(
                input.name,
                input.span.source,
                input.lineno,
                input.rawsource,
            ));
            return;
        }
        let mut node = Node::elem(kind, input.span);
        if let Some(OptVal::StrList(classes)) = opt_get(&input.options, "class") {
            node.attrs.classes.extend(classes.iter().cloned());
        }
        self.directive_add_name(
            &mut node,
            &input.options,
            input.span.source,
            input.lineno,
            out,
        );
        let content = self.parse_nested(&input.content, kind);
        node.children.extend(content);
        out.push(node);
    }

    fn run_generic_admonition(&mut self, input: DirectiveInput<'_>, out: &mut Vec<Node>) {
        if input.content.is_empty() {
            out.push(self.directive_content_error(
                input.name,
                input.span.source,
                input.lineno,
                input.rawsource,
            ));
            return;
        }
        let title_text = input.arguments[0].clone();
        let mut node = Node::elem("admonition", input.span);
        match opt_get(&input.options, "class") {
            Some(OptVal::StrList(classes)) => {
                node.attrs.classes.extend(classes.iter().cloned());
            }
            _ => {
                // Auto class from the title, only without :class:
                // (admonitions.py:44-46).
                node.attrs
                    .classes
                    .push(format!("admonition-{}", ids::make_id(&title_text)));
            }
        }
        self.directive_add_name(
            &mut node,
            &input.options,
            input.span.source,
            input.lineno,
            out,
        );
        let inline = self.inline(&title_text, input.span, input.lineno);
        let mut title = Node::elem(kinds::TITLE, input.span);
        title.children = inline.nodes;
        node.children.push(title);
        for m in inline.messages {
            node.children.push(m);
        }
        let content = self.parse_nested(&input.content, "admonition");
        node.children.extend(content);
        out.push(node);
    }

    fn run_image(&mut self, input: DirectiveInput<'_>, out: &mut Vec<Node>) {
        match self.build_image(&input, out) {
            Ok(node) => out.push(node),
            Err(msg) => out.push(*msg),
        }
    }

    /// images.py Image.run(): builds the image node (possibly wrapped in a
    /// reference); Err carries the system_message. Shared with figure.
    fn build_image(
        &mut self,
        input: &DirectiveInput<'_>,
        out: &mut Vec<Node>,
    ) -> Result<Node, Box<Node>> {
        // Two-stage :align: validation (images.py:53-63): the converter
        // accepted all six values; at body level only horizontal ones are
        // legal, inside a substitution definition only vertical ones. The
        // DirectiveError text CONTAINS its own 'Error in …' lead — the
        // machinery adds no prefix. Two spaces before 'Valid' are
        // docutils-verbatim.
        if let Some(OptVal::Str(align)) = opt_get(&input.options, "align") {
            let in_subst = self.substitution_ctx.is_some();
            let bad = if in_subst {
                matches!(align.as_str(), "left" | "center" | "right")
            } else {
                matches!(align.as_str(), "top" | "middle" | "bottom")
            };
            if bad {
                let (ctx_txt, valid) = if in_subst {
                    (
                        " within a substitution definition",
                        "\"top\", \"middle\", \"bottom\"",
                    )
                } else {
                    ("", "\"left\", \"center\", \"right\"")
                };
                return Err(Box::new(self.directive_run_error(
                    &format!(
                        "Error in \"{}\" directive: \"{}\" is not a valid value for the \"align\" option{}.  Valid values for \"align\" are: {}.",
                        input.name, align, ctx_txt, valid
                    ),
                    input.span.source, input.lineno,
                    input.rawsource,
                )));
            }
        }
        let uri = uri_from_argument(&input.arguments[0]);
        // :target: wraps the image in a reference (images.py:74-93).
        let mut reference: Option<Node> = None;
        if let Some(OptVal::Str(target)) = opt_get(&input.options, "target") {
            let mut node = Node::elem(kinds::REFERENCE, input.span);
            match parse_image_target(target) {
                ImageTarget::Refname { name, refname } => {
                    node.set("name", AttrValue::Str(name));
                    node.set("refname", AttrValue::Str(refname));
                }
                ImageTarget::Refuri(refuri) => {
                    node.set("refuri", AttrValue::Str(refuri));
                }
            }
            reference = Some(node);
        }
        let mut image = Node::elem("image", input.span);
        for (name, val) in &input.options {
            match (name.as_str(), val) {
                ("alt", OptVal::Str(v)) => image.set("alt", AttrValue::Str(v.clone())),
                ("height", OptVal::Str(v)) => image.set("height", AttrValue::Str(v.clone())),
                ("width", OptVal::Str(v)) => image.set("width", AttrValue::Str(v.clone())),
                ("align", OptVal::Str(v)) => image.set("align", AttrValue::Str(v.clone())),
                ("loading", OptVal::Str(v)) => image.set("loading", AttrValue::Str(v.clone())),
                ("scale", OptVal::Int(v)) => image.set("scale", AttrValue::Int(*v)),
                // Arbitrary-precision values carry the exact digit string.
                ("scale", OptVal::Str(v)) => image.set("scale", AttrValue::Str(v.clone())),
                ("class", OptVal::StrList(v)) => {
                    image.attrs.classes.extend(v.iter().cloned());
                }
                // `name`/`target` are consumed by add_name / the
                // reference wrapper.
                _ => {}
            }
        }
        image.set("uri", AttrValue::Str(uri));
        self.directive_add_name(
            &mut image,
            &input.options,
            input.span.source,
            input.lineno,
            out,
        );
        Ok(match reference {
            Some(mut r) => {
                r.children.push(image);
                r
            }
            None => image,
        })
    }

    /// figure (images.py:110-186), plus sphinx's override (patches.py:33-56)
    /// which moves `:name:` from the inner image onto the figure itself.
    fn run_figure(&mut self, input: DirectiveInput<'_>, out: &mut Vec<Node>) {
        // sphinx pops `name` before delegating to docutils, so the image
        // never sees it, and re-applies it to the figure node afterwards —
        // but only on the success path (a figure returned *with* an error
        // node, or an error alone, keeps no name at all).
        let name_on_figure = self.sphinx;
        let image_input = DirectiveInput {
            name: input.name,
            arguments: input.arguments.clone(),
            options: input
                .options
                .iter()
                .filter(|(n, _)| {
                    !matches!(n.as_str(), "figwidth" | "figclass" | "align")
                        && !(name_on_figure && n == "name")
                })
                .cloned()
                .collect(),
            content: Vec::new(),
            span: input.span,
            lineno: input.lineno,
            rawsource: input.rawsource,
        };
        let image_node = match self.build_image(&image_input, out) {
            Ok(n) => n,
            Err(msg) => {
                // Inner image error short-circuits: no <figure> at all.
                out.push(*msg);
                return;
            }
        };
        let mut figure = Node::elem("figure", input.span);
        match opt_get(&input.options, "figwidth") {
            // ':figwidth: image' needs PIL, which the oracle environment
            // lacks: silent no-op (images.py:150-159).
            Some(OptVal::Str(w)) if w == "image" => {}
            Some(OptVal::Str(w)) => figure.set("width", AttrValue::Str(w.clone())),
            _ => {}
        }
        if let Some(OptVal::StrList(cls)) = opt_get(&input.options, "figclass") {
            figure.attrs.classes.extend(cls.iter().cloned());
        }
        if let Some(OptVal::Str(a)) = opt_get(&input.options, "align") {
            figure.set("align", AttrValue::Str(a.clone()));
        }
        figure.children.push(image_node);
        if !input.content.is_empty() {
            let children = self.parse_nested(&input.content, "figure");
            let mut caption_done = false;
            let mut legend_children: Vec<Node> = Vec::new();
            for child in children {
                if caption_done {
                    legend_children.push(child);
                    continue;
                }
                if child.kind == kinds::TARGET {
                    figure.children.push(child);
                } else if child.kind == kinds::PARAGRAPH {
                    let mut caption = Node::elem("caption", input.span);
                    caption.children = child.children;
                    figure.children.push(caption);
                    caption_done = true;
                } else if child.kind == kinds::COMMENT && child.children.is_empty() {
                    caption_done = true;
                } else {
                    // Unlike other directives, the figure node is emitted
                    // BEFORE the error (images.py:176-181).
                    out.push(figure);
                    out.push(self.directive_run_error(
                        "Figure caption must be a paragraph or empty comment.",
                        input.span.source,
                        input.lineno,
                        input.rawsource,
                    ));
                    return;
                }
            }
            if !legend_children.is_empty() {
                let mut legend = Node::elem("legend", input.span);
                legend.children = legend_children;
                figure.children.push(legend);
            }
        }
        if name_on_figure {
            // After the nested parse, exactly where sphinx calls it — the
            // caption's own targets are registered first.
            self.directive_add_name(
                &mut figure,
                &input.options,
                input.span.source,
                input.lineno,
                out,
            );
        }
        out.push(figure);
    }

    /// code (body.py:149-211). The parity oracle runs docutils WITHOUT
    /// Pygments: a language argument fails the whole directive with a
    /// WARNING; language-less code emits a plain classes="code" literal.
    fn run_code(&mut self, input: DirectiveInput<'_>, out: &mut Vec<Node>) {
        if input.content.is_empty() {
            out.push(self.directive_content_error(
                input.name,
                input.span.source,
                input.lineno,
                input.rawsource,
            ));
            return;
        }
        if !input.arguments.is_empty() {
            out.push(self.directive_run_message(
                messages::WARNING,
                "Cannot analyze code. Pygments package not found.",
                input.span.source,
                input.lineno,
                input.rawsource,
            ));
            return;
        }
        let number_lines = match opt_get(&input.options, "number-lines") {
            Some(OptVal::Str(v)) => {
                let raw = if v.is_empty() { "1" } else { v.as_str() };
                match py_int(raw) {
                    Some(n) => Some(n),
                    None => {
                        out.push(self.directive_run_error(
                            ":number-lines: with non-integer start value",
                            input.span.source,
                            input.lineno,
                            input.rawsource,
                        ));
                        return;
                    }
                }
            }
            _ => None,
        };
        let code_lines: Vec<String> = input
            .content
            .iter()
            .map(|l| self.sources.line_text(*l).to_string())
            .collect();
        let mut node = Node::elem(kinds::LITERAL_BLOCK, input.span);
        node.attrs.classes.push("code".to_string());
        if let Some(OptVal::StrList(classes)) = opt_get(&input.options, "class") {
            node.attrs.classes.extend(classes.iter().cloned());
        }
        node.set("xml:space", AttrValue::Str("preserve".to_string()));
        match number_lines {
            Some(start) => {
                // NumberLines (docutils/utils/code_analyzer.py): a padded
                // 'ln' inline before every line.
                let endline = start.saturating_add(input.content.len() as i64);
                let width = endline.to_string().len();
                for (i, line) in code_lines.iter().enumerate() {
                    let lineno = start.saturating_add(i as i64);
                    let mut ln = Node::elem("inline", input.span);
                    ln.attrs.classes.push("ln".to_string());
                    ln.children
                        .push(Node::text_node(format!("{lineno:>width$} "), input.span));
                    node.children.push(ln);
                    let text = if i + 1 == code_lines.len() {
                        (*line).to_string()
                    } else {
                        format!("{line}\n")
                    };
                    node.children.push(Node::text_node(text, input.span));
                }
            }
            None => {
                node.children
                    .push(Node::text_node(code_lines.join("\n"), input.span));
            }
        }
        self.directive_add_name(
            &mut node,
            &input.options,
            input.span.source,
            input.lineno,
            out,
        );
        out.push(node);
    }

    /// math (body.py:214-237): blank-line-separated blocks become sibling
    /// math_block nodes; :name: only lands on the first (options.pop).
    fn run_math(&mut self, input: DirectiveInput<'_>, out: &mut Vec<Node>) {
        if input.content.is_empty() {
            out.push(self.directive_content_error(
                input.name,
                input.span.source,
                input.lineno,
                input.rawsource,
            ));
            return;
        }
        let joined = self.join_lines(&input.content);
        let mut named = false;
        for block in joined.split("\n\n") {
            if block.is_empty() {
                continue;
            }
            let mut node = Node::elem("math_block", input.span);
            node.set("xml:space", AttrValue::Str("preserve".to_string()));
            if let Some(OptVal::StrList(classes)) = opt_get(&input.options, "class") {
                node.attrs.classes.extend(classes.iter().cloned());
            }
            node.children.push(Node::text_node(block, input.span));
            if !named {
                self.directive_add_name(
                    &mut node,
                    &input.options,
                    input.span.source,
                    input.lineno,
                    out,
                );
                named = true;
            }
            out.push(node);
        }
    }

    /// raw (misc.py:270-354).
    fn run_raw(&mut self, input: DirectiveInput<'_>, out: &mut Vec<Node>) {
        let has_file = opt_get(&input.options, "file").is_some();
        let has_url = opt_get(&input.options, "url").is_some();
        let text: String;
        let mut source_attr: Option<String> = None;
        if !input.content.is_empty() {
            if has_file || has_url {
                out.push(self.directive_run_error(
                    &format!(
                        "\"{}\" directive may not both specify an external file and have content.",
                        input.name
                    ),
                    input.span.source,
                    input.lineno,
                    input.rawsource,
                ));
                return;
            }
            text = self.join_lines(&input.content);
        } else if has_file {
            if has_url {
                out.push(self.directive_run_error(
                    &format!(
                        "The \"file\" and \"url\" options may not be simultaneously specified for the \"{}\" directive.",
                        input.name
                    ),
                    input.span.source, input.lineno,
                    input.rawsource,
                ));
                return;
            }
            let Some(OptVal::Str(path)) = opt_get(&input.options, "file") else {
                unreachable!("file option is Path-converted");
            };
            let base = std::path::Path::new(self.sources.path(0))
                .parent()
                .map(|p| p.to_path_buf())
                .unwrap_or_default();
            let full = base.join(path);
            match std::fs::read_to_string(&full) {
                Ok(t) => {
                    // docutils strips ONE trailing newline via rstrip
                    // hazard; keep verbatim minus trailing newline.
                    text = t.trim_end_matches('\n').to_string();
                    source_attr = Some(path.clone());
                }
                Err(_) => {
                    out.push(self.directive_run_message(
                        messages::SEVERE,
                        &format!(
                            "Problems with \"{}\" directive path:\nInputError: [Errno 2] No such file or directory: {}.",
                            input.name,
                            py_repr(Some(path))
                        ),
                        input.span.source, input.lineno,
                        input.rawsource,
                    ));
                    return;
                }
            }
        } else if has_url {
            // URL fetching is out of parse-layer scope; the corpus only
            // pins the mutual-exclusivity errors above.
            out.push(self.directive_run_message(
                messages::SEVERE,
                &format!(
                    "Problems with \"{}\" directive URL: fetching is not supported.",
                    input.name
                ),
                input.span.source,
                input.lineno,
                input.rawsource,
            ));
            return;
        } else {
            out.push(self.directive_content_error(
                input.name,
                input.span.source,
                input.lineno,
                input.rawsource,
            ));
            return;
        }
        let format = input.arguments[0]
            .to_lowercase()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        let mut node = Node::elem("raw", input.span);
        node.set("format", AttrValue::Str(format));
        node.set("xml:space", AttrValue::Str("preserve".to_string()));
        if let Some(src) = source_attr {
            node.set("source", AttrValue::Str(src));
        }
        if let Some(OptVal::StrList(classes)) = opt_get(&input.options, "class") {
            node.attrs.classes.extend(classes.iter().cloned());
        }
        node.children.push(Node::text_node(text, input.span));
        out.push(node);
    }

    /// line-block directive (body.py:99-129): same tree as `|` syntax.
    fn run_line_block(&mut self, input: DirectiveInput<'_>, out: &mut Vec<Node>) {
        if input.content.is_empty() {
            out.push(self.directive_content_error(
                input.name,
                input.span.source,
                input.lineno,
                input.rawsource,
            ));
            return;
        }
        let mut resolved: Vec<(usize, Vec<Node>)> = Vec::with_capacity(input.content.len());
        let mut lb_messages: Vec<Node> = Vec::new();
        let mut prev_depth = 0usize;
        for l in &input.content {
            if l.is_blank() {
                resolved.push((prev_depth, Vec::new()));
                continue;
            }
            let depth = l.indent();
            prev_depth = depth;
            let text = self.sources.line_text(*l).trim().to_string();
            let inline = self.inline(&text, input.span, l.lineno);
            lb_messages.extend(inline.messages);
            resolved.push((depth, inline.nodes));
        }
        let mut block = build_line_block(&mut resolved, input.span, 0);
        if let Some(OptVal::StrList(classes)) = opt_get(&input.options, "class") {
            block.attrs.classes.extend(classes.iter().cloned());
        }
        self.directive_add_name(
            &mut block,
            &input.options,
            input.span.source,
            input.lineno,
            out,
        );
        out.push(block);
        out.append(&mut lb_messages);
    }

    /// class (misc.py:434-469): with content, classes apply directly to
    /// every top-level child; without, a pending node is emitted for the
    /// ClassAttribute transform.
    fn run_class(&mut self, input: DirectiveInput<'_>, out: &mut Vec<Node>) {
        let class_values = match convert_option(Conv::ClassOption, Some(&input.arguments[0])) {
            Ok(OptVal::StrList(list)) => list,
            _ => {
                out.push(self.directive_run_error(
                    &format!(
                        "Invalid class attribute value for \"{}\" directive: \"{}\".",
                        input.name, input.arguments[0]
                    ),
                    input.span.source,
                    input.lineno,
                    input.rawsource,
                ));
                return;
            }
        };
        if !input.content.is_empty() {
            let mut children = self.parse_nested(&input.content, "element");
            for child in &mut children {
                child.attrs.classes.extend(class_values.iter().cloned());
            }
            out.extend(children);
        } else if self.sphinx {
            // Sphinx's read phase runs ClassAttribute; the pending node
            // never survives into the doctree — stamp the next sibling.
            self.pending_classes = Some(class_values);
        } else {
            let mut pending = Node::elem("pending", input.span);
            let details = format!(
                ".. internal attributes:\n     .transform: docutils.transforms.misc.ClassAttribute\n     .details:\n       class: [{}]\n       directive: {}",
                class_values
                    .iter()
                    .map(|c| py_repr(Some(c)))
                    .collect::<Vec<_>>()
                    .join(", "),
                py_repr(Some(input.name)),
            );
            pending.children.push(Node::text_node(details, input.span));
            out.push(pending);
        }
    }

    /// `.. |name| directive::` substitution definitions
    /// (states.py:2140-2217 + the SubstitutionDef state 2806-2829).
    /// Returns true when consumed; false = malformed marker — the caller
    /// falls through to the comment path with `construct_error` set.
    fn parse_substitution_def(
        &mut self,
        lines: &[LineRec],
        pos: &mut usize,
        rest: &str,
        out: &mut Vec<Node>,
        construct_error: &mut Option<Node>,
    ) -> bool {
        let start = *pos;
        let marker_rec = lines[start];
        let lineno = marker_rec.lineno;
        let msg_source = marker_rec.source;
        let (block, consumed, _indent, _term) = indented_block(lines, start + 1);
        let span = self.span_of(lines, start, start + consumed);
        // blocktext for message literals: the raw marker line + raw block
        // (trailing blanks already trimmed by indented_block).
        let mut blocktext = self.sources.line_text(marker_rec).to_string();
        for l in &lines[start + 1..start + 1 + consumed] {
            blocktext.push('\n');
            blocktext.push_str(self.sources.line_text(*l));
        }

        // Marker scan: `|name|` possibly joined across adjacent block lines
        // (states.py:2151-2160). Failure at end-of-block = MarkupError.
        let mut acc: String = rest.to_string();
        let mut used = 0usize;
        let marker = loop {
            if let Some(m) = match_substitution_marker(&acc) {
                break m;
            }
            if used >= block.len() || block[used].is_blank() {
                *construct_error = Some(self.msg(
                    messages::WARNING,
                    "malformed substitution definition.",
                    msg_source,
                    lineno,
                ));
                return false;
            }
            acc.push(' ');
            acc.push_str(self.sources.line_text(block[used]).trim());
            used += 1;
        };
        *pos = start + 1 + consumed;
        // Remainder after the marker lives on ONE physical line: `rest`
        // when the marker was single-line, else the last joined block line.
        let (rem_rec, rem_lineno): (LineRec, u32) = if used == 0 {
            // `rest` is the marker line's text past `.. ` and its leading
            // spaces; recover its byte offset within the line to re-wrap.
            let rest_offset = self.sources.line_text(marker_rec).len() - rest.len();
            (
                self.rewrap_from(marker_rec, rest_offset + marker.remainder_start),
                lineno,
            )
        } else {
            let last = block[used - 1];
            let last_text = self.sources.line_text(last);
            let trimmed = last_text.trim();
            let seg_start = acc.len() - trimmed.len();
            let within = marker.remainder_start.saturating_sub(seg_start);
            let base = last_text.len() - last_text.trim_start().len();
            (self.rewrap_from(last, base + within), last.lineno)
        };
        let mut content_block: Vec<LineRec> = block[used..].to_vec();

        let subname_ws = ids::whitespace_normalize_name(&marker.name);
        // Missing contents (states.py:2168-2176).
        if self.sources.line_text(rem_rec).trim().is_empty()
            && content_block.iter().all(|l| l.is_blank())
        {
            out.push(messages::with_literal(
                self.msg(
                    messages::WARNING,
                    &format!(
                        "Substitution definition \"{}\" missing contents.",
                        marker.name
                    ),
                    msg_source,
                    lineno,
                ),
                &blocktext,
            ));
            self.warn_explicit_markup_end(lines, *pos, out);
            return true;
        }

        let mut subst = Node::elem("substitution_definition", span);
        subst.attrs.names.push(subname_ws.clone());

        // Locate the embedded-directive line: the marker remainder, else
        // the first non-blank content line (hanging-indent form).
        // raw_content_from tracks the BLOCK index where the directive's
        // continuation lines begin, for rawsource reconstruction with
        // original indentation (docutils strip_indent=False).
        let mut raw_content_from = used;
        let (dline, dlineno) = if !self.sources.line_text(rem_rec).trim().is_empty() {
            let rem_text = self.sources.line_text(rem_rec);
            let spaces = rem_text.len() - rem_text.trim_start_matches(' ').len();
            (self.rewrap_from(rem_rec, spaces), rem_lineno)
        } else {
            while content_block.first().map(|l| l.is_blank()).unwrap_or(false) {
                content_block.remove(0);
                raw_content_from += 1;
            }
            let first = content_block.remove(0);
            raw_content_from += 1;
            let dedent = first.indent();
            (first.dedented(dedent), first.lineno)
        };
        // Embedded directive marker: simplename + `::` + (space|EOL) — NO
        // optional space before `::` (SubstitutionDef state pattern).
        let mut produced: Vec<Node> = Vec::new();
        let dline_src = self.sources.arc(dline.source);
        let dline_text = dline.slice(&dline_src);
        if let Some((dname, dfirst_rest)) = match_embedded_directive(dline_text) {
            let dblock = dedent_by_min(&content_block);
            // rawsource with ORIGINAL indentation (the nested state
            // machine's lines are strip_indent=False; fixture-verified).
            let embedded_raw = {
                let mut raw = dline_text.to_string();
                for l in &lines[start + 1 + raw_content_from..start + 1 + consumed] {
                    raw.push('\n');
                    raw.push_str(self.sources.line_text(*l));
                }
                raw
            };
            let dfirst = {
                let t = dfirst_rest.trim_start_matches(' ');
                let offset = dline_text.len() - t.len();
                self.rewrap_from(dline, offset)
            };
            self.substitution_ctx = Some(SubstCtx::default());
            let saved_kind = self.nested_node_kind.replace("substitution_definition");
            self.run_directive_core(
                &dname,
                dfirst,
                &dblock,
                &embedded_raw,
                dlineno,
                span,
                vec![("alt".to_string(), OptVal::Str(subname_ws.clone()))],
                &mut produced,
            );
            self.nested_node_kind = saved_kind;
            let ctx = self.substitution_ctx.take().unwrap_or_default();
            if ctx.ltrim {
                subst.set("ltrim", AttrValue::Int(1));
            }
            if ctx.rtrim {
                subst.set("rtrim", AttrValue::Int(1));
            }
        }
        // Hoist non-inline children to the parent, in document order
        // (states.py:2184-2191); inline/Text stay in the definition.
        for n in produced {
            if n.kind == kinds::TEXT || is_inline_kind(n.kind) {
                subst.children.push(n);
            } else {
                out.push(n);
            }
        }
        // Problematic content check (states.py:2194-2201).
        if tree_any(&subst, &|n| n.kind == kinds::PROBLEMATIC) {
            let mut msg = self.msg(
                messages::ERROR,
                "Problematic content in substitution definition",
                msg_source,
                lineno,
            );
            let mut lb = Node::elem(kinds::LITERAL_BLOCK, Span::ZERO);
            lb.set("xml:space", AttrValue::Str("preserve".to_string()));
            lb.children.push(Node::text_node(&blocktext, Span::ZERO));
            msg.children.push(lb);
            let mut bq = Node::elem(kinds::BLOCK_QUOTE, Span::ZERO);
            let mut para = Node::elem(kinds::PARAGRAPH, Span::ZERO);
            para.children = std::mem::take(&mut subst.children);
            bq.children.push(para);
            msg.children.push(bq);
            out.push(msg);
            self.warn_explicit_markup_end(lines, *pos, out);
            return true;
        }
        // Disallowed content (states.py:2219-2227).
        if let Some(phrase) = find_disallowed_in_substitution(&subst) {
            out.push(messages::with_literal(
                self.msg(
                    messages::ERROR,
                    &format!("{phrase} are not supported in a substitution definition."),
                    msg_source,
                    lineno,
                ),
                &blocktext,
            ));
            self.warn_explicit_markup_end(lines, *pos, out);
            return true;
        }
        // Empty or invalid (states.py:2203-2210).
        if subst.children.is_empty() {
            out.push(messages::with_literal(
                self.msg(
                    messages::WARNING,
                    &format!(
                        "Substitution definition \"{}\" empty or invalid.",
                        marker.name
                    ),
                    msg_source,
                    lineno,
                ),
                &blocktext,
            ));
            self.warn_explicit_markup_end(lines, *pos, out);
            return true;
        }
        // note_substitution_def (nodes.py:2056-2073): duplicate names are
        // case-sensitively compared; the error precedes the new node and
        // the OLD node loses its name (post-parse walk).
        if self.substitution_names_seen.contains(&subname_ws) {
            out.push(self.msg(
                messages::ERROR,
                &format!("Duplicate substitution definition name: \"{subname_ws}\"."),
                msg_source,
                lineno,
            ));
            if !self.substitution_dupnames.contains(&subname_ws) {
                self.substitution_dupnames.push(subname_ws.clone());
            }
        } else {
            self.substitution_names_seen.push(subname_ws);
        }
        out.push(subst);
        self.warn_explicit_markup_end(lines, *pos, out);
        true
    }

    fn parse_anonymous_shortcut(
        &mut self,
        lines: &[LineRec],
        pos: &mut usize,
        rest: &str,
        out: &mut Vec<Node>,
    ) {
        let start = *pos;
        let mut consumed = 0usize;
        while lines
            .get(start + 1 + consumed)
            .map(|l| !l.is_blank() && l.indent() > 0)
            .unwrap_or(false)
        {
            consumed += 1;
        }
        let span = self.span_of(lines, start, start + consumed);
        let mut link = rest.trim().to_string();
        for l in &lines[start + 1..start + 1 + consumed] {
            if !link.is_empty() {
                link.push('\n');
            }
            link.push_str(self.sources.line_text(*l).trim());
        }
        *pos = start + 1 + consumed;
        let mut target = Node::elem(kinds::TARGET, span);
        target.set("anonymous", AttrValue::Int(1));
        if !link.is_empty() {
            if let Some(refname) = reference_name_from_link(&link) {
                target.set("refname", AttrValue::Str(refname));
            } else {
                let uri: String = link
                    .chars()
                    .filter(|c| !c.is_whitespace() && *c != '\\')
                    .collect();
                target.set("refuri", AttrValue::Str(uri));
            }
        }
        self.registry.set_id_anonymous(&mut target);
        out.push(target);
        self.warn_explicit_markup_end(lines, *pos, out);
    }
}

// ----------------------------------------------------------------------
// free helpers
// ----------------------------------------------------------------------

/// Field marker: `:name:` where the name may not start with `:`/space,
/// may not end with a space, and interior `:` is allowed unless followed
/// by space, backtick, or EOL. The marker must close with `:` + space/EOL.
/// Returns (raw name, byte index just past the closing colon).
fn field_marker(text: &str) -> Option<(String, usize)> {
    let mut chars = text.char_indices();
    let (_, first) = chars.next()?;
    if first != ':' {
        return None;
    }
    let mut name = String::new();
    let mut prev_char: Option<char> = None;
    let mut it = chars.peekable();
    // reject :: and ": "
    match it.peek() {
        Some((_, ':')) | Some((_, ' ')) | None => return None,
        _ => {}
    }
    while let Some((i, c)) = it.next() {
        match c {
            '\\' => {
                name.push(c);
                if let Some((_, esc)) = it.next() {
                    name.push(esc);
                    prev_char = Some(esc);
                }
            }
            ':' => {
                let next = it.peek().map(|(_, c)| *c);
                match next {
                    None | Some(' ') => {
                        // closing colon; name may not end with a space
                        if prev_char == Some(' ') || name.is_empty() {
                            return None;
                        }
                        return Some((name, i + 1));
                    }
                    Some('`') => return None,
                    _ => {
                        name.push(':');
                        prev_char = Some(':');
                    }
                }
            }
            _ => {
                name.push(c);
                prev_char = Some(c);
            }
        }
    }
    None
}

/// Option-group marker: synonyms split on `, ` (not inside `<>`), each a
/// short (`-x`/`+x` with optional attached/spaced arg) or long
/// (`--name`/`/name` with `=`/space arg) option. Returns the specs plus
/// the description remainder (after 2+ spaces), or None when any synonym
/// is malformed.
#[allow(clippy::type_complexity)]
fn option_group_marker(text: &str) -> Option<(Vec<(String, Option<(String, String)>)>, &str)> {
    // split marker from description at the first run of 2+ spaces
    // OUTSIDE angle brackets
    let mut in_angle = false;
    let mut marker_end = text.len();
    let bytes: Vec<(usize, char)> = text.char_indices().collect();
    let mut k = 0;
    while k < bytes.len() {
        let (i, c) = bytes[k];
        match c {
            '<' => in_angle = true,
            '>' => in_angle = false,
            ' ' if !in_angle && bytes.get(k + 1).map(|(_, c)| *c == ' ').unwrap_or(false) => {
                marker_end = i;
                break;
            }
            _ => {}
        }
        k += 1;
    }
    let marker = &text[..marker_end];
    let desc = text[marker_end..].trim_start();

    // split synonyms on ', ' outside <>
    let mut specs = Vec::new();
    let mut parts: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut in_angle = false;
    let mchars: Vec<char> = marker.chars().collect();
    let mut idx = 0;
    while idx < mchars.len() {
        let c = mchars[idx];
        match c {
            '<' => {
                in_angle = true;
                cur.push(c);
            }
            '>' => {
                in_angle = false;
                cur.push(c);
            }
            ',' if !in_angle && mchars.get(idx + 1) == Some(&' ') => {
                parts.push(std::mem::take(&mut cur));
                idx += 1; // skip the space
            }
            _ => cur.push(c),
        }
        idx += 1;
    }
    parts.push(cur);

    for part in &parts {
        specs.push(parse_one_option(part)?);
    }
    Some((specs, desc))
}

/// One option synonym -> (option_string, Some((delimiter, argument))).
fn parse_one_option(part: &str) -> Option<(String, Option<(String, String)>)> {
    let optarg_ok = |s: &str| -> bool {
        if let Some(inner) = s.strip_prefix('<') {
            return inner.ends_with('>') && !inner[..inner.len() - 1].contains(['<', '>']);
        }
        let mut cs = s.chars();
        matches!(cs.next(), Some(c) if c.is_ascii_alphabetic())
            && cs.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-'))
    };
    if let Some(rest) = part.strip_prefix("--").or_else(|| part.strip_prefix('/')) {
        let prefix = if part.starts_with("--") { "--" } else { "/" };
        // optname [ =|space optarg ]
        let name_end = rest.find([' ', '=']).unwrap_or(rest.len());
        let (name, tail) = rest.split_at(name_end);
        let mut nc = name.chars();
        let name_ok = matches!(nc.next(), Some(c) if c.is_ascii_alphanumeric())
            && nc.all(|c| c.is_ascii_alphanumeric() || matches!(c, '_' | '-'));
        if !name_ok {
            return None;
        }
        if tail.is_empty() {
            return Some((format!("{prefix}{name}"), None));
        }
        let delim = &tail[..1];
        let arg = &tail[1..];
        if !optarg_ok(arg) {
            return None;
        }
        return Some((
            format!("{prefix}{name}"),
            Some((delim.to_string(), arg.to_string())),
        ));
    }
    let rest = part.strip_prefix('-').or_else(|| part.strip_prefix('+'))?;
    let prefix = &part[..1];
    let mut rc = rest.chars();
    let letter = rc.next().filter(|c| c.is_ascii_alphanumeric())?;
    let tail: String = rc.collect();
    if tail.is_empty() {
        return Some((format!("{prefix}{letter}"), None));
    }
    if let Some(arg) = tail.strip_prefix(' ') {
        if !optarg_ok(arg) {
            return None;
        }
        return Some((
            format!("{prefix}{letter}"),
            Some((" ".to_string(), arg.to_string())),
        ));
    }
    if !optarg_ok(&tail) {
        return None;
    }
    Some((format!("{prefix}{letter}"), Some((String::new(), tail))))
}

fn is_grid_table_top(text: &str) -> bool {
    // \+-[-+]+-\+ *$  (minimum "+-x-+": 5 chars)
    let t = text.trim_end();
    let chars: Vec<char> = t.chars().collect();
    chars.len() >= 5
        && chars[0] == '+'
        && chars[chars.len() - 1] == '+'
        && chars[1] == '-'
        && chars[chars.len() - 2] == '-'
        && chars[1..chars.len() - 1]
            .iter()
            .all(|c| matches!(c, '-' | '+'))
}

fn is_grid_head_sep(text: &str) -> bool {
    // \+=[=+]+=\+ *$  (minimum 5 chars)
    let t = text.trim_end();
    let chars: Vec<char> = t.chars().collect();
    chars.len() >= 5
        && chars[0] == '+'
        && chars[chars.len() - 1] == '+'
        && chars[1] == '='
        && chars[chars.len() - 2] == '='
        && chars[1..chars.len() - 1]
            .iter()
            .all(|c| matches!(c, '=' | '+'))
}

/// `=+[ =]*$` — a candidate simple-table border (incl. solid runs).
fn is_simple_table_border(text: &str) -> bool {
    let t = text.trim_end();
    !t.is_empty() && t.starts_with('=') && t.chars().all(|c| matches!(c, '=' | ' '))
}

fn is_simple_table_top(text: &str) -> bool {
    // =+( +=+)+ *$  (two or more '=' runs)
    let t = text.trim_end();
    if t.is_empty() {
        return false;
    }
    let mut runs = 0;
    let mut in_run = false;
    for c in t.chars() {
        match c {
            '=' => {
                if !in_run {
                    runs += 1;
                    in_run = true;
                }
            }
            ' ' => in_run = false,
            _ => return false,
        }
    }
    runs >= 2
}

/// Byte offsets per DISPLAY column (east-asian wide chars occupy two
/// columns; the second maps to the char's end so mid-char boundaries
/// exclude it — matching docutils' double-width padding behavior).
fn display_byte_index(text: &str) -> Vec<usize> {
    let mut index = Vec::with_capacity(text.len() + 1);
    for (b, c) in text.char_indices() {
        index.push(b);
        if unicode_width::UnicodeWidthChar::width(c).unwrap_or(1) == 2 {
            index.push(b + c.len_utf8());
        }
    }
    index.push(text.len());
    index
}

/// Slice by DISPLAY column range (byte-safe; mid-wide-char boundaries
/// clamp to char edges).
fn display_slice(text: &str, from: usize, to: usize) -> &str {
    let (start, end) = display_range(text, from, to);
    &text[start..end]
}

/// The byte range [`display_slice`] would take — for carving a `LineRec`
/// sub-view rather than a borrowed slice.
fn display_range(text: &str, from: usize, to: usize) -> (usize, usize) {
    let index = display_byte_index(text);
    let n = index.len() - 1;
    let start = index[from.min(n)];
    let end = index[to.min(n)];
    if start >= end {
        (start, start)
    } else {
        (start, end)
    }
}

/// Trace one grid cell from its top-left '+': returns (bottom, right,
/// column separators seen, row separators seen).
#[allow(clippy::type_complexity)]
fn trace_cell(
    grid: &[Vec<char>],
    top: usize,
    left: usize,
) -> Option<(usize, usize, Vec<usize>, Vec<usize>)> {
    let at =
        |r: usize, c: usize| -> Option<char> { grid.get(r).and_then(|row| row.get(c)).copied() };
    let width = grid.get(top).map(|r| r.len()).unwrap_or(0);
    // scan right along the top border
    let mut c = left + 1;
    let mut top_corners = Vec::new();
    loop {
        match at(top, c) {
            Some('+') => top_corners.push(c),
            Some('-') => {}
            _ => break,
        }
        c += 1;
        if c > width + 1 {
            break;
        }
    }
    for &right in &top_corners {
        // scan down the right edge
        let mut r = top + 1;
        let mut right_corners = Vec::new();
        loop {
            match at(r, right) {
                Some('+') => right_corners.push(r),
                Some('|') => {}
                _ => break,
            }
            r += 1;
            if r > grid.len() {
                break;
            }
        }
        for &bottom in &right_corners {
            // scan left along the bottom, then up the left edge
            let mut ok = true;
            let mut cseps = vec![left, right];
            for cc in left + 1..right {
                match at(bottom, cc) {
                    Some('+') => cseps.push(cc),
                    Some('-') => {}
                    _ => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                continue;
            }
            let mut rseps = vec![top, bottom];
            for rr in top + 1..bottom {
                match at(rr, left) {
                    Some('+') => rseps.push(rr),
                    Some('|') => {}
                    _ => {
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                continue;
            }
            return Some((bottom, right, cseps, rseps));
        }
    }
    None
}

/// Directive marker on the text after `.. `: `name[ ]?::` then space+rest
/// or EOL (probe-verified: at most ONE space before `::`; dangling
/// separators or `:` alone fall through to comment).
fn directive_marker(rest: &str) -> Option<(String, &str)> {
    let chars: Vec<char> = rest.chars().collect();
    let name_len = match_simplename_chars(&chars, 0)?;
    let mut j = name_len;
    if chars.get(j) == Some(&' ') {
        j += 1;
    }
    if chars.get(j) != Some(&':') || chars.get(j + 1) != Some(&':') {
        return None;
    }
    let after = j + 2;
    match chars.get(after) {
        None => {}
        Some(' ') => {}
        _ => return None,
    }
    let name: String = chars[..name_len].iter().collect();
    // byte offset of the remainder after ":: "
    let byte_after: usize = rest
        .char_indices()
        .nth(after + 1)
        .map(|(b, _)| b)
        .unwrap_or(rest.len());
    Some((name, &rest[byte_after..]))
}

#[derive(Clone, Copy)]
enum DirectiveKind {
    /// note/warning/... : content-only, node kind = tagname.
    Admonition(&'static str),
    /// `.. admonition:: Title` with required title argument.
    GenericAdmonition,
    /// `.. image:: uri` (images.py Image).
    Image,
    /// topic / sidebar (body.py BasePseudoSection).
    PseudoSection(&'static str),
    Rubric,
    /// epigraph / highlights / pull-quote: block_quote + class.
    QuoteClass(&'static str),
    Compound,
    Container,
    ParsedLiteral,
    Figure,
    Code,
    MathBlock,
    Raw,
    LineBlockDir,
    ClassDir,
    RstTable,
    CsvTable,
    ListTable,
    Replace,
    UnicodeDir,
    DateDir,
    /// sphinx toctree (sphinx/directives/other.py TocTree).
    Toctree,
    /// versionadded/versionchanged/deprecated/versionremoved:
    /// (type name, label class, lead-in format).
    VersionChange(&'static (&'static str, &'static str, &'static str)),
    SeeAlso,
    /// sphinx code-block/sourcecode (sphinx/directives/code.py).
    SphinxCodeBlock,
    Highlight,
    Only,
    SphinxMath,
    IndexDir,
    HList,
    Glossary,
    /// `.. describe::`/`.. object::`, `.. envvar::`, `.. confval::`,
    /// `.. option::`/`.. cmdoption::` — sphinx `ObjectDescription.run`
    /// (`directives/__init__.py:183-314`) with a per-directive
    /// `handle_signature`/`add_target_and_index`.
    ObjectDesc(ObjectDescKind),
    /// The `py:*` object-description family (`sphinx/domains/python`),
    /// running through the same `ObjectDescription.run` anatomy with the
    /// py-domain `handle_signature`/`add_target_and_index`/`before_content`
    /// overrides ([PY §1.3-1.5, §7]).
    PyObjectDesc(PyDirective),
    /// `.. py:module::` — a plain `SphinxDirective`, NOT an
    /// ObjectDescription (`domains/python/__init__.py:473-536`).
    PyModule,
    /// `.. py:currentmodule::` — pure ref_context state, emits nothing
    /// (`__init__.py:539-556`).
    PyCurrentModule,
    /// `.. program::` (`domains/std/__init__.py:333-348`).
    ProgramDir,
    /// `.. default-domain::` (`directives/__init__.py:353-366`).
    DefaultDomainDir,
    /// Test-only exercise of the [`SpliceRequest`] channel: content lines
    /// become a spliced source named by the argument. No shipping
    /// directive produces a splice until T12's `include`.
    #[cfg(test)]
    TestSplice,
}

/// Which Python object a `py:*` object-description directive describes.
/// Values are the sphinx directive classes AFTER name aliasing [PY §1.1]:
/// `py:classmethod`/`py:staticmethod`/`py:decoratormethod` are `Method`,
/// `py:decorator` is `Function` — their `run()` rewrites `self.name`
/// before the base run partitions it, so the desc's objtype is the
/// aliased kind's (trap 13).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PyObjectKind {
    Function,
    Data,
    Class,
    Exception,
    Method,
    Attribute,
    Property,
    TypeAlias,
}

impl PyObjectKind {
    /// The desc `objtype`/`desctype` string (also the second desc class).
    fn objtype(self) -> &'static str {
        match self {
            PyObjectKind::Function => "function",
            PyObjectKind::Data => "data",
            PyObjectKind::Class => "class",
            PyObjectKind::Exception => "exception",
            PyObjectKind::Method => "method",
            PyObjectKind::Attribute => "attribute",
            PyObjectKind::Property => "property",
            PyObjectKind::TypeAlias => "type",
        }
    }
}

/// One `py:*` object directive after spec-lookup-time aliasing: the
/// [`PyObjectKind`] plus what the aliasing directives' `run()` injects.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct PyDirective {
    kind: PyObjectKind,
    /// `py:classmethod`/`py:staticmethod` inject `options['classmethod']`
    /// / `options['staticmethod']` in `run()` (`__init__.py:287-303`)
    /// rather than accepting the flag as an option (their option_spec is
    /// the plain `PyObject` copy).
    injected: Option<&'static str>,
    /// `py:decorator`/`py:decoratormethod`: `needs_arglist()` forced off
    /// and a leading `desc_addname('@')` (`__init__.py:116-130`, `306-319`).
    decorator: bool,
}

impl PyDirective {
    /// `needs_arglist()`: True only for PyFunction and PyMethod
    /// (`__init__.py:92-93`, `230-231`); decorators override it back to
    /// False (`:129-130`, `:318-319`).
    fn needs_arglist(self) -> bool {
        matches!(self.kind, PyObjectKind::Function | PyObjectKind::Method) && !self.decorator
    }

    /// `allow_nesting`: PyClasslike only (`__init__.py:186`).
    fn allow_nesting(self) -> bool {
        matches!(self.kind, PyObjectKind::Class | PyObjectKind::Exception)
    }
}

/// Which `ObjectDescription` subclass a `desc`-producing directive is.
#[derive(Clone, Copy, PartialEq)]
enum ObjectDescKind {
    /// The bare `ObjectDescription`, registered with docutils under
    /// `describe`/`object` (`directives/__init__.py:375-377`): its
    /// `handle_signature` always raises and its `add_target_and_index` is a
    /// no-op, so it emits desc anatomy with no ids, no index entries and no
    /// std-domain registration.
    Describe,
    /// `GenericObject` (`domains/std/__init__.py:50-88`) — `envvar` is the
    /// only one this crate registers.
    EnvVar,
    /// `ConfigurationValue` (`domains/std/__init__.py:115-185`).
    Confval,
    /// `Cmdoption` (`domains/std/__init__.py:226-330`).
    Cmdoption,
}

/// The `run_object_description` dispatch: which family's overrides run on
/// top of the shared `ObjectDescription.run` anatomy.
#[derive(Clone, Copy, PartialEq)]
enum DescDispatch {
    Std(ObjectDescKind),
    Py(PyDirective),
}

/// Sphinx-mode registry: overlays/extends the docutils-native table.
fn directive_spec_mode(lower: &str, sphinx: bool) -> Option<DirectiveSpec> {
    #[cfg(test)]
    if lower == "sphinx-ultra-test-splice" {
        return Some(DirectiveSpec {
            required_arguments: 1,
            optional_arguments: 0,
            final_argument_whitespace: true,
            has_content: true,
            option_spec: &[],
            kind: DirectiveKind::TestSplice,
        });
    }
    if sphinx {
        if let Some(s) = sphinx_directive_spec(lower) {
            return Some(s);
        }
    }
    directive_spec(lower)
}

/// Suffixes a toctree entry may spell out and still name a document
/// (sphinx `config.source_suffix`, whose default is `{'.rst': ...}`). These
/// are the extensions `SphinxBuilder::is_source_file` discovers.
const SOURCE_SUFFIXES: &[&str] = &[".rst", ".md", ".txt"];

const TOCTREE_OPTS: &[(&str, Conv)] = &[
    ("maxdepth", Conv::PyIntAny),
    ("name", Conv::Unchanged),
    ("class", Conv::ClassOption),
    ("caption", Conv::UnchangedRequired),
    ("glob", Conv::Flag),
    ("hidden", Conv::Flag),
    ("includehidden", Conv::Flag),
    ("numbered", Conv::Unchanged),
    ("titlesonly", Conv::Flag),
    ("reversed", Conv::Flag),
];

const VERSIONADDED: (&str, &str, &str) = ("versionadded", "added", "Added in version {}");
const VERSIONCHANGED: (&str, &str, &str) = ("versionchanged", "changed", "Changed in version {}");
const DEPRECATED: (&str, &str, &str) = ("deprecated", "deprecated", "Deprecated since version {}");
const VERSIONREMOVED: (&str, &str, &str) = ("versionremoved", "removed", "Removed in version {}");

const CODE_BLOCK_OPTS: &[(&str, Conv)] = &[
    ("force", Conv::Flag),
    ("linenos", Conv::Flag),
    ("dedent", Conv::PyIntAny),
    ("lineno-start", Conv::PyIntAny),
    ("emphasize-lines", Conv::UnchangedRequired),
    ("caption", Conv::UnchangedRequired),
    ("class", Conv::ClassOption),
    ("name", Conv::Unchanged),
];

const HIGHLIGHT_OPTS: &[(&str, Conv)] =
    &[("linenothreshold", Conv::PyIntAny), ("force", Conv::Flag)];

/// sphinx.util.parselinenos: 1-based spec ('1,3-5', open ends '-4'/'4-')
/// against `nlines` total lines; invalid or reversed specs raise. Range
/// materialization is clamped to nlines so a huge upper bound cannot
/// blow memory (values past nlines are filtered anyway).
fn parse_linenos(spec: &str, nlines: i64) -> Result<Vec<i64>, String> {
    let invalid = || format!("invalid line number spec: {}", py_repr(Some(spec)));
    let mut out = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if let Some((a, b)) = part.split_once('-') {
            let (a, b) = (a.trim(), b.trim());
            let start = if a.is_empty() {
                1
            } else {
                py_int(a).ok_or_else(invalid)?
            };
            let end = if b.is_empty() {
                nlines
            } else {
                py_int(b).ok_or_else(invalid)?
            };
            if start > end {
                return Err(invalid());
            }
            let clamped_end = end.min(nlines);
            let mut n = start.max(1);
            while n <= clamped_end {
                out.push(n);
                n += 1;
            }
        } else if !part.is_empty() {
            let n = py_int(part).ok_or_else(invalid)?;
            if n >= 1 && n <= nlines {
                out.push(n);
            }
        } else {
            return Err(invalid());
        }
    }
    Ok(out)
}

fn sphinx_directive_spec(lower: &str) -> Option<DirectiveSpec> {
    let version_change = |info: &'static (&'static str, &'static str, &'static str)| {
        Some(DirectiveSpec {
            required_arguments: 1,
            optional_arguments: 1,
            final_argument_whitespace: true,
            has_content: true,
            option_spec: &[],
            kind: DirectiveKind::VersionChange(info),
        })
    };
    match lower {
        "toctree" => Some(DirectiveSpec {
            required_arguments: 0,
            optional_arguments: 0,
            final_argument_whitespace: false,
            has_content: true,
            option_spec: TOCTREE_OPTS,
            kind: DirectiveKind::Toctree,
        }),
        "versionadded" => version_change(&VERSIONADDED),
        "versionchanged" => version_change(&VERSIONCHANGED),
        "deprecated" => version_change(&DEPRECATED),
        "versionremoved" => version_change(&VERSIONREMOVED),
        "seealso" => Some(DirectiveSpec {
            required_arguments: 0,
            optional_arguments: 0,
            final_argument_whitespace: true,
            has_content: true,
            option_spec: ADMONITION_OPTS,
            kind: DirectiveKind::SeeAlso,
        }),
        "code-block" | "sourcecode" => Some(DirectiveSpec {
            required_arguments: 0,
            optional_arguments: 1,
            final_argument_whitespace: false,
            has_content: true,
            option_spec: CODE_BLOCK_OPTS,
            kind: DirectiveKind::SphinxCodeBlock,
        }),
        "highlight" => Some(DirectiveSpec {
            required_arguments: 1,
            optional_arguments: 0,
            final_argument_whitespace: false,
            has_content: false,
            option_spec: HIGHLIGHT_OPTS,
            kind: DirectiveKind::Highlight,
        }),
        "only" => Some(DirectiveSpec {
            required_arguments: 1,
            optional_arguments: 0,
            final_argument_whitespace: true,
            has_content: true,
            option_spec: &[],
            kind: DirectiveKind::Only,
        }),
        "math" => Some(DirectiveSpec {
            required_arguments: 0,
            optional_arguments: 1,
            final_argument_whitespace: true,
            has_content: true,
            option_spec: SPHINX_MATH_OPTS,
            kind: DirectiveKind::SphinxMath,
        }),
        "index" => Some(DirectiveSpec {
            required_arguments: 1,
            optional_arguments: 0,
            final_argument_whitespace: true,
            has_content: false,
            option_spec: NAME_ONLY_OPTS,
            kind: DirectiveKind::IndexDir,
        }),
        "hlist" => Some(DirectiveSpec {
            required_arguments: 0,
            optional_arguments: 0,
            final_argument_whitespace: false,
            has_content: true,
            option_spec: HLIST_OPTS,
            kind: DirectiveKind::HList,
        }),
        "glossary" => Some(DirectiveSpec {
            required_arguments: 0,
            optional_arguments: 0,
            final_argument_whitespace: false,
            has_content: true,
            option_spec: GLOSSARY_OPTS,
            kind: DirectiveKind::Glossary,
        }),
        "describe" | "object" => Some(object_desc_spec(ObjectDescKind::Describe)),
        "envvar" => Some(object_desc_spec(ObjectDescKind::EnvVar)),
        "confval" => Some(object_desc_spec(ObjectDescKind::Confval)),
        "option" | "cmdoption" => Some(object_desc_spec(ObjectDescKind::Cmdoption)),
        // `PythonDomain.directives` (`domains/python/__init__.py:739-754`)
        // with the run()-time name aliasing resolved at spec-lookup time
        // [PY §1.1].
        "py:function" => Some(py_object_desc_spec(PyObjectKind::Function, None, false)),
        "py:data" => Some(py_object_desc_spec(PyObjectKind::Data, None, false)),
        "py:class" => Some(py_object_desc_spec(PyObjectKind::Class, None, false)),
        "py:exception" => Some(py_object_desc_spec(PyObjectKind::Exception, None, false)),
        "py:method" => Some(py_object_desc_spec(PyObjectKind::Method, None, false)),
        "py:classmethod" => Some(py_object_desc_spec(
            PyObjectKind::Method,
            Some("classmethod"),
            false,
        )),
        "py:staticmethod" => Some(py_object_desc_spec(
            PyObjectKind::Method,
            Some("staticmethod"),
            false,
        )),
        "py:attribute" => Some(py_object_desc_spec(PyObjectKind::Attribute, None, false)),
        "py:property" => Some(py_object_desc_spec(PyObjectKind::Property, None, false)),
        "py:type" => Some(py_object_desc_spec(PyObjectKind::TypeAlias, None, false)),
        "py:decorator" => Some(py_object_desc_spec(PyObjectKind::Function, None, true)),
        "py:decoratormethod" => Some(py_object_desc_spec(PyObjectKind::Method, None, true)),
        "py:module" => Some(DirectiveSpec {
            required_arguments: 1,
            optional_arguments: 0,
            final_argument_whitespace: false,
            has_content: true,
            option_spec: PY_MODULE_OPTS,
            kind: DirectiveKind::PyModule,
        }),
        "py:currentmodule" => Some(DirectiveSpec {
            required_arguments: 1,
            optional_arguments: 0,
            final_argument_whitespace: false,
            has_content: false,
            option_spec: &[],
            kind: DirectiveKind::PyCurrentModule,
        }),
        "program" => Some(DirectiveSpec {
            required_arguments: 1,
            optional_arguments: 0,
            final_argument_whitespace: true,
            has_content: false,
            option_spec: &[],
            kind: DirectiveKind::ProgramDir,
        }),
        "default-domain" => Some(DirectiveSpec {
            required_arguments: 1,
            optional_arguments: 0,
            final_argument_whitespace: false,
            has_content: false,
            option_spec: &[],
            kind: DirectiveKind::DefaultDomainDir,
        }),
        _ => None,
    }
}

/// `PyObject.option_spec` (`domains/python/_object.py:172-185`) plus each
/// subclass's additions ([PY §1.2]). The macro keeps the shared twelve in
/// one place.
macro_rules! py_object_opts {
    ($($extra:tt)*) => {
        &[
            ("no-index", Conv::Flag),
            ("no-index-entry", Conv::Flag),
            ("no-contents-entry", Conv::Flag),
            ("no-typesetting", Conv::Flag),
            ("noindex", Conv::Flag),
            ("noindexentry", Conv::Flag),
            ("nocontentsentry", Conv::Flag),
            ("single-line-parameter-list", Conv::Flag),
            ("single-line-type-parameter-list", Conv::Flag),
            ("module", Conv::Unchanged),
            ("canonical", Conv::Unchanged),
            ("annotation", Conv::Unchanged),
            $($extra)*
        ]
    };
}

const PY_OBJECT_OPTS: &[(&str, Conv)] = py_object_opts!();
const PY_FUNCTION_OPTS: &[(&str, Conv)] = py_object_opts!(("async", Conv::Flag),);
const PY_VARIABLE_OPTS: &[(&str, Conv)] =
    py_object_opts!(("type", Conv::Unchanged), ("value", Conv::Unchanged),);
const PY_CLASSLIKE_OPTS: &[(&str, Conv)] =
    py_object_opts!(("abstract", Conv::Flag), ("final", Conv::Flag),);
const PY_METHOD_OPTS: &[(&str, Conv)] = py_object_opts!(
    ("abstract", Conv::Flag),
    ("abstractmethod", Conv::Flag),
    ("async", Conv::Flag),
    ("classmethod", Conv::Flag),
    ("final", Conv::Flag),
    ("staticmethod", Conv::Flag),
);
const PY_PROPERTY_OPTS: &[(&str, Conv)] = py_object_opts!(
    ("abstract", Conv::Flag),
    ("abstractmethod", Conv::Flag),
    ("classmethod", Conv::Flag),
    ("type", Conv::Unchanged),
);

/// `PyModule.option_spec` (`__init__.py:480-490`): note **no
/// `noindexentry`** old spelling (probe `module_bad_option`: it is the
/// unknown-option error), and `no-typesetting` is accepted but unused by
/// `PyModule.run` (probe `module_no_typesetting`: inert). `platform`/
/// `synopsis` are identity lambdas in sphinx — `Conv::Unchanged` differs
/// only for a bare valueless option (`''` here vs Python `None`, probe
/// `module_synopsis_bare`), which the record keeps as `''`.
const PY_MODULE_OPTS: &[(&str, Conv)] = &[
    ("platform", Conv::Unchanged),
    ("synopsis", Conv::Unchanged),
    ("no-index", Conv::Flag),
    ("no-index-entry", Conv::Flag),
    ("no-contents-entry", Conv::Flag),
    ("no-typesetting", Conv::Flag),
    ("noindex", Conv::Flag),
    ("nocontentsentry", Conv::Flag),
    ("deprecated", Conv::Flag),
];

/// The `py:*` object-description directives share `ObjectDescription`'s
/// class-level shape ([`object_desc_spec`]); the option spec is the
/// subclass's — with `py:classmethod`/`py:staticmethod` RESET to the plain
/// `PyObject.option_spec.copy()` (`__init__.py:285`, `:297`): their flags
/// arrive via [`PyDirective::injected`], not as options.
fn py_object_desc_spec(
    kind: PyObjectKind,
    injected: Option<&'static str>,
    decorator: bool,
) -> DirectiveSpec {
    let option_spec: &'static [(&'static str, Conv)] = if injected.is_some() {
        PY_OBJECT_OPTS
    } else {
        match kind {
            PyObjectKind::Function => PY_FUNCTION_OPTS,
            PyObjectKind::Data | PyObjectKind::Attribute => PY_VARIABLE_OPTS,
            PyObjectKind::Class | PyObjectKind::Exception => PY_CLASSLIKE_OPTS,
            PyObjectKind::Method => PY_METHOD_OPTS,
            PyObjectKind::Property => PY_PROPERTY_OPTS,
            // PyTypeAlias re-declares `canonical`, which the base set
            // already carries with the same conversion (`__init__.py:436-439`).
            PyObjectKind::TypeAlias => PY_OBJECT_OPTS,
        }
    };
    DirectiveSpec {
        required_arguments: 1,
        optional_arguments: 0,
        final_argument_whitespace: true,
        has_content: true,
        option_spec,
        kind: DirectiveKind::PyObjectDesc(PyDirective {
            kind,
            injected,
            decorator,
        }),
    }
}

/// `ObjectDescription`'s class-level directive shape
/// (`directives/__init__.py:51-63`): one whitespace-joined argument
/// (multiple signatures arrive as its embedded newlines) and content.
fn object_desc_spec(kind: ObjectDescKind) -> DirectiveSpec {
    DirectiveSpec {
        required_arguments: 1,
        optional_arguments: 0,
        final_argument_whitespace: true,
        has_content: true,
        // `ConfigurationValue` REPLACES the inherited option_spec: it adds
        // `:type:`/`:default:` and drops the three deprecated aliases
        // (`domains/std/__init__.py:117-124`).
        option_spec: match kind {
            ObjectDescKind::Confval => CONFVAL_OPTS,
            _ => OBJECT_DESCRIPTION_OPTS,
        },
        kind: DirectiveKind::ObjectDesc(kind),
    }
}

/// sphinx `ws_re.sub(repl, s)` (`util/__init__.py`, `ws_re = re.compile(r'\s+')`).
fn ws_collapse(s: &str, repl: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_ws = false;
    for c in s.chars() {
        if c.is_whitespace() {
            if !in_ws {
                out.push_str(repl);
                in_ws = true;
            }
        } else {
            out.push(c);
            in_ws = false;
        }
    }
    out
}

/// `ObjectDescription.get_signatures` (`directives/__init__.py:88-98`):
/// backslash-newline pairs vanish (`nl_escape_re`), then one stripped
/// signature per line — each put through `strip_backslash_re.sub(r'\1', …)`
/// when `strip_signature_backslash` is on (probe strip_backslash_on:
/// `f(a\_b)` documents parameter `a_b`).
fn object_signatures(argument: &str, strip_signature_backslash: bool) -> Vec<String> {
    argument
        .replace("\\\n", "")
        .split('\n')
        .map(|line| {
            let line = line.trim();
            if strip_signature_backslash {
                strip_backslashes(line)
            } else {
                line.to_string()
            }
        })
        .collect()
}

/// `strip_backslash_re.sub(r'\1', line)` — `\\(.)`: every backslash
/// followed by a character is removed keeping the character; a lone
/// trailing backslash has no `.` to consume and survives.
fn strip_backslashes(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some(next) => out.push(next),
                None => out.push('\\'),
            }
        } else {
            out.push(c);
        }
    }
    out
}

/// `py_sig_re` (`domains/python/_object.py:41-50`) as a [`PySigMatch`]:
/// groups (prefix, name, tp_list, arglist, retann) plus the byte spans of
/// groups 3/4 that the multi-line measurement subtracts. The regex crate's
/// leftmost-first captures match Python's backtracking on this pattern
/// (pinned by the py_sig_match tests, incl. the greedy-arglist edge).
fn py_sig_match(sig: &str) -> Option<crate::py::arglist::PySigMatch> {
    lazy_static::lazy_static! {
        static ref PY_SIG_RE: regex::Regex = regex::Regex::new(
            r"(?x)^ ([\w.]*\.)?               # class name(s)
                  (\w+) \s*                   # thing name
                  (?: \[ \s* (.*?) \s* \] )?  # optional: type parameters list
                  (?: \( \s* (.*) \s* \)      # optional: arguments
                   (?: \s* ->\s* (.*) )?      #           return annotation
                  )? $",
        )
        .expect("py_sig_re compiles");
    }
    let caps = PY_SIG_RE.captures(sig)?;
    let group = |i: usize| caps.get(i).map(|m| m.as_str().to_string());
    let span_of = |i: usize| caps.get(i).map(|m| (m.start(), m.end())).unwrap_or((0, 0));
    Some(crate::py::arglist::PySigMatch {
        prefix: group(1),
        name: group(2).unwrap_or_default(),
        tp_list: group(3),
        arglist: group(4),
        retann: group(5),
        tp_span: span_of(3),
        arg_span: span_of(4),
    })
}

/// `get_signature_prefix` per py kind (`__init__.py`, [PY §1.3]): the
/// keyword set in FIXED order, each keyword a `desc_sig_keyword` +
/// `desc_sig_space` pair. Note `staticmethod` prints keyword `static`.
fn py_signature_prefix(py: PyDirective, input: &DirectiveInput<'_>) -> Vec<Node> {
    let has = |n: &'static str| opt_get(&input.options, n).is_some() || py.injected == Some(n);
    let mut words: Vec<&str> = Vec::new();
    match py.kind {
        PyObjectKind::Function => {
            if has("async") {
                words.push("async");
            }
        }
        PyObjectKind::Class | PyObjectKind::Exception => {
            if has("final") {
                words.push("final");
            }
            if has("abstract") {
                words.push("abstract");
            }
            words.push(py.kind.objtype());
        }
        PyObjectKind::Method => {
            if has("final") {
                words.push("final");
            }
            if has("abstract") || has("abstractmethod") {
                words.push("abstractmethod");
            }
            if has("async") {
                words.push("async");
            }
            if has("classmethod") {
                words.push("classmethod");
            }
            if has("staticmethod") {
                words.push("static");
            }
        }
        PyObjectKind::Property => {
            if has("abstract") || has("abstractmethod") {
                words.push("abstract");
            }
            if has("classmethod") {
                words.push("class");
            }
            words.push("property");
        }
        PyObjectKind::TypeAlias => words.push("type"),
        PyObjectKind::Data | PyObjectKind::Attribute => {}
    }
    words
        .iter()
        .flat_map(|word| {
            [
                crate::py::annotations::desc_sig_keyword(word),
                crate::py::annotations::desc_sig_space(),
            ]
        })
        .collect()
}

/// `get_index_text` per py kind ([PY §1.4]); PyFunction returns `''` and
/// adds its entry in its own `add_target_and_index` instead.
fn py_index_text(
    py: PyDirective,
    input: &DirectiveInput<'_>,
    modname: Option<&str>,
    name: &str,
    add_module_names: bool,
) -> String {
    let has = |n: &'static str| opt_get(&input.options, n).is_some() || py.injected == Some(n);
    // `clsname, attrname = name.rsplit('.', 1)` with the add_module_names
    // qualification (`__init__.py:262-279` and friends).
    let split = |name: &str| -> Option<(String, String)> {
        let (cls, last) = name.rsplit_once('.')?;
        let cls = match modname {
            Some(modname) if add_module_names => format!("{modname}.{cls}"),
            _ => cls.to_string(),
        };
        Some((cls, last.to_string()))
    };
    match py.kind {
        PyObjectKind::Function => String::new(),
        PyObjectKind::Data => match modname {
            Some(modname) => format!("{name} (in module {modname})"),
            None => format!("{name} (built-in variable)"),
        },
        PyObjectKind::Class => match modname {
            Some(modname) => format!("{name} (class in {modname})"),
            None => format!("{name} (built-in class)"),
        },
        // Exception index entries are the bare name (trap 10).
        PyObjectKind::Exception => name.to_string(),
        PyObjectKind::Method => match split(name) {
            Some((cls, meth)) => {
                if has("classmethod") {
                    format!("{meth}() ({cls} class method)")
                } else if has("staticmethod") {
                    format!("{meth}() ({cls} static method)")
                } else {
                    format!("{meth}() ({cls} method)")
                }
            }
            None => match modname {
                Some(modname) => format!("{name}() (in module {modname})"),
                None => format!("{name}()"),
            },
        },
        PyObjectKind::Attribute => match split(name) {
            Some((cls, attr)) => format!("{attr} ({cls} attribute)"),
            None => match modname {
                Some(modname) => format!("{name} (in module {modname})"),
                None => name.to_string(),
            },
        },
        PyObjectKind::Property => match split(name) {
            Some((cls, attr)) => format!("{attr} ({cls} property)"),
            None => match modname {
                Some(modname) => format!("{name} (in module {modname})"),
                None => name.to_string(),
            },
        },
        PyObjectKind::TypeAlias => match split(name) {
            Some((cls, attr)) => format!("{attr} (type alias in {cls})"),
            None => match modname {
                Some(modname) => format!("{name} (in module {modname})"),
                None => name.to_string(),
            },
        },
    }
}

/// `PyObject._toc_entry_name` (`_object.py:505-522`): parens for the
/// callable objtypes iff `add_function_parentheses`, then the
/// `toc_object_entries_show_parents` shape (unknown values fall through to
/// `''`, matching the un-handled `return` path).
fn py_toc_entry_name(
    parts: &[String],
    fullname: &str,
    kind: PyObjectKind,
    cfg: &crate::py::PySigConfig,
) -> String {
    let Some(last) = parts.last() else {
        return String::new();
    };
    let callable = matches!(kind, PyObjectKind::Function | PyObjectKind::Method);
    let parens = if cfg.add_function_parentheses && callable {
        "()"
    } else {
        ""
    };
    match cfg.toc_object_entries_show_parents.as_str() {
        "domain" => format!("{fullname}{parens}"),
        "hide" => format!("{last}{parens}"),
        "all" => {
            let mut joined = parts[..parts.len() - 1].to_vec();
            joined.push(format!("{last}{parens}"));
            joined.join(".")
        }
        _ => String::new(),
    }
}

/// Python tuple-repr of a string sequence: `()`, `('a',)`, `('a', 'b')` —
/// the `_toc_parts` pformat shape.
fn py_tuple_repr(parts: &[String]) -> String {
    match parts {
        [] => "()".to_string(),
        [one] => format!("({},)", py_repr(Some(one))),
        _ => format!(
            "({})",
            parts
                .iter()
                .map(|part| py_repr(Some(part)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
    }
}

/// `addnodes.desc_annotation` — a `FixedTextElement` (xml:space preserve)
/// with no extra classes; also the shape of `desc_returns`.
fn desc_annotation_node(span: Span) -> Node {
    let mut node = Node::elem("desc_annotation", span);
    node.set("xml:space", AttrValue::Str("preserve".to_string()));
    node
}

// ====================================================================
// Doc-field transformation (M2 wave 4.5 task 7): the
// `sphinx.util.docfields.DocFieldTransformer` port scoped to the py
// field set (`PyObject.doc_field_types`, `_object.py:187-232`), run as a
// desc_content post-pass for py object kinds only [PY §1.6 "Doc fields"].
// ====================================================================

/// One `PyObject.doc_field_types` entry. The five entries mirror
/// `_object.py:187-232` exactly (names/typenames/labels/roles verified
/// against the 9.1.0 source dump).
/// Sphinx keys these by `Field.name` (`parameter`/`variable`/
/// `exceptions`/`returnvalue`/`returntype`); the port keys the grouped
/// entries and the `types` map by [`PY_DOC_FIELDS`] index instead.
struct PyDocField {
    /// The rendered `field_name` label.
    label: &'static str,
    /// `GroupedField`: every occurrence collects into ONE field.
    is_grouped: bool,
    /// `TypedField`: `:type x:` companions and `:param type name:` syntax.
    is_typed: bool,
    /// Single-item groups collapse to a bare paragraph (no bullet_list).
    can_collapse: bool,
    /// Whether the field REQUIRES an argument (`Field.has_arg`); a
    /// mismatch in either direction demotes the field to unknown.
    has_arg: bool,
    /// Role for the field-argument xrefs (only raises' `exc`).
    rolename: &'static str,
    /// Role for typed fields' type xrefs (`class`).
    typerolename: &'static str,
    /// Role for a single-text BODY (only rtype's `class`).
    bodyrolename: &'static str,
    /// Whether the field class carries `PyXrefMixin` — `returnvalue` is a
    /// plain `docfields.Field`, everything else is a `Py*Field`. The mixin
    /// is what splits multi-type strings and stamps the py attrs.
    py_xref: bool,
}

/// Indices into [`PY_DOC_FIELDS`].
const PY_FIELD_PARAMETER: usize = 0;
const PY_FIELD_VARIABLE: usize = 1;
const PY_FIELD_EXCEPTIONS: usize = 2;
const PY_FIELD_RETURNVALUE: usize = 3;
const PY_FIELD_RETURNTYPE: usize = 4;

const PY_DOC_FIELDS: [PyDocField; 5] = [
    PyDocField {
        label: "Parameters",
        is_grouped: true,
        is_typed: true,
        can_collapse: true,
        has_arg: true,
        rolename: "",
        typerolename: "class",
        bodyrolename: "",
        py_xref: true,
    },
    PyDocField {
        label: "Variables",
        is_grouped: true,
        is_typed: true,
        can_collapse: true,
        has_arg: true,
        rolename: "",
        typerolename: "class",
        bodyrolename: "",
        py_xref: true,
    },
    PyDocField {
        label: "Raises",
        is_grouped: true,
        is_typed: false,
        can_collapse: true,
        has_arg: true,
        rolename: "exc",
        typerolename: "",
        bodyrolename: "",
        py_xref: true,
    },
    PyDocField {
        label: "Returns",
        is_grouped: false,
        is_typed: false,
        can_collapse: false,
        has_arg: false,
        rolename: "",
        typerolename: "",
        bodyrolename: "",
        py_xref: false,
    },
    PyDocField {
        label: "Return type",
        is_grouped: false,
        is_typed: false,
        can_collapse: false,
        has_arg: false,
        rolename: "",
        typerolename: "",
        bodyrolename: "class",
        py_xref: true,
    },
];

/// A directive's `get_field_type_map()` lookup: field-name ->
/// `(doc_field_types index, is_typefield)`.
type DocFieldTypeMap = fn(&str) -> Option<(usize, bool)>;

/// `get_field_type_map()` for the std object-description kinds
/// (Describe/EnvVar/Confval/Cmdoption): none of them declare
/// `doc_field_types`, so the map is empty and every field takes the
/// transformer's unknown branch — capitalized name, body passed through
/// (oracle probes envvar_param/describe_param/option_param/
/// confval_type_and_field/envvar_multi_fields).
fn std_field_type_map(_name: &str) -> Option<(usize, bool)> {
    None
}

/// `ObjectDescription.get_field_type_map()` for the py set: field-name ->
/// `(doc_field_types index, is_typefield)`.
fn py_field_type_map(name: &str) -> Option<(usize, bool)> {
    Some(match name {
        "param" | "parameter" | "arg" | "argument" | "keyword" | "kwarg" | "kwparam" => {
            (PY_FIELD_PARAMETER, false)
        }
        "paramtype" | "type" => (PY_FIELD_PARAMETER, true),
        "var" | "ivar" | "cvar" => (PY_FIELD_VARIABLE, false),
        "vartype" => (PY_FIELD_VARIABLE, true),
        "raises" | "raise" | "exception" | "except" => (PY_FIELD_EXCEPTIONS, false),
        "returns" | "return" => (PY_FIELD_RETURNVALUE, false),
        "rtype" => (PY_FIELD_RETURNTYPE, false),
        _ => return None,
    })
}

/// `filter_meta_fields` (`domains/python/__init__.py:603-617`), fired on
/// the `object-description-transform` event — which the base `run` emits
/// BEFORE `DocFieldTransformer.transform_all` (`directives/__init__.py`),
/// so `:meta:` fields vanish from the raw list and the transformer then
/// replaces the emptied list with an empty `<field_list>` that REMAINS
/// [PY §1.6 meta probe]. py domain only (the event handler checks).
fn filter_meta_fields(content: &mut Node) {
    for child in &mut content.children {
        if child.kind == kinds::FIELD_LIST {
            child.children.retain(|field| {
                if field.kind != kinds::FIELD {
                    return true;
                }
                let name = field.children.first().map(Node::astext).unwrap_or_default();
                let name = name.trim();
                !(name == "meta" || name.starts_with("meta "))
            });
        }
    }
}

/// `_is_single_paragraph` (`docfields.py:34-42`): exactly one paragraph,
/// tolerating trailing system_messages.
fn is_single_field_paragraph(field_body: &Node) -> bool {
    if field_body.children.is_empty() {
        return false;
    }
    if field_body.children[1..]
        .iter()
        .any(|n| n.kind != kinds::SYSTEM_MESSAGE)
    {
        return false;
    }
    field_body.children[0].kind == kinds::PARAGRAPH
}

/// Python `str.split(None, maxsplit=1)` on a field name: leading
/// whitespace skipped, the remainder trimmed at its start. A missing or
/// all-whitespace remainder is the `ValueError` path — the ORIGINAL text
/// comes back whole with an empty argument (`docfields.py:384-389`).
fn split_field_name(text: &str) -> (String, String) {
    let trimmed = text.trim_start();
    if let Some(i) = trimmed.find(char::is_whitespace) {
        let rest = trimmed[i..].trim_start();
        if !rest.is_empty() {
            return (trimmed[..i].to_string(), rest.to_string());
        }
    }
    (text.to_string(), String::new())
}

/// Python `fieldarg.rsplit(None, 1)` for the `:param type name:` syntax
/// (`docfields.py:448-455`): `None` is the single-token `ValueError` path.
fn rsplit_field_arg(arg: &str) -> Option<(String, String)> {
    let trimmed = arg.trim_end();
    let (i, ws) = trimmed
        .char_indices()
        .rev()
        .find(|(_, c)| c.is_whitespace())?;
    let head = trimmed[..i].trim_end();
    if head.is_empty() {
        return None;
    }
    Some((head.to_string(), trimmed[i + ws.len_utf8()..].to_string()))
}

/// Python `s[0:1].upper() + s[1:]` (unknown-field renaming).
fn capitalize_first(s: &str) -> String {
    let mut chars = s.chars();
    match chars.next() {
        Some(c) => c.to_uppercase().collect::<String>() + chars.as_str(),
        None => String::new(),
    }
}

/// docutils `nodes.Inline` membership for the kinds this parser emits,
/// plus `Text` — the filter typed-field bodies pass through
/// (`docfields.py:442`; block-level nodes would render invalid markup).
fn is_inline_or_text(node: &Node) -> bool {
    matches!(
        node.kind,
        kinds::TEXT
            | "abbreviation"
            | "acronym"
            | "citation_reference"
            | "emphasis"
            | "footnote_reference"
            | "generated"
            | "image"
            | "index"
            | "inline"
            | "literal"
            | "literal_emphasis"
            | "literal_strong"
            | "math"
            | "pending_xref"
            | "problematic"
            | "raw"
            | "reference"
            | "strong"
            | "subscript"
            | "substitution_reference"
            | "superscript"
            | "target"
            | "title_reference"
    )
}

/// `PyXrefMixin._delimiters_re` split with the captured delimiters kept
/// (Python `re.split` with a group) and empties dropped (`filter(None)`).
/// Returns `(piece, is_delimiter)`; a text piece can never start with a
/// delimiter match (the scan would have split there), so the flag is
/// exactly `self._delimiters_re.match(sub_target)` (`_object.py:589`).
fn split_type_delimiters(target: &str) -> Vec<(String, bool)> {
    lazy_static::lazy_static! {
        static ref DELIMITERS_RE: regex::Regex =
            regex::Regex::new(r"\s*[\[\](),](?:\s*o[rf]\s)?\s*|\s+o[rf]\s+|\s*\|\s*|\.\.\.")
                .unwrap();
    }
    let mut out = Vec::new();
    let mut last = 0;
    for m in DELIMITERS_RE.find_iter(target) {
        if m.start() > last {
            out.push((target[last..m.start()].to_string(), false));
        }
        out.push((target[m.start()..m.end()].to_string(), true));
        last = m.end();
    }
    if last < target.len() {
        out.push((target[last..].to_string(), false));
    }
    out
}

/// A `TextElement(rawsource, text)`: element node with one Text child
/// (none when the text is empty, like docutils).
fn doc_field_inline(kind: &'static str, text: &str, span: Span) -> Node {
    let mut node = Node::elem(kind, span);
    if !text.is_empty() {
        node.children.push(Node::text_node(text, span));
    }
    node
}

/// `PyXrefMixin.make_xref` (`_object.py:514-562`) over
/// `Field.make_xref` (`docfields.py:78-119`). The mixin always calls the
/// base with `inliner=None`, so a non-empty rolename ALWAYS yields a
/// `pending_xref` (never the role-run inline), which then gets
/// `refspecific=1`, the `py:module`/`py:class` ref_context attrs, the
/// `parse_reftarget` title rewrite, and — only when title == target — the
/// `python_use_unqualified_type_names` two-condition wrapping with the
/// innernode INSIDE each condition [SIG §4.2 item 2, probes F-U1/F-U2].
/// The directive/environment state every field builder reads: the
/// ref_context slice the xrefs stamp, the py signature config, and the
/// span new nodes carry (docutils tracks no provenance for them; the
/// enclosing field_list's span keeps ours structural).
struct DocFieldEnv<'a> {
    /// The directive's `get_field_type_map()` (py table or the empty std
    /// one).
    map: DocFieldTypeMap,
    ctx: &'a crate::py::annotations::PyRefContext,
    cfg: &'a crate::py::PySigConfig,
    span: Span,
}

fn py_make_doc_xref(
    rolename: &str,
    target: &str,
    innernode: &'static str,
    contnode: Option<Node>,
    env: &DocFieldEnv<'_>,
) -> Node {
    let span = env.span;
    if rolename.is_empty() {
        // `return contnode or innernode(target, target)` — no xref, no
        // mixin post-processing (the result is not a pending_xref).
        return contnode.unwrap_or_else(|| doc_field_inline(innernode, target, span));
    }
    let mut refnode = Node::elem("pending_xref", span);
    refnode.set("refdomain", AttrValue::Str("py".to_string()));
    // Python bools render as 0/1 in pformat (`Element.starttag`).
    refnode.set("refexplicit", AttrValue::Int(0));
    refnode.set("reftype", AttrValue::Str(rolename.to_string()));
    refnode.set("reftarget", AttrValue::Str(target.to_string()));
    refnode
        .children
        .push(contnode.unwrap_or_else(|| doc_field_inline(innernode, target, span)));
    // `PythonDomain.process_field_xref` is a no-op in 9.1.0.

    // PyXrefMixin post-processing (`_object.py:537-562`).
    refnode.set("refspecific", AttrValue::Int(1));
    // ref_context attrs are Python None outside a py scope; pformat
    // renders None as the "True" sentinel (same convention as
    // `crate::py::annotations::type_to_xref`).
    refnode.set(
        "py:module",
        AttrValue::Str(env.ctx.module.clone().unwrap_or_else(|| "True".to_string())),
    );
    refnode.set(
        "py:class",
        AttrValue::Str(env.ctx.class_.clone().unwrap_or_else(|| "True".to_string())),
    );
    let (reftype, reftarget, reftitle, _refspecific) =
        crate::py::annotations::parse_reftarget(target);
    if reftarget != reftitle {
        // `~pkg.Cls` / leading-`.` / `typing.` rewrite — takes precedence
        // over the unqualified-names branch (elif).
        refnode.set("reftype", AttrValue::Str(reftype));
        refnode.set("reftarget", AttrValue::Str(reftarget));
        refnode.children.clear();
        refnode
            .children
            .push(doc_field_inline(innernode, &reftitle, span));
    } else if env.cfg.python_use_unqualified_type_names {
        let children = std::mem::take(&mut refnode.children);
        // `shortname = target.rpartition('.')[-1]`.
        let shortname = target.rsplit('.').next().unwrap_or(target);
        let textnode = doc_field_inline(innernode, shortname, span);
        for (condition, nodes) in [("resolved", vec![textnode]), ("*", children)] {
            let mut cond = Node::elem("pending_xref_condition", span);
            cond.set("condition", AttrValue::Str(condition.to_string()));
            cond.children = nodes;
            refnode.children.push(cond);
        }
    }
    refnode
}

/// `make_xrefs`: `PyXrefMixin.make_xrefs` (`_object.py:568-608`) for the
/// `Py*Field` classes — delimiter split, sticky `Literal[...]`
/// suppression — or the plain single-xref `Field.make_xrefs`
/// (`docfields.py:121-136`) for `returnvalue`.
fn py_make_doc_xrefs(
    spec: &PyDocField,
    rolename: &str,
    target: &str,
    innernode: &'static str,
    contnode: Option<&Node>,
    env: &DocFieldEnv<'_>,
) -> Vec<Node> {
    if !spec.py_xref {
        return vec![py_make_doc_xref(
            rolename,
            target,
            innernode,
            contnode.cloned(),
            env,
        )];
    }
    let split_contnode = contnode.is_some_and(|c| c.astext() == target);
    let mut in_literal = false;
    let mut results = Vec::new();
    for (sub_target, is_delim) in split_type_delimiters(target) {
        let cont: Option<Node> = if split_contnode {
            Some(Node::text_node(sub_target.clone(), env.span))
        } else {
            contnode.cloned()
        };
        if in_literal || is_delim {
            results
                .push(cont.unwrap_or_else(|| doc_field_inline(innernode, &sub_target, env.span)));
        } else {
            results.push(py_make_doc_xref(
                rolename,
                &sub_target,
                innernode,
                cont,
                env,
            ));
        }
        if matches!(
            sub_target.as_str(),
            "Literal" | "typing.Literal" | "~typing.Literal"
        ) {
            in_literal = true;
        }
    }
    results
}

/// One collected entry: a pass-through original field, or a field type
/// with its `(fieldarg, content)` items (grouped types collect many).
enum DocFieldEntry {
    Pass(Node),
    Typed {
        ftype: usize,
        items: Vec<(String, Vec<Node>)>,
    },
}

/// `DocFieldTransformer.transform` for ONE `field_list` node, with the
/// directive's typemap. The list's children are rebuilt in place
/// (docutils `replace_self` with a fresh `field_list`, so any attributes
/// are dropped too).
fn transform_doc_field_list(
    node: &mut Node,
    map: DocFieldTypeMap,
    ctx: &crate::py::annotations::PyRefContext,
    cfg: &crate::py::PySigConfig,
) {
    use std::collections::HashMap;
    let env = DocFieldEnv {
        map,
        ctx,
        cfg,
        span: node.span,
    };
    let fields = std::mem::take(&mut node.children);
    node.attrs = crate::doctree::Attrs::default();

    // Step 1: collect field types and content (`docfields.py:374-482`).
    let mut entries: Vec<DocFieldEntry> = Vec::new();
    let mut group_indices: HashMap<usize, usize> = HashMap::new();
    let mut types: HashMap<usize, HashMap<String, Vec<Node>>> = HashMap::new();
    for field in fields {
        doc_field_step1(field, &mut entries, &mut types, &mut group_indices, &env);
    }

    // Step 2: construct the new field list (`docfields.py:484-510`).
    for entry in entries {
        match entry {
            DocFieldEntry::Pass(field) => node.children.push(field),
            DocFieldEntry::Typed { ftype, items } => {
                let mut empty = HashMap::new();
                let fieldtypes = types.get_mut(&ftype).unwrap_or(&mut empty);
                node.children.push(make_doc_field(
                    &PY_DOC_FIELDS[ftype],
                    items,
                    fieldtypes,
                    &env,
                ));
            }
        }
    }
}

/// `DocFieldTransformer._transform_step_1` (`docfields.py:374-482`),
/// minus the translatable-inline wrapper: sphinx wraps grouped/plain
/// content in `nodes.inline(translatable=True)`, which the
/// `RemoveTranslatableInline(999)` transform splices away again on every
/// untranslated build — the harness3 probes never see it, so the port
/// skips the round-trip.
fn doc_field_step1(
    mut field: Node,
    entries: &mut Vec<DocFieldEntry>,
    types: &mut std::collections::HashMap<usize, std::collections::HashMap<String, Vec<Node>>>,
    group_indices: &mut std::collections::HashMap<usize, usize>,
    env: &DocFieldEnv<'_>,
) {
    // `assert len(field) == 2` — the parser always emits [name, body].
    if field.children.len() != 2 {
        entries.push(DocFieldEntry::Pass(field));
        return;
    }
    let name_text = field.children[0].astext();
    let (fieldtype_name, mut fieldarg) = split_field_name(&name_text);
    let lookup = (env.map)(&fieldtype_name);

    // Collect the content, trying not to keep unnecessary paragraphs.
    let single_para = is_single_field_paragraph(&field.children[1]);
    let content: Vec<Node> = if single_para {
        field.children[1].children[0].children.clone()
    } else {
        field.children[1].children.clone()
    };

    // Sort out unknown fields (or an argument mismatching the spec):
    // capitalize the field name and pass the field through untouched —
    // except a lone typefield body, which still gets type-linked.
    let known = lookup.is_some_and(|(i, _)| PY_DOC_FIELDS[i].has_arg == !fieldarg.is_empty());
    if !known {
        let mut new_fieldname = capitalize_first(&fieldtype_name);
        if !fieldarg.is_empty() {
            new_fieldname.push(' ');
            new_fieldname.push_str(&fieldarg);
        }
        // `field_name[0] = nodes.Text(new_fieldname)` — only the FIRST
        // child is replaced.
        let name_span = field.children[0].span;
        if !field.children[0].children.is_empty() {
            field.children[0].children[0] = Node::text_node(new_fieldname, name_span);
        }
        if let Some((ftype, true)) = lookup {
            // "but if this has a type then we can at least link it"
            if content.len() == 1 && content[0].kind == kinds::TEXT {
                let spec = &PY_DOC_FIELDS[ftype];
                let target = content[0].astext();
                let xrefs = py_make_doc_xrefs(
                    spec,
                    spec.typerolename,
                    &target,
                    "emphasis",
                    Some(&content[0]),
                    env,
                );
                let body = &mut field.children[1];
                if single_para {
                    body.children[0].children = xrefs;
                } else {
                    let mut para = Node::elem(kinds::PARAGRAPH, env.span);
                    para.children = xrefs;
                    body.children = vec![para];
                }
            }
        }
        entries.push(DocFieldEntry::Pass(field));
        return;
    }
    let (ftype, is_typefield) = lookup.expect("known implies present");
    let spec = &PY_DOC_FIELDS[ftype];

    // A typefield puts its content into the types collection and emits
    // nothing itself; only inline nodes survive the trip.
    if is_typefield {
        let filtered: Vec<Node> = content.into_iter().filter(is_inline_or_text).collect();
        if !filtered.is_empty() {
            types.entry(ftype).or_default().insert(fieldarg, filtered);
        }
        return;
    }

    // Also support the `:param type name:` syntax.
    if spec.is_typed {
        if let Some((argtype, argname)) = rsplit_field_arg(&fieldarg) {
            types
                .entry(ftype)
                .or_default()
                .insert(argname.clone(), vec![Node::text_node(argtype, env.span)]);
            fieldarg = argname;
        }
    }

    if spec.is_grouped {
        if let Some(&i) = group_indices.get(&ftype) {
            if let DocFieldEntry::Typed { items, .. } = &mut entries[i] {
                items.push((fieldarg, content));
            }
        } else {
            group_indices.insert(ftype, entries.len());
            entries.push(DocFieldEntry::Typed {
                ftype,
                items: vec![(fieldarg, content)],
            });
        }
    } else {
        entries.push(DocFieldEntry::Typed {
            ftype,
            items: vec![(fieldarg, content)],
        });
    }
}

/// `make_field` dispatch on the field class: `TypedField.make_field`
/// (`docfields.py:286-339`), `GroupedField.make_field` (`:214-248`), or
/// `Field.make_field` (`:141-184`).
fn make_doc_field(
    spec: &PyDocField,
    items: Vec<(String, Vec<Node>)>,
    fieldtypes: &mut std::collections::HashMap<String, Vec<Node>>,
    env: &DocFieldEnv<'_>,
) -> Node {
    let span = env.span;
    let mut fieldname = Node::elem(kinds::FIELD_NAME, span);
    fieldname.children.push(Node::text_node(spec.label, span));

    let fieldbody_children: Vec<Node> = if spec.is_typed {
        // TypedField: `name ( <type xrefs> ) -- description`, with the
        // type popped from the `:type x:` map (pop guards a doubled
        // `:param x:` from inserting the same type nodes twice).
        let handle_item = |fieldarg: String,
                           content: Vec<Node>,
                           fieldtypes: &mut std::collections::HashMap<String, Vec<Node>>|
         -> Node {
            let mut par = Node::elem(kinds::PARAGRAPH, span);
            par.children.extend(py_make_doc_xrefs(
                spec,
                spec.rolename,
                &fieldarg,
                "literal_strong",
                None,
                env,
            ));
            if let Some(fieldtype) = fieldtypes.remove(&fieldarg) {
                par.children.push(Node::text_node(" (", span));
                if fieldtype.len() == 1 && fieldtype[0].kind == kinds::TEXT {
                    let typename = fieldtype[0].astext();
                    par.children.extend(py_make_doc_xrefs(
                        spec,
                        spec.typerolename,
                        &typename,
                        "literal_emphasis",
                        None,
                        env,
                    ));
                } else {
                    par.children.extend(fieldtype);
                }
                par.children.push(Node::text_node(")", span));
            }
            let has_content = content.iter().any(|c| !c.astext().trim().is_empty());
            if has_content {
                par.children.push(Node::text_node(" -- ", span));
                par.children.extend(content);
            }
            par
        };

        if items.len() == 1 && spec.can_collapse {
            let (fieldarg, content) = items.into_iter().next().expect("one item");
            vec![handle_item(fieldarg, content, fieldtypes)]
        } else {
            let mut listnode = Node::elem(kinds::BULLET_LIST, span);
            for (fieldarg, content) in items {
                let mut li = Node::elem(kinds::LIST_ITEM, span);
                li.children.push(handle_item(fieldarg, content, fieldtypes));
                listnode.children.push(li);
            }
            vec![listnode]
        }
    } else if spec.is_grouped {
        // GroupedField: `<arg xref> -- description` items (the ` -- ` is
        // unconditional here, unlike TypedField's).
        let mut list_items: Vec<Node> = Vec::new();
        for (fieldarg, content) in items {
            let mut par = Node::elem(kinds::PARAGRAPH, span);
            par.children.extend(py_make_doc_xrefs(
                spec,
                spec.rolename,
                &fieldarg,
                "literal_strong",
                None,
                env,
            ));
            par.children.push(Node::text_node(" -- ", span));
            par.children.extend(content);
            let mut li = Node::elem(kinds::LIST_ITEM, span);
            li.children.push(par);
            list_items.push(li);
        }
        if list_items.len() == 1 && spec.can_collapse {
            let mut li = list_items.pop().expect("one item");
            vec![li.children.pop().expect("item paragraph")]
        } else {
            let mut listnode = Node::elem(kinds::BULLET_LIST, span);
            listnode.children = list_items;
            vec![listnode]
        }
    } else {
        // Field: single entry; a single-Text body may get a body role
        // (rtype's `class`). Both py Field-type fields have
        // `has_arg=False`, so the fieldarg-in-name branch is unreachable.
        let (_fieldarg, mut content) = items.into_iter().next().expect("one item");
        let single_textish = content.len() == 1
            && (content[0].kind == kinds::TEXT
                || (content[0].kind == "inline"
                    && content[0].children.len() == 1
                    && content[0].children[0].kind == kinds::TEXT));
        if single_textish {
            let target = content[0].astext();
            let contnode = content[0].clone();
            content = py_make_doc_xrefs(
                spec,
                spec.bodyrolename,
                &target,
                "emphasis",
                Some(&contnode),
                env,
            );
        }
        let mut par = Node::elem(kinds::PARAGRAPH, span);
        par.children = content;
        vec![par]
    };

    let mut fieldbody = Node::elem(kinds::FIELD_BODY, span);
    fieldbody.children = fieldbody_children;
    let mut fieldnode = Node::elem(kinds::FIELD, span);
    fieldnode.children.push(fieldname);
    fieldnode.children.push(fieldbody);
    fieldnode
}

/// `option_desc_re = r'((?:/|--|-|\+)?[^\s=]+)(=?\s*.*)'` matched with
/// `re.match` (anchored at the start only). The optional prefix backtracks:
/// `--` is tried before `-`, and both before the empty alternative, so a
/// bare `--` matches as prefix `-` + name `-`.
fn option_desc_match(s: &str) -> Option<(String, String)> {
    let mut prefixes: Vec<usize> = Vec::new();
    if s.starts_with('/') {
        prefixes.push(1);
    }
    if s.starts_with("--") {
        prefixes.push(2);
    }
    if s.starts_with('-') {
        prefixes.push(1);
    }
    if s.starts_with('+') {
        prefixes.push(1);
    }
    prefixes.push(0);
    for prefix in prefixes {
        // `[^\s=]+`, greedy and at least one character long.
        let taken: usize = s[prefix..]
            .chars()
            .take_while(|c| !c.is_whitespace() && *c != '=')
            .map(char::len_utf8)
            .sum();
        if taken > 0 {
            return Some((
                s[..prefix + taken].to_string(),
                s[prefix + taken..].to_string(),
            ));
        }
    }
    None
}

/// `addnodes.desc_name` — `_DescClassesInjector` stamps the two classes and
/// `FixedTextElement` the `xml:space` (`sphinx/addnodes.py`).
fn desc_name_node(text: &str, span: Span) -> Node {
    sig_text_node("desc_name", ["sig-name", "descname"], text, span)
}

/// `addnodes.desc_addname`.
fn desc_addname_node(text: &str, span: Span) -> Node {
    sig_text_node("desc_addname", ["sig-prename", "descclassname"], text, span)
}

fn sig_text_node(kind: &'static str, classes: [&str; 2], text: &str, span: Span) -> Node {
    let mut node = Node::elem(kind, span);
    node.attrs
        .classes
        .extend(classes.iter().map(|c| c.to_string()));
    node.set("xml:space", AttrValue::Str("preserve".to_string()));
    // `TextElement(rawsource, text)` adds no child for an empty text — the
    // `desc_addname` of an argument-less option is an empty element.
    if !text.is_empty() {
        node.children.push(Node::text_node(text, span));
    }
    node
}

/// `[node_id for el in node.findall(nodes.Element) for node_id in el['ids']]`
/// — `findall` yields the node itself first, then its descendants in
/// document order, and skips Text nodes (they are not Elements).
fn collect_element_ids(node: &Node, out: &mut Vec<String>) {
    if node.kind == kinds::TEXT {
        return;
    }
    out.extend(node.attrs.ids.iter().cloned());
    for child in &node.children {
        collect_element_ids(child, out);
    }
}

/// `ObjectDescription.option_spec` (`directives/__init__.py:55-63`).
const OBJECT_DESCRIPTION_OPTS: &[(&str, Conv)] = &[
    ("no-index", Conv::Flag),
    ("no-index-entry", Conv::Flag),
    ("no-contents-entry", Conv::Flag),
    ("no-typesetting", Conv::Flag),
    ("noindex", Conv::Flag),
    ("noindexentry", Conv::Flag),
    ("nocontentsentry", Conv::Flag),
];

/// `ConfigurationValue.option_spec` (`domains/std/__init__.py:117-124`).
const CONFVAL_OPTS: &[(&str, Conv)] = &[
    ("no-index", Conv::Flag),
    ("no-index-entry", Conv::Flag),
    ("no-contents-entry", Conv::Flag),
    ("no-typesetting", Conv::Flag),
    ("type", Conv::UnchangedRequired),
    ("default", Conv::UnchangedRequired),
];

/// One `index['entries']` 5-tuple, rendered the way `str(tuple)` renders it
/// in Python — the *unescaped* item docutils then puts through
/// `serial_escape` when it prints the list attribute, which is why the
/// `entries` attribute is an [`AttrValue::List`] rather than a pre-joined
/// string: only the list form doubles a backslash inside a value, as
/// docutils does.
///
/// [`crate::env::genindex::parse_index_entries`] is the exact inverse, and
/// is what lifts these back out of a doctree for the index domain.
pub(crate) fn index_entry_tuple(
    entrytype: &str,
    value: &str,
    target_id: &str,
    main: &str,
    key: Option<&str>,
) -> String {
    format!(
        "({}, {}, {}, {}, {})",
        py_repr(Some(entrytype)),
        py_repr(Some(value)),
        py_repr(Some(target_id)),
        py_repr(Some(main)),
        py_repr(key)
    )
}

/// `process_index_entry` (`sphinx/util/nodes.py:431-482`): one `.. index::`
/// line as serialized 5-tuples.
///
/// Two details the shape of this function turns on: the `!` main marker is
/// stripped *with the whitespace behind it* (`entry[1:].lstrip()`), and the
/// comma shorthand re-splits `oentry` — the line *before* that strip — so
/// each comma-separated value re-reads its own `!`.
///
/// The legacy `module:`/`keyword:`/... prefixes raise `ValueError` in
/// sphinx; here they fall through to the shorthand branch (hardening note —
/// the oracle corpus avoids them).
fn process_index_entry(entry: &str, target_id: &str) -> Vec<String> {
    const TYPES: &[&str] = &["single", "pair", "double", "triple", "see", "seealso"];
    let oentry = entry.trim();
    let stripped = match oentry.strip_prefix('!') {
        Some(rest) => rest.trim_start(),
        None => oentry,
    };
    let main = if oentry.starts_with('!') { "main" } else { "" };
    for t in TYPES {
        if let Some(value) = stripped.strip_prefix(&format!("{t}:")) {
            let value = value.trim();
            let ty = if *t == "double" { "pair" } else { t };
            return vec![index_entry_tuple(ty, value, target_id, main, None)];
        }
    }
    // Shorthand notation for single entries: every comma-separated value of
    // the *original* line, each carrying its own `!` marker.
    oentry
        .split(',')
        .filter_map(|value| {
            let value = value.trim();
            let (main, value) = match value.strip_prefix('!') {
                Some(rest) => ("main", rest.trim_start()),
                None => ("", value),
            };
            if value.is_empty() {
                return None;
            }
            Some(index_entry_tuple("single", value, target_id, main, None))
        })
        .collect()
}

const SPHINX_MATH_OPTS: &[(&str, Conv)] = &[
    ("label", Conv::Unchanged),
    ("name", Conv::Unchanged),
    ("class", Conv::ClassOption),
    ("no-wrap", Conv::Flag),
    ("nowrap", Conv::Flag),
];

const HLIST_OPTS: &[(&str, Conv)] = &[("columns", Conv::PyIntAny)];

const GLOSSARY_OPTS: &[(&str, Conv)] = &[("sorted", Conv::Flag)];

const UNICODE_OPTS: &[(&str, Conv)] = &[
    ("trim", Conv::Flag),
    ("ltrim", Conv::Flag),
    ("rtrim", Conv::Flag),
];

/// `( |\n|^)\.\. ` comment split for the unicode directive (misc.py:399):
/// returns the byte index where the argument text is cut.
fn unicode_comment_cut(text: &str) -> usize {
    if text.starts_with(".. ") {
        return 0;
    }
    let bytes = text.as_bytes();
    for i in 0..text.len() {
        if (bytes[i] == b' ' || bytes[i] == b'\n') && text[i + 1..].starts_with(".. ") {
            return i;
        }
    }
    text.len()
}

/// Minimal strftime over the current LOCAL-approximated (UTC) time:
/// %Y/%m/%d/%H/%M/%S/%% expand, other bytes pass through.
fn strftime_now(format: &str) -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let days = (secs / 86400) as i64;
    let (h, mi, s) = ((secs % 86400) / 3600, (secs % 3600) / 60, secs % 60);
    // Howard Hinnant's civil_from_days.
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    let mut out = String::new();
    let mut chars = format.chars();
    while let Some(c) = chars.next() {
        if c != '%' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('Y') => out.push_str(&year.to_string()),
            Some('m') => out.push_str(&format!("{m:02}")),
            Some('d') => out.push_str(&format!("{d:02}")),
            Some('H') => out.push_str(&format!("{h:02}")),
            Some('M') => out.push_str(&format!("{mi:02}")),
            Some('S') => out.push_str(&format!("{s:02}")),
            Some('%') => out.push('%'),
            Some(other) => {
                out.push('%');
                out.push(other);
            }
            None => out.push('%'),
        }
    }
    out
}

const TABLE_OPTS: &[(&str, Conv)] = &[
    ("align", Conv::Choice(H_ALIGN_VALUES)),
    ("class", Conv::ClassOption),
    ("name", Conv::Unchanged),
    ("width", Conv::LengthOrPercentageOrUnitless("")),
    ("widths", Conv::WidthsAutoGrid),
];

const CSV_TABLE_OPTS: &[(&str, Conv)] = &[
    ("header-rows", Conv::NonnegativeInt),
    ("stub-columns", Conv::NonnegativeInt),
    ("header", Conv::Unchanged),
    ("width", Conv::LengthOrPercentageOrUnitless("")),
    ("widths", Conv::WidthsAuto),
    ("file", Conv::Path),
    ("url", Conv::Uri),
    ("encoding", Conv::Encoding),
    ("class", Conv::ClassOption),
    ("name", Conv::Unchanged),
    ("align", Conv::Choice(H_ALIGN_VALUES)),
    ("delim", Conv::SingleCharOrWhitespaceOrUnicode),
    ("keepspace", Conv::Flag),
    ("quote", Conv::SingleCharOrUnicode),
    ("escape", Conv::SingleCharOrUnicode),
];

const LIST_TABLE_OPTS: &[(&str, Conv)] = &[
    ("header-rows", Conv::NonnegativeInt),
    ("stub-columns", Conv::NonnegativeInt),
    ("width", Conv::LengthOrPercentageOrUnitless("")),
    ("widths", Conv::WidthsAuto),
    ("class", Conv::ClassOption),
    ("name", Conv::Unchanged),
    ("align", Conv::Choice(H_ALIGN_VALUES)),
];

/// Python csv.reader over the option-configured dialect
/// (tables.py DocutilsDialect): doublequote unless an escapechar is set,
/// skipinitialspace unless :keepspace:, quoted cells may span lines.
fn parse_csv_text(
    text: &str,
    delim: char,
    quote: char,
    escape: Option<char>,
    doublequote: bool,
    skipinitialspace: bool,
) -> Vec<Vec<String>> {
    let mut rows: Vec<Vec<String>> = Vec::new();
    let mut row: Vec<String> = Vec::new();
    let mut cell = String::new();
    let mut in_quotes = false;
    let mut cell_started = false;
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if in_quotes {
            if Some(c) == escape {
                if let Some(n) = chars.next() {
                    cell.push(n);
                }
            } else if c == quote {
                if doublequote && chars.peek() == Some(&quote) {
                    chars.next();
                    cell.push(quote);
                } else {
                    in_quotes = false;
                }
            } else {
                cell.push(c);
            }
            continue;
        }
        match c {
            c if c == quote && !cell_started => {
                in_quotes = true;
                cell_started = true;
            }
            c if c == delim => {
                row.push(std::mem::take(&mut cell));
                cell_started = false;
                if skipinitialspace {
                    while chars.peek() == Some(&' ') {
                        chars.next();
                    }
                }
            }
            '\n' => {
                row.push(std::mem::take(&mut cell));
                rows.push(std::mem::take(&mut row));
                cell_started = false;
            }
            c => {
                cell.push(c);
                cell_started = true;
            }
        }
    }
    if !cell.is_empty() || !row.is_empty() {
        row.push(cell);
        rows.push(row);
    }
    rows
}

/// The docutils Directive class contract (rst/__init__.py:305-318).
#[derive(Clone, Copy)]
struct DirectiveSpec {
    required_arguments: usize,
    optional_arguments: usize,
    final_argument_whitespace: bool,
    has_content: bool,
    option_spec: &'static [(&'static str, Conv)],
    kind: DirectiveKind,
}

/// Option converters (directives/__init__.py:156-481). Each mirrors one
/// docutils conversion function, including its exact error text.
#[derive(Clone, Copy)]
enum Conv {
    Flag,
    Unchanged,
    UnchangedRequired,
    NonnegativeInt,
    Percentage,
    LengthOrUnitless,
    /// The &str is the docutils `default` unit suffix appended to unitless
    /// values ("" for image width, "px" for figwidth).
    LengthOrPercentageOrUnitless(&'static str),
    ClassOption,
    Choice(&'static [&'static str]),
    Path,
    Uri,
    /// codecs.lookup validation is approximated as accept-any (hardening
    /// note: exotic names docutils rejects are accepted here).
    Encoding,
    /// figure :figwidth:: the literal 'image' keyword or a length.
    Figwidth,
    /// Plain Python int() — negatives allowed (sphinx maxdepth).
    PyIntAny,
    SingleCharOrUnicode,
    SingleCharOrWhitespaceOrUnicode,
    /// value_or(('auto', 'grid'), positive_int_list) — the table :widths:.
    WidthsAutoGrid,
    /// value_or(('auto',), positive_int_list) — csv/list-table :widths:.
    WidthsAuto,
    /// positive_int as used by the widths list elements.
    PositiveIntForList,
}

/// Converted option values (Python-typed in docutils: None/str/int/list).
#[derive(Clone, Debug, PartialEq)]
enum OptVal {
    /// flag options convert to Python None.
    Null,
    Str(String),
    Int(i64),
    IntList(Vec<i64>),
    StrList(Vec<String>),
}

/// The arguments/options/content/etc. handed to a directive's run().
struct DirectiveInput<'r> {
    /// The directive name AS WRITTEN (docutils self.name; error messages
    /// reproduce the original case).
    name: &'r str,
    arguments: Vec<String>,
    options: Vec<(String, OptVal)>,
    content: Vec<LineRec>,
    span: Span,
    lineno: u32,
    rawsource: &'r str,
}

fn opt_get<'o>(options: &'o [(String, OptVal)], name: &str) -> Option<&'o OptVal> {
    options.iter().find(|(n, _)| n == name).map(|(_, v)| v)
}

const ADMONITION_OPTS: &[(&str, Conv)] = &[("class", Conv::ClassOption), ("name", Conv::Unchanged)];

const SIDEBAR_OPTS: &[(&str, Conv)] = &[
    ("subtitle", Conv::UnchangedRequired),
    ("class", Conv::ClassOption),
    ("name", Conv::Unchanged),
];

const NAME_ONLY_OPTS: &[(&str, Conv)] = &[("name", Conv::Unchanged)];

const IMAGE_ALIGN_VALUES: &[&str] = &["top", "middle", "bottom", "left", "center", "right"];
const IMAGE_LOADING_VALUES: &[&str] = &["embed", "link", "lazy"];
const IMAGE_OPTS: &[(&str, Conv)] = &[
    ("alt", Conv::Unchanged),
    ("height", Conv::LengthOrUnitless),
    ("width", Conv::LengthOrPercentageOrUnitless("")),
    ("scale", Conv::Percentage),
    ("align", Conv::Choice(IMAGE_ALIGN_VALUES)),
    ("target", Conv::UnchangedRequired),
    ("loading", Conv::Choice(IMAGE_LOADING_VALUES)),
    ("class", Conv::ClassOption),
    ("name", Conv::Unchanged),
];

const H_ALIGN_VALUES: &[&str] = &["left", "center", "right"];
const FIGURE_OPTS: &[(&str, Conv)] = &[
    ("alt", Conv::Unchanged),
    ("height", Conv::LengthOrUnitless),
    ("width", Conv::LengthOrPercentageOrUnitless("")),
    ("scale", Conv::Percentage),
    ("align", Conv::Choice(H_ALIGN_VALUES)),
    ("target", Conv::UnchangedRequired),
    ("loading", Conv::Choice(IMAGE_LOADING_VALUES)),
    ("class", Conv::ClassOption),
    ("name", Conv::Unchanged),
    ("figwidth", Conv::Figwidth),
    ("figclass", Conv::ClassOption),
];

const CODE_OPTS: &[(&str, Conv)] = &[
    ("class", Conv::ClassOption),
    ("name", Conv::Unchanged),
    ("number-lines", Conv::Unchanged),
];

const RAW_OPTS: &[(&str, Conv)] = &[
    ("file", Conv::Path),
    ("url", Conv::Uri),
    ("encoding", Conv::Encoding),
    ("class", Conv::ClassOption),
];

fn directive_spec(lower: &str) -> Option<DirectiveSpec> {
    let adm = |k: &'static str| {
        Some(DirectiveSpec {
            required_arguments: 0,
            optional_arguments: 0,
            final_argument_whitespace: true,
            has_content: true,
            option_spec: ADMONITION_OPTS,
            kind: DirectiveKind::Admonition(k),
        })
    };
    match lower {
        "note" => adm("note"),
        "warning" => adm("warning"),
        "tip" => adm("tip"),
        "hint" => adm("hint"),
        "important" => adm("important"),
        "caution" => adm("caution"),
        "danger" => adm("danger"),
        "error" => adm("error"),
        "attention" => adm("attention"),
        "admonition" => Some(DirectiveSpec {
            required_arguments: 1,
            optional_arguments: 0,
            final_argument_whitespace: true,
            has_content: true,
            option_spec: ADMONITION_OPTS,
            kind: DirectiveKind::GenericAdmonition,
        }),
        "image" => Some(DirectiveSpec {
            required_arguments: 1,
            optional_arguments: 0,
            final_argument_whitespace: true,
            has_content: false,
            option_spec: IMAGE_OPTS,
            kind: DirectiveKind::Image,
        }),
        "topic" => Some(DirectiveSpec {
            required_arguments: 1,
            optional_arguments: 0,
            final_argument_whitespace: true,
            has_content: true,
            option_spec: ADMONITION_OPTS,
            kind: DirectiveKind::PseudoSection("topic"),
        }),
        "sidebar" => Some(DirectiveSpec {
            required_arguments: 0,
            optional_arguments: 1,
            final_argument_whitespace: true,
            has_content: true,
            option_spec: SIDEBAR_OPTS,
            kind: DirectiveKind::PseudoSection("sidebar"),
        }),
        "rubric" => Some(DirectiveSpec {
            required_arguments: 1,
            optional_arguments: 0,
            final_argument_whitespace: true,
            has_content: false,
            option_spec: ADMONITION_OPTS,
            kind: DirectiveKind::Rubric,
        }),
        "epigraph" => Some(quote_class_spec("epigraph")),
        "highlights" => Some(quote_class_spec("highlights")),
        "pull-quote" => Some(quote_class_spec("pull-quote")),
        "compound" => Some(DirectiveSpec {
            required_arguments: 0,
            optional_arguments: 0,
            final_argument_whitespace: false,
            has_content: true,
            option_spec: ADMONITION_OPTS,
            kind: DirectiveKind::Compound,
        }),
        "container" => Some(DirectiveSpec {
            required_arguments: 0,
            optional_arguments: 1,
            final_argument_whitespace: true,
            has_content: true,
            option_spec: NAME_ONLY_OPTS,
            kind: DirectiveKind::Container,
        }),
        "parsed-literal" => Some(DirectiveSpec {
            required_arguments: 0,
            optional_arguments: 0,
            final_argument_whitespace: false,
            has_content: true,
            option_spec: ADMONITION_OPTS,
            kind: DirectiveKind::ParsedLiteral,
        }),
        "figure" => Some(DirectiveSpec {
            required_arguments: 1,
            optional_arguments: 0,
            final_argument_whitespace: true,
            has_content: true,
            option_spec: FIGURE_OPTS,
            kind: DirectiveKind::Figure,
        }),
        "code" | "code-block" | "sourcecode" => Some(DirectiveSpec {
            required_arguments: 0,
            optional_arguments: 1,
            final_argument_whitespace: false,
            has_content: true,
            option_spec: CODE_OPTS,
            kind: DirectiveKind::Code,
        }),
        "math" => Some(DirectiveSpec {
            required_arguments: 0,
            optional_arguments: 0,
            final_argument_whitespace: false,
            has_content: true,
            option_spec: ADMONITION_OPTS,
            kind: DirectiveKind::MathBlock,
        }),
        "raw" => Some(DirectiveSpec {
            required_arguments: 1,
            optional_arguments: 0,
            final_argument_whitespace: true,
            has_content: true,
            option_spec: RAW_OPTS,
            kind: DirectiveKind::Raw,
        }),
        "line-block" => Some(DirectiveSpec {
            required_arguments: 0,
            optional_arguments: 0,
            final_argument_whitespace: false,
            has_content: true,
            option_spec: ADMONITION_OPTS,
            kind: DirectiveKind::LineBlockDir,
        }),
        // en-alias table entries whose canonical directive is implemented
        // (languages/en.py: code-block/sourcecode -> code, rst-class ->
        // class, section-numbering -> sectnum [unimplemented]).
        "class" | "rst-class" => Some(DirectiveSpec {
            required_arguments: 1,
            optional_arguments: 0,
            final_argument_whitespace: true,
            has_content: true,
            option_spec: &[],
            kind: DirectiveKind::ClassDir,
        }),
        "table" => Some(DirectiveSpec {
            required_arguments: 0,
            optional_arguments: 1,
            final_argument_whitespace: true,
            has_content: true,
            option_spec: TABLE_OPTS,
            kind: DirectiveKind::RstTable,
        }),
        "csv-table" => Some(DirectiveSpec {
            required_arguments: 0,
            optional_arguments: 1,
            final_argument_whitespace: true,
            has_content: true,
            option_spec: CSV_TABLE_OPTS,
            kind: DirectiveKind::CsvTable,
        }),
        "list-table" => Some(DirectiveSpec {
            required_arguments: 0,
            optional_arguments: 1,
            final_argument_whitespace: true,
            has_content: true,
            option_spec: LIST_TABLE_OPTS,
            kind: DirectiveKind::ListTable,
        }),
        "replace" => Some(DirectiveSpec {
            required_arguments: 0,
            optional_arguments: 0,
            final_argument_whitespace: false,
            has_content: true,
            option_spec: &[],
            kind: DirectiveKind::Replace,
        }),
        "unicode" => Some(DirectiveSpec {
            required_arguments: 1,
            optional_arguments: 0,
            final_argument_whitespace: true,
            has_content: false,
            option_spec: UNICODE_OPTS,
            kind: DirectiveKind::UnicodeDir,
        }),
        "date" => Some(DirectiveSpec {
            required_arguments: 0,
            optional_arguments: 0,
            final_argument_whitespace: false,
            has_content: true,
            option_spec: &[],
            kind: DirectiveKind::DateDir,
        }),
        _ => None,
    }
}

/// epigraph/highlights/pull-quote: content-only, NO options at all
/// (body.py:257-283 — option_spec is not declared).
fn quote_class_spec(class: &'static str) -> DirectiveSpec {
    DirectiveSpec {
        required_arguments: 0,
        optional_arguments: 0,
        final_argument_whitespace: false,
        has_content: true,
        option_spec: &[],
        kind: DirectiveKind::QuoteClass(class),
    }
}

/// parse_directive_arguments (states.py:2365-2380).
fn parse_directive_arguments(arg_text: &str, spec: &DirectiveSpec) -> Result<Vec<String>, String> {
    let required = spec.required_arguments;
    let optional = spec.optional_arguments;
    let words: Vec<&str> = arg_text.split_whitespace().collect();
    if words.len() < required {
        return Err(format!(
            "{} argument(s) required, {} supplied",
            required,
            words.len()
        ));
    }
    if words.len() > required + optional {
        if spec.final_argument_whitespace {
            return Ok(py_split_max(arg_text, required + optional - 1));
        }
        return Err(format!(
            "maximum {} argument(s) allowed, {} supplied",
            required + optional,
            words.len()
        ));
    }
    Ok(words.iter().map(|w| w.to_string()).collect())
}

/// Python `str.split(None, maxsplit)`: whitespace runs separate the first
/// `maxsplit` tokens; the remainder keeps internal whitespace verbatim.
fn py_split_max(text: &str, maxsplit: usize) -> Vec<String> {
    let mut out = Vec::new();
    let mut rest = text.trim_start();
    for _ in 0..maxsplit {
        if rest.is_empty() {
            return out;
        }
        match rest.find(char::is_whitespace) {
            Some(i) => {
                out.push(rest[..i].to_string());
                rest = rest[i..].trim_start();
            }
            None => {
                out.push(rest.to_string());
                return out;
            }
        }
    }
    if !rest.is_empty() {
        out.push(rest.to_string());
    }
    out
}

/// parse_extension_options + extract_options + assemble_option_dict
/// (states.py:2382-2413, utils.py:274-369). Errors return the MarkupError
/// detail string; the caller adds the 'Error in "X" directive:' wrapper.
fn parse_extension_options(
    sources: &SourceTable,
    opt_block: &[LineRec],
    option_spec: &'static [(&'static str, Conv)],
) -> Result<Vec<(String, OptVal)>, String> {
    // Pass 1 (extract_options): collect (lowercased name, body) fields.
    // A multi-word field name errors during this pass, in field order.
    let mut fields: Vec<(String, Option<String>)> = Vec::new();
    let mut i = 0usize;
    while i < opt_block.len() {
        let l = opt_block[i];
        let l_text = sources.line_text(l);
        let marker = if l.indent() == 0 {
            field_marker(l_text)
        } else {
            None
        };
        let Some((raw_name, body_start)) = marker else {
            return Err("invalid option block".to_string());
        };
        let mut body_lines: Vec<&str> = Vec::new();
        let first = l_text[body_start..].trim_start_matches(' ');
        if !first.is_empty() {
            body_lines.push(first);
        }
        // Continuation lines (any deeper indent) join the field body,
        // dedented by their common indent, '\n'-separated.
        let mut j = i + 1;
        while j < opt_block.len() && opt_block[j].indent() > 0 {
            j += 1;
        }
        let conts = &opt_block[i + 1..j];
        let min_indent = conts.iter().map(|c| c.indent()).min().unwrap_or(0);
        for c in conts {
            body_lines.push(&sources.line_text(*c)[min_indent.min(c.indent())..]);
        }
        if raw_name.split_whitespace().count() != 1 {
            return Err(
                "invalid option data: extension option field name may not contain multiple words"
                    .to_string(),
            );
        }
        let body = if body_lines.is_empty() {
            None
        } else {
            Some(body_lines.join("\n"))
        };
        fields.push((raw_name.to_lowercase(), body));
        i = j;
    }
    // Pass 2 (assemble_option_dict): unknown, then duplicate, then convert.
    let mut out: Vec<(String, OptVal)> = Vec::new();
    for (name, body) in &fields {
        let Some((_, conv)) = option_spec.iter().find(|(n, _)| n == name) else {
            return Err(format!("unknown option: \"{name}\""));
        };
        if out.iter().any(|(n, _)| n == name) {
            return Err(format!("invalid option data: duplicate option \"{name}\""));
        }
        match convert_option(*conv, body.as_deref()) {
            Ok(v) => out.push((name.clone(), v)),
            Err(detail) => {
                return Err(format!(
                    "invalid option value: (option: \"{}\"; value: {})\n{}",
                    name,
                    py_repr(body.as_deref()),
                    detail
                ));
            }
        }
    }
    Ok(out)
}

fn convert_option(conv: Conv, value: Option<&str>) -> Result<OptVal, String> {
    match conv {
        Conv::Flag => match value {
            Some(v) if !v.trim().is_empty() => {
                Err(format!("no argument is allowed; \"{v}\" supplied"))
            }
            _ => Ok(OptVal::Null),
        },
        Conv::PyIntAny => {
            let Some(v) = value else {
                return Err(
                    "int() argument must be a string, a bytes-like object or a real number, not 'NoneType'"
                        .to_string(),
                );
            };
            match py_int_canonical(v) {
                Some((neg, digits)) => Ok(int_optval(neg, &digits)),
                None => Err(format!(
                    "invalid literal for int() with base 10: {}",
                    py_repr(Some(v))
                )),
            }
        }
        Conv::NonnegativeInt => {
            let Some(v) = value else {
                return Err(
                    "int() argument must be a string, a bytes-like object or a real number, not 'NoneType'"
                        .to_string(),
                );
            };
            nonnegative_int(v)
        }
        Conv::SingleCharOrUnicode | Conv::SingleCharOrWhitespaceOrUnicode => {
            let Some(v) = value else {
                return Err("argument required but none supplied".to_string());
            };
            if matches!(conv, Conv::SingleCharOrWhitespaceOrUnicode) {
                if v == "tab" {
                    return Ok(OptVal::Str("\t".to_string()));
                }
                if v == "space" {
                    return Ok(OptVal::Str(" ".to_string()));
                }
            }
            let decoded = unicode_code(v)?;
            if decoded.chars().count() != 1 {
                return Err(format!(
                    "{} invalid; must be a single character or a Unicode code",
                    py_repr(Some(&decoded))
                ));
            }
            Ok(OptVal::Str(decoded))
        }
        Conv::WidthsAutoGrid | Conv::WidthsAuto => {
            let Some(v) = value else {
                return Err("argument required but none supplied".to_string());
            };
            let keywords: &[&str] = if matches!(conv, Conv::WidthsAutoGrid) {
                &["auto", "grid"]
            } else {
                &["auto"]
            };
            if keywords.contains(&v) {
                return Ok(OptVal::Str(v.to_string()));
            }
            let parts: Vec<&str> = if v.contains(',') {
                v.split(',').collect()
            } else {
                v.split_whitespace().collect()
            };
            let mut list = Vec::new();
            for p in parts {
                match convert_option(Conv::PositiveIntForList, Some(p.trim()))? {
                    OptVal::Int(n) => list.push(n),
                    _ => unreachable!(),
                }
            }
            Ok(OptVal::IntList(list))
        }
        Conv::PositiveIntForList => {
            let Some(v) = value else {
                return Err("argument required but none supplied".to_string());
            };
            match py_int(v) {
                Some(n) if n >= 1 => Ok(OptVal::Int(n)),
                Some(_) => Err("negative or zero value; must be positive".to_string()),
                None => Err(format!(
                    "invalid literal for int() with base 10: {}",
                    py_repr(Some(v))
                )),
            }
        }
        Conv::Unchanged => Ok(OptVal::Str(value.unwrap_or("").to_string())),
        Conv::UnchangedRequired => match value {
            None => Err("argument required but none supplied".to_string()),
            Some(v) => Ok(OptVal::Str(v.to_string())),
        },
        Conv::Percentage => {
            // percentage(): rstrip(' %'), then nonnegative_int; None slips
            // through to int(None)'s TypeError (directives/__init__.py:235).
            let Some(v) = value else {
                return Err(
                    "int() argument must be a string, a bytes-like object or a real number, not 'NoneType'"
                        .to_string(),
                );
            };
            nonnegative_int(v.trim_end_matches([' ', '%']))
        }
        Conv::LengthOrUnitless => {
            let Some(v) = value else {
                return Err("expected string or bytes-like object, got 'NoneType'".to_string());
            };
            let mut units: Vec<&str> = CSS3_LENGTH_UNITS.to_vec();
            units.push("");
            get_measure(v, &units).map(OptVal::Str)
        }
        Conv::LengthOrPercentageOrUnitless(default) => {
            let Some(v) = value else {
                return Err("expected string or bytes-like object, got 'NoneType'".to_string());
            };
            let mut units: Vec<&str> = CSS3_LENGTH_UNITS.to_vec();
            units.push("%");
            match get_measure(v, &units) {
                Ok(m) => Ok(OptVal::Str(m)),
                Err(first_error) => match get_measure(v, &[""]) {
                    Ok(m) => Ok(OptVal::Str(format!("{m}{default}"))),
                    Err(_) => Err(first_error),
                },
            }
        }
        Conv::Path => {
            let Some(v) = value else {
                return Err("argument required but none supplied".to_string());
            };
            Ok(OptVal::Str(
                v.lines().map(str::trim).collect::<Vec<_>>().join(""),
            ))
        }
        Conv::Uri => {
            let Some(v) = value else {
                return Err("argument required but none supplied".to_string());
            };
            Ok(OptVal::Str(uri_from_argument(v)))
        }
        Conv::Encoding => {
            let Some(v) = value else {
                return Err("argument required but none supplied".to_string());
            };
            Ok(OptVal::Str(v.to_string()))
        }
        Conv::Figwidth => {
            let Some(v) = value else {
                return Err("expected string or bytes-like object, got 'NoneType'".to_string());
            };
            if v.eq_ignore_ascii_case("image") {
                return Ok(OptVal::Str("image".to_string()));
            }
            convert_option(Conv::LengthOrPercentageOrUnitless("px"), Some(v))
        }
        Conv::ClassOption => {
            let Some(v) = value else {
                return Err("argument required but none supplied".to_string());
            };
            let mut names = Vec::new();
            for word in v.split_whitespace() {
                let id = ids::make_id(word);
                if id.is_empty() {
                    return Err(format!("cannot make \"{word}\" into a class name"));
                }
                names.push(id);
            }
            Ok(OptVal::StrList(names))
        }
        Conv::Choice(values) => {
            let Some(v) = value else {
                return Err(format!(
                    "must supply an argument; choose from {}",
                    format_choice_values(values)
                ));
            };
            let lowered = v.trim().to_lowercase();
            if values.contains(&lowered.as_str()) {
                Ok(OptVal::Str(lowered))
            } else {
                Err(format!(
                    "\"{v}\" unknown; choose from {}",
                    format_choice_values(values)
                ))
            }
        }
    }
}

/// format_values (directives/__init__.py:448-450).
fn format_choice_values(values: &[&str]) -> String {
    let init = values[..values.len() - 1]
        .iter()
        .map(|v| format!("\"{v}\""))
        .collect::<Vec<_>>()
        .join(", ");
    format!("{}, or \"{}\"", init, values[values.len() - 1])
}

/// unicode_code (directives/__init__.py:330-352): decimal, hex forms
/// (0x/x/\x/U+/\u/&#x...;), or the text itself when neither matches.
fn unicode_code(code: &str) -> Result<String, String> {
    // Python gates on str.isdigit() (Nd digits AND digit-typed No chars
    // like '²'), then int() — which only accepts the Nd ones.
    if !code.is_empty() && code.chars().all(super::digits::is_python_digit) {
        let Some((false, digits)) = py_int_canonical(code) else {
            return Err(format!(
                "invalid literal for int() with base 10: {}",
                py_repr(Some(code))
            ));
        };
        let n: u32 = digits
            .parse()
            .map_err(|_| format!("code too large ({code})"))?;
        return char::from_u32(n)
            .map(|c| c.to_string())
            .ok_or_else(|| "chr() arg not in range(0x110000)".to_string());
    }
    let lower = code.to_lowercase();
    let hex = ["0x", "x", "\\x", "u+", "u", "\\u"]
        .iter()
        .find_map(|p| lower.strip_prefix(p))
        .filter(|h| !h.is_empty() && h.bytes().all(|b| b.is_ascii_hexdigit()))
        .map(|h| h.to_string())
        .or_else(|| {
            lower
                .strip_prefix("&#x")
                .and_then(|h| h.strip_suffix(';'))
                .filter(|h| !h.is_empty() && h.bytes().all(|b| b.is_ascii_hexdigit()))
                .map(|h| h.to_string())
        });
    match hex {
        Some(h) => {
            let n = u32::from_str_radix(&h, 16).map_err(|_| format!("code too large ({h})"))?;
            char::from_u32(n)
                .map(|c| c.to_string())
                .ok_or_else(|| "chr() arg not in range(0x110000)".to_string())
        }
        None => Ok(code.to_string()),
    }
}

/// The converted int as an OptVal: i64 when it fits, else the canonical
/// decimal string (Python ints are arbitrary precision; pformat renders
/// both identically).
fn int_optval(neg: bool, digits: &str) -> OptVal {
    let display = py_int_display(neg, digits);
    match display.parse::<i64>() {
        Ok(n) => OptVal::Int(n),
        Err(_) => OptVal::Str(display),
    }
}

/// nonnegative_int (directives/__init__.py:224-231), with Python's own
/// int() error text for bad literals.
fn nonnegative_int(s: &str) -> Result<OptVal, String> {
    match py_int_canonical(s) {
        Some((true, _)) => Err("negative value; must be positive or zero".to_string()),
        Some((false, digits)) => Ok(int_optval(false, &digits)),
        None => Err(format!(
            "invalid literal for int() with base 10: {}",
            py_repr(Some(s))
        )),
    }
}

/// Python int(str), canonicalized: arbitrary precision (the value is the
/// canonical ASCII decimal string), Unicode Nd digits accepted with their
/// decimal values, single underscores allowed BETWEEN digits, surrounding
/// whitespace ignored. Returns (negative, digits-without-sign, canonical
/// leading-zero-stripped ASCII string WITH sign).
fn py_int_canonical(s: &str) -> Option<(bool, String)> {
    let t = s.trim();
    let (neg, body) = match t.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, t.strip_prefix('+').unwrap_or(t)),
    };
    if body.is_empty() {
        return None;
    }
    let mut digits = String::with_capacity(body.len());
    let mut prev_underscore = true; // leading underscore rejected
    for c in body.chars() {
        if c == '_' {
            if prev_underscore {
                return None;
            }
            prev_underscore = true;
            continue;
        }
        let d = super::digits::decimal_digit_value(c)?;
        digits.push(char::from(b'0' + d as u8));
        prev_underscore = false;
    }
    if prev_underscore {
        // trailing underscore (or all-underscores)
        return None;
    }
    let stripped = digits.trim_start_matches('0');
    let canonical = if stripped.is_empty() { "0" } else { stripped };
    Some((neg && canonical != "0", canonical.to_string()))
}

fn py_int_display(neg: bool, digits: &str) -> String {
    if neg {
        format!("-{digits}")
    } else {
        digits.to_string()
    }
}

/// Python int(str) for numeric consumers; None when invalid OR outside
/// i64 (attr-facing paths must use [`py_int_canonical`] to keep exact
/// digits for values Python would carry at arbitrary precision).
fn py_int(s: &str) -> Option<i64> {
    let (neg, digits) = py_int_canonical(s)?;
    py_int_display(neg, &digits).parse::<i64>().ok()
}

/// Python repr() for option-value error messages (strings and None).
pub(crate) fn py_repr(value: Option<&str>) -> String {
    match value {
        None => "None".to_string(),
        Some(s) => {
            let quote = if s.contains('\'') && !s.contains('"') {
                '"'
            } else {
                '\''
            };
            let mut out = String::new();
            out.push(quote);
            for c in s.chars() {
                match c {
                    '\\' => out.push_str("\\\\"),
                    '\n' => out.push_str("\\n"),
                    '\r' => out.push_str("\\r"),
                    '\t' => out.push_str("\\t"),
                    c if c == quote => {
                        out.push('\\');
                        out.push(c);
                    }
                    c => out.push(c),
                }
            }
            out.push(quote);
            out
        }
    }
}

/// CSS3_LENGTH_UNITS (directives/__init__.py:247-248).
const CSS3_LENGTH_UNITS: &[&str] = &[
    "em", "ex", "ch", "rem", "vw", "vh", "vmin", "vmax", "cm", "mm", "Q", "in", "pt", "pc", "px",
];

/// get_measure (directives/__init__.py:260-274) over nodes.parse_measure
/// (nodes.py:3084-3107). Returns the normalized `{value}{unit}` string.
fn get_measure(argument: &str, units: &[&str]) -> Result<String, String> {
    let no_valid = || format!("\"{argument}\" is no valid measure.");
    // fullmatch: (-?[0-9.]+) *([a-zA-Zµ]*|%?)
    let s = argument;
    let digits_start = if s.starts_with('-') { 1 } else { 0 };
    let mut j = digits_start;
    let bytes = s.as_bytes();
    while j < bytes.len() && (bytes[j].is_ascii_digit() || bytes[j] == b'.') {
        j += 1;
    }
    if j == digits_start {
        return Err(no_valid());
    }
    let number = &s[..j];
    let mut k = j;
    while k < bytes.len() && bytes[k] == b' ' {
        k += 1;
    }
    let unit = &s[k..];
    let unit_ok = unit == "%" || unit.chars().all(|c| c.is_ascii_alphabetic() || c == 'µ');
    if !unit_ok {
        return Err(no_valid());
    }
    // Python: int() first (arbitrary precision — exact digits preserved),
    // float() second; negative or unlisted unit is the units-list error.
    let (negative, norm) = if let Some((neg, digits)) = py_int_canonical(number) {
        (neg, py_int_display(neg, &digits))
    } else if let Ok(f) = number.parse::<f64>() {
        (f < 0.0, py_float_str(f))
    } else {
        return Err(no_valid());
    };
    if negative || !units.contains(&unit) {
        return Err(format!(
            "not a positive number or measure of one of the following units:\n{}",
            units
                .iter()
                .filter(|u| !u.is_empty())
                .copied()
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    Ok(format!("{norm}{unit}"))
}

/// Python float repr for simple decimals (1.0 -> "1.0", 1.5 -> "1.5").
fn py_float_str(f: f64) -> String {
    if f == f.trunc() && f.abs() < 1e16 {
        format!("{f:.1}")
    } else {
        format!("{f}")
    }
}

/// directives.uri (directives/__init__.py:209-221): unescaped whitespace is
/// removed; backslash-escaped whitespace separates space-joined parts.
fn uri_from_argument(argument: &str) -> String {
    let escaped = super::inline::escape2null(argument);
    let mut parts: Vec<&str> = Vec::new();
    for chunk in escaped.split("\x00 ") {
        parts.extend(chunk.split("\x00\n"));
    }
    parts
        .iter()
        .map(|p| {
            super::inline::unescape(p, false)
                .split_whitespace()
                .collect::<String>()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// states.py parse_target (2095-2113) for the image :target: option: a
/// block whose last line ends in `_` may be an indirect reference;
/// otherwise it is a refuri with all whitespace removed.
enum ImageTarget {
    Refname { name: String, refname: String },
    Refuri(String),
}

fn parse_image_target(target: &str) -> ImageTarget {
    let lines: Vec<&str> = target.lines().collect();
    let ends_underscore = lines
        .iter()
        .rev()
        .find(|l| !l.trim().is_empty())
        .map(|l| l.trim().ends_with('_'))
        .unwrap_or(false);
    if ends_underscore {
        let joined = lines.iter().map(|l| l.trim()).collect::<Vec<_>>().join(" ");
        if let Some(data) = reference_data_from_link(&joined) {
            return ImageTarget::Refname {
                name: ids::whitespace_normalize_name(&data),
                refname: ids::fully_normalize_name(&data),
            };
        }
    }
    ImageTarget::Refuri(target.split_whitespace().collect::<String>())
}

/// `|name|` marker in a (possibly line-joined) substitution-def head:
/// `\|(?![ ])(?P<name>.+?)(?<![\s\x00])\|([ ]+|$)` (states.py:1992-2001).
struct SubstMarker {
    name: String,
    /// Byte index where the remainder after the marker + separator spaces
    /// begins (== input length when the marker ends the line).
    remainder_start: usize,
}

fn match_substitution_marker(acc: &str) -> Option<SubstMarker> {
    let cs: Vec<(usize, char)> = acc.char_indices().collect();
    if cs.len() < 3 || cs[0].1 != '|' || cs[1].1 == ' ' {
        return None;
    }
    for k in 2..cs.len() {
        if cs[k].1 != '|' || cs[k - 1].1.is_whitespace() {
            continue;
        }
        let name = acc[cs[1].0..cs[k].0].to_string();
        let after = &acc[cs[k].0 + 1..];
        if after.is_empty() {
            return Some(SubstMarker {
                name,
                remainder_start: acc.len(),
            });
        }
        if after.starts_with(' ') {
            let spaces = after.len() - after.trim_start_matches(' ').len();
            return Some(SubstMarker {
                name,
                remainder_start: cs[k].0 + 1 + spaces,
            });
        }
        // Closing pipe not followed by space/EOL: the non-greedy regex
        // tries a later close.
    }
    None
}

/// SubstitutionDef embedded-directive marker: `(simplename)::( +|$)` —
/// unlike the body-level form, NO space is allowed before `::`.
fn match_embedded_directive(text: &str) -> Option<(String, &str)> {
    let chars: Vec<char> = text.chars().collect();
    let name_len = match_simplename_chars(&chars, 0)?;
    if chars.get(name_len) != Some(&':') || chars.get(name_len + 1) != Some(&':') {
        return None;
    }
    let after = name_len + 2;
    match chars.get(after) {
        None => {}
        Some(' ') => {}
        _ => return None,
    }
    let name: String = chars[..name_len].iter().collect();
    let byte_after = text
        .char_indices()
        .nth(after + 1)
        .map(|(b, _)| b)
        .unwrap_or(text.len());
    Some((name, &text[byte_after..]))
}

fn dedent_by_min(block: &[LineRec]) -> Vec<LineRec> {
    let min = block
        .iter()
        .filter(|l| !l.is_blank())
        .map(|l| l.indent())
        .min()
        .unwrap_or(0);
    block.iter().map(|l| l.dedented(min)).collect()
}

/// docutils nodes.Inline membership for the kinds this parser emits
/// (image/target/raw are genuinely Inline in docutils' class hierarchy).
fn is_inline_kind(kind: &str) -> bool {
    matches!(
        kind,
        "emphasis"
            | "strong"
            | "literal"
            | "reference"
            | "title_reference"
            | "abbreviation"
            | "acronym"
            | "subscript"
            | "superscript"
            | "math"
            | "image"
            | "problematic"
            | "inline"
            | "substitution_reference"
            | "footnote_reference"
            | "citation_reference"
            | "target"
            | "raw"
    )
}

fn tree_any(node: &Node, pred: &dyn Fn(&Node) -> bool) -> bool {
    node.children.iter().any(|c| pred(c) || tree_any(c, pred))
}

fn has_extra_attr(node: &Node, key: &str) -> bool {
    node.attrs.extra.iter().any(|(k, _)| *k == key)
}

/// disallowed_inside_substitution_definitions (states.py:2219-2227),
/// first hit in document order wins.
fn find_disallowed_in_substitution(node: &Node) -> Option<&'static str> {
    for c in &node.children {
        let hit = if c.kind == kinds::REFERENCE && has_extra_attr(c, "anonymous") {
            Some("Anonymous references")
        } else if c.kind == kinds::FOOTNOTE_REFERENCE && has_extra_attr(c, "auto") {
            Some("References to auto-numbered and auto-symbol footnotes")
        } else if !c.attrs.names.is_empty() || !c.attrs.ids.is_empty() {
            Some("Targets (names and identifiers)")
        } else {
            None
        };
        if hit.is_some() {
            return hit;
        }
        if let Some(h) = find_disallowed_in_substitution(c) {
            return Some(h);
        }
    }
    None
}

fn count_subst_defs(node: &Node, name: &str) -> usize {
    let mut c = usize::from(
        node.kind == "substitution_definition" && node.attrs.names.iter().any(|n| n == name),
    );
    for ch in &node.children {
        c += count_subst_defs(ch, name);
    }
    c
}

fn dupname_subst_defs(node: &mut Node, name: &str, remaining: &mut usize) {
    if *remaining == 0 {
        return;
    }
    if node.kind == "substitution_definition" && node.attrs.names.iter().any(|n| n == name) {
        node.attrs.names.retain(|n| n != name);
        node.attrs.dupnames.push(name.to_string());
        *remaining -= 1;
        return;
    }
    for ch in &mut node.children {
        dupname_subst_defs(ch, name, remaining);
        if *remaining == 0 {
            return;
        }
    }
}

/// Like [`reference_name_from_link`] but returns the reference TEXT
/// (simple name or phrase) before normalization — docutils is_reference().
fn reference_data_from_link(link: &str) -> Option<String> {
    let joined = ids::whitespace_normalize_name(link);
    let body = joined.strip_suffix('_')?;
    if body.ends_with('\\') {
        return None;
    }
    if let Some(phrase) = body.strip_prefix('`').and_then(|b| b.strip_suffix('`')) {
        if phrase.is_empty() {
            return None;
        }
        return Some(phrase.to_string());
    }
    if !body.is_empty()
        && !body.ends_with('_')
        && !body.contains(char::is_whitespace)
        && !body.contains('`')
        && !body.contains('\\')
    {
        return Some(body.to_string());
    }
    None
}

/// docutils `simplename` over a char slice (see rst::inline for the
/// pattern description).
fn match_simplename_chars(chars: &[char], at: usize) -> Option<usize> {
    let n = chars.len();
    let mut i = at;
    let atom = |c: char| (c.is_alphanumeric() || c == '_') && c != '_';
    if i >= n || !atom(chars[i]) {
        return None;
    }
    while i < n && atom(chars[i]) {
        i += 1;
    }
    loop {
        if i < n && matches!(chars[i], '-' | '.' | '_' | '+' | ':') {
            let sep_end = i + 1;
            if sep_end < n && atom(chars[sep_end]) {
                i = sep_end + 1;
                while i < n && atom(chars[i]) {
                    i += 1;
                }
                continue;
            }
        }
        break;
    }
    Some(i - at)
}

/// Consume an indented block starting at `start`: lines while blank or
/// indented, up to the LAST indented line (trailing blanks are neither
/// consumed nor included; callers see them).
/// Returns (dedented block, consumed line count, base indent, adjacency
/// terminator line number when the block ends at an adjacent non-blank
/// column-0 line).
fn indented_block(
    lines: &[LineRec],
    start: usize,
) -> (Vec<LineRec>, usize, usize, Option<(u16, u32)>) {
    let mut end = start;
    let mut last_content = None;
    while end < lines.len() {
        let l = lines[end];
        if l.is_blank() {
            end += 1;
            continue;
        }
        if l.indent() > 0 {
            last_content = Some(end);
            end += 1;
        } else {
            break;
        }
    }
    let last_content = match last_content {
        Some(l) => l,
        None => return (Vec::new(), 0, 0, None),
    };
    let block_end = last_content + 1;
    let base = lines[start..block_end]
        .iter()
        .filter(|l| !l.is_blank())
        .map(|l| l.indent())
        .min()
        .unwrap_or(0);
    let block: Vec<LineRec> = lines[start..block_end]
        .iter()
        .map(|l| if l.is_blank() { *l } else { l.dedented(base) })
        .collect();
    let terminator = lines
        .get(block_end)
        .filter(|l| !l.is_blank())
        .map(|l| (l.source, l.lineno));
    (block, block_end - start, base, terminator)
}

fn strip_literal_colons(text: &str) -> (String, bool) {
    if text == "::" {
        return (String::new(), true);
    }
    if let Some(head) = text.strip_suffix("::") {
        if head.is_empty() {
            return (String::new(), true);
        }
        let last = head.chars().last().unwrap();
        if last == ' ' || last == '\n' {
            return (head.trim_end().to_string(), true);
        }
        return (text[..text.len() - 1].to_string(), true);
    }
    (text.to_string(), false)
}

fn attribution_from_chunk(
    sources: &SourceTable,
    chunk: &[LineRec],
    span: Span,
) -> Option<(Node, u32)> {
    let first = chunk.first()?;
    if first.indent() != 0 {
        return None;
    }
    let first_text = sources.line_text(*first);
    // Fixture-verified marker rules: `--`/`---` (not followed by another
    // hyphen) or an em dash, then ZERO or more spaces (all consumed), then
    // non-space text.
    let after = match first_text.strip_prefix('\u{2014}') {
        Some(r) => r,
        None => {
            // `---` then `--`; a further hyphen means an adornment, not a
            // marker. The `---` arm runs first, so the `--` arm's remainder
            // can only start with `-` for exactly `---x`-shaped input.
            let r = first_text
                .strip_prefix("---")
                .or_else(|| first_text.strip_prefix("--"))?;
            if r.starts_with('-') {
                return None;
            }
            r
        }
    };
    let rest = after.trim_start_matches(' ');
    if rest.is_empty() {
        return None;
    }
    // Continuation lines must share ONE uniform indent (else the chunk is
    // not an attribution at all) and dedent by exactly that indent.
    let mut text = rest.to_string();
    if chunk.len() > 1 {
        let indent = chunk[1].indent();
        for l in &chunk[1..] {
            if l.indent() != indent {
                return None;
            }
        }
        for l in &chunk[1..] {
            text.push('\n');
            text.push_str(&sources.line_text(*l)[indent..]);
        }
    }
    let mut attribution = Node::elem(kinds::ATTRIBUTION, span);
    attribution.children.push(Node::text_node(text, span));
    Some((attribution, first.lineno))
}

fn build_line_block(items: &mut [(usize, Vec<Node>)], span: Span, guard: usize) -> Node {
    let mut lb = Node::elem(kinds::LINE_BLOCK, span);
    // Totality guard mirroring MAX_NEST_DEPTH: absurd nesting flattens
    // instead of overflowing the stack (docutils crashes here).
    if guard >= MAX_NEST_DEPTH {
        for (_, children) in items.iter_mut() {
            let mut line = Node::elem(kinds::LINE, span);
            line.children = std::mem::take(children);
            lb.children.push(line);
        }
        return lb;
    }
    let base = items.iter().map(|(d, _)| *d).min().unwrap_or(0);
    let mut i = 0usize;
    while i < items.len() {
        if items[i].0 <= base {
            let mut line = Node::elem(kinds::LINE, span);
            line.children = std::mem::take(&mut items[i].1);
            lb.children.push(line);
            i += 1;
        } else {
            let run_start = i;
            while i < items.len() && items[i].0 > base {
                i += 1;
            }
            lb.children
                .push(build_line_block(&mut items[run_start..i], span, guard + 1));
        }
    }
    lb
}

// ----------------------------------------------------------------------
// enumerators
// ----------------------------------------------------------------------

#[derive(Debug, Clone)]
struct Enumerator {
    literal: String,
    prefix: &'static str,
    suffix: &'static str,
    auto: bool,
    /// Marker followed by end-of-line with no text (fixture-verified: valid
    /// for a lone first item, never for a successor).
    rest_empty: bool,
    /// Characters the marker occupies (prefix + literal + suffix).
    marker_chars: usize,
}

/// One possible (sequence, ordinal) interpretation of a list so far.
/// `initial` is the first item's ordinal under this sequence; `current` the
/// most recent item's. Priority order = docutils resolution order.
#[derive(Debug, Clone)]
struct EnumCandidate {
    seq: &'static str,
    initial: u64,
    current: u64,
}

fn roman_value(text: &str, lower: bool) -> Option<u64> {
    let t: String = if lower {
        text.to_string()
    } else {
        text.to_lowercase()
    };
    if t.is_empty() {
        return None;
    }
    // canonical: m{0,4}(cm|cd|d?c{0,3})(xc|xl|l?x{0,3})(ix|iv|v?i{0,3})
    let mut rest = t.as_str();
    let mut value = 0u64;
    let mut m_count = 0;
    while rest.starts_with('m') && m_count < 4 {
        value += 1000;
        rest = &rest[1..];
        m_count += 1;
    }
    for (nine, four, five, one, unit) in [
        ("cm", "cd", 'd', 'c', 100u64),
        ("xc", "xl", 'l', 'x', 10u64),
        ("ix", "iv", 'v', 'i', 1u64),
    ] {
        if let Some(r) = rest.strip_prefix(nine) {
            value += 9 * unit;
            rest = r;
            continue;
        }
        if let Some(r) = rest.strip_prefix(four) {
            value += 4 * unit;
            rest = r;
            continue;
        }
        if rest.starts_with(five) {
            value += 5 * unit;
            rest = &rest[1..];
        }
        let mut ones = 0;
        while rest.starts_with(one) && ones < 3 {
            value += unit;
            rest = &rest[1..];
            ones += 1;
        }
    }
    if rest.is_empty() && value > 0 {
        Some(value)
    } else {
        None
    }
}

/// Ordinal of `body` interpreted in a KNOWN sequence.
fn ordinal_in_sequence(body: &str, seq: &str) -> Option<u64> {
    match seq {
        "arabic" => body.parse::<u64>().ok(),
        "loweralpha" => {
            let mut chars = body.chars();
            let c = chars.next()?;
            (chars.next().is_none() && c.is_ascii_lowercase())
                .then(|| (c as u64) - ('a' as u64) + 1)
        }
        "upperalpha" => {
            let mut chars = body.chars();
            let c = chars.next()?;
            (chars.next().is_none() && c.is_ascii_uppercase())
                .then(|| (c as u64) - ('A' as u64) + 1)
        }
        "lowerroman" => roman_value(body, true),
        "upperroman" => roman_value(body, false),
        _ => None,
    }
}

/// Candidate interpretations of a FIRST enumerator, in docutils resolution
/// priority (probe-verified: `i`/`I` prefer roman; all other single letters
/// prefer alpha; multi-char roman must be canonically valid).
fn initial_candidates(body: &str, auto: bool) -> Vec<EnumCandidate> {
    let mk = |seq: &'static str, n: u64| EnumCandidate {
        seq,
        initial: n,
        current: n,
    };
    if auto {
        return vec![mk("arabic", 1)];
    }
    if body.chars().all(|c| c.is_ascii_digit()) && !body.is_empty() {
        return body
            .parse::<u64>()
            .ok()
            .filter(|v| *v <= i64::MAX as u64)
            .map(|v| vec![mk("arabic", v)])
            .unwrap_or_default();
    }
    let chars: Vec<char> = body.chars().collect();
    if chars.len() == 1 {
        // Probe-verified: single-letter firsts have NO ambiguity in docutils
        // 0.22.4 — 'i'/'I' are roman(1) ONLY ("i. x\nj. y" is a paragraph),
        // every other letter is alpha ONLY ("v. five\nvi. six" is a
        // paragraph). Successors reinterpret via ordinal_in_sequence, which
        // is how "h. i. j." stays alpha.
        let c = chars[0];
        return match c {
            'i' => vec![mk("lowerroman", 1)],
            'I' => vec![mk("upperroman", 1)],
            _ if c.is_ascii_lowercase() => {
                vec![mk("loweralpha", (c as u64) - ('a' as u64) + 1)]
            }
            _ if c.is_ascii_uppercase() => {
                vec![mk("upperalpha", (c as u64) - ('A' as u64) + 1)]
            }
            _ => Vec::new(),
        };
    }
    if chars.iter().all(|c| "ivxlcdm".contains(*c)) {
        if let Some(v) = roman_value(body, true) {
            return vec![mk("lowerroman", v)];
        }
    }
    if chars.iter().all(|c| "IVXLCDM".contains(*c)) {
        if let Some(v) = roman_value(body, false) {
            return vec![mk("upperroman", v)];
        }
    }
    Vec::new()
}

/// Narrow candidates by the next item's enumerator; ordinals advance.
fn advance_candidates(candidates: &[EnumCandidate], next: &Enumerator) -> Vec<EnumCandidate> {
    candidates
        .iter()
        .filter_map(|c| {
            let expected = c.current + 1;
            let ok = next.auto || ordinal_in_sequence(&next.literal, c.seq) == Some(expected);
            ok.then_some(EnumCandidate {
                seq: c.seq,
                initial: c.initial,
                current: expected,
            })
        })
        .collect()
}

fn parse_enumerator(text: &str) -> Option<Enumerator> {
    let (prefix, after_prefix): (&'static str, &str) = match text.strip_prefix('(') {
        Some(r) => ("(", r),
        None => ("", text),
    };
    let body_end = after_prefix
        .char_indices()
        .find(|(_, c)| !c.is_ascii_alphanumeric() && *c != '#')
        .map(|(i, _)| i)?;
    if body_end == 0 {
        return None;
    }
    let body = &after_prefix[..body_end];
    let after_body = &after_prefix[body_end..];
    let (suffix, rest): (&'static str, &str) = if prefix == "(" {
        (")", after_body.strip_prefix(')')?)
    } else if let Some(r) = after_body.strip_prefix('.') {
        (".", r)
    } else {
        (")", after_body.strip_prefix(')')?)
    };
    if !(rest.is_empty() || rest.starts_with(' ')) {
        return None;
    }
    let auto = body == "#";
    if initial_candidates(body, auto).is_empty() {
        return None;
    }
    Some(Enumerator {
        literal: body.to_string(),
        prefix,
        suffix,
        auto,
        rest_empty: rest.trim().is_empty(),
        marker_chars: prefix.len() + body.len() + 1,
    })
}

// ----------------------------------------------------------------------
// targets
// ----------------------------------------------------------------------

struct TargetMarker {
    name: String,
    anonymous: bool,
    link: String,
}

/// Parse `_name: link`, ``_`name`: link``, `__: link` forms from the
/// (possibly multi-line, newline-joined) text after `..`. Returns None for
/// MALFORMED targets (the caller emits a comment + "malformed hyperlink
/// target." warning): missing colon, colon not followed by space/EOL,
/// empty or unclosed backtick phrase, empty plain name, bare `__`.
fn parse_target_marker(rest: &str) -> Option<TargetMarker> {
    let after = rest.strip_prefix('_')?;
    if let Some(a) = after.strip_prefix('_') {
        // `.. __:` / `.. __: uri` anonymous form; bare `.. __` is malformed.
        let link = a.strip_prefix(':')?;
        if !(link.is_empty() || link.starts_with(' ') || link.starts_with('\n')) {
            return None;
        }
        return Some(TargetMarker {
            name: String::new(),
            anonymous: true,
            link: link.trim().to_string(),
        });
    }
    if let Some(quoted) = after.strip_prefix('`') {
        let close = quoted.find('`')?;
        let name = &quoted[..close];
        if name.is_empty() {
            return None;
        }
        let link = quoted[close + 1..].strip_prefix(':')?;
        if !(link.is_empty() || link.starts_with(' ') || link.starts_with('\n')) {
            return None;
        }
        return Some(TargetMarker {
            name: name.to_string(),
            anonymous: false,
            link: link.trim().to_string(),
        });
    }
    // Plain name: scan to the first unescaped ':', which must be followed by
    // space, newline, or end of input.
    let mut name = String::new();
    let mut chars = after.char_indices();
    while let Some((i, c)) = chars.next() {
        match c {
            '\\' => {
                if let Some((_, esc)) = chars.next() {
                    name.push(esc);
                }
            }
            ':' => {
                if name.is_empty() {
                    return None;
                }
                let link = &after[i + 1..];
                if !(link.is_empty() || link.starts_with(' ') || link.starts_with('\n')) {
                    return None;
                }
                return Some(TargetMarker {
                    name,
                    anonymous: false,
                    link: link.trim().to_string(),
                });
            }
            _ => name.push(c),
        }
    }
    None
}

/// `name_` or `` `phrase`_ `` → normalized reference name (indirect
/// target). The check runs on the whitespace-joined link block; an escaped
/// trailing underscore (`uri\_`) is NOT a reference (fixture-verified).
fn reference_name_from_link(link: &str) -> Option<String> {
    let joined = ids::whitespace_normalize_name(link);
    let body = joined.strip_suffix('_')?;
    if body.ends_with('\\') {
        return None;
    }
    if let Some(phrase) = body.strip_prefix('`').and_then(|b| b.strip_suffix('`')) {
        if phrase.is_empty() {
            return None;
        }
        return Some(ids::fully_normalize_name(phrase));
    }
    if !body.is_empty()
        && !body.ends_with('_')
        && !body.contains(char::is_whitespace)
        && !body.contains('`')
        && !body.contains('\\')
    {
        return Some(ids::fully_normalize_name(body));
    }
    None
}

// ----------------------------------------------------------------------
// tests (plan tasks 7-12; expectations probe-verified against docutils
// 0.22.4 parse-layer output — see 2026-08-07-m2-wave1-probes.md)
// ----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rst::{parse_rst, ParseOptions};

    /// The `(source, lineno)` sequence of a line stream.
    fn stream_of(lines: &[LineRec]) -> Vec<(u16, u32)> {
        lines.iter().map(|l| (l.source, l.lineno)).collect()
    }

    #[test]
    fn splicing_a_pushed_source_inserts_its_recs_at_the_cursor() {
        let mut p = BlockParser::new("one\ntwo\nthree", "<doc>");
        let mut stream = std::mem::take(&mut p.top);
        assert_eq!(stream_of(&stream), vec![(0, 1), (0, 2), (0, 3)]);

        p.apply_splice(
            &mut stream,
            1,
            SpliceRequest {
                lines: vec!["alpha".to_string(), "beta".to_string()],
                source_path: "inc.rst".to_string(),
                base_lineno_override: None,
            },
        );

        assert_eq!(
            stream_of(&stream),
            vec![(0, 1), (1, 1), (1, 2), (0, 2), (0, 3)],
            "the pushed source's lines join the stream at the cursor"
        );
        assert_eq!(p.sources.len(), 2, "source_texts gained the new entry");
        assert_eq!(p.sources.path(1), "inc.rst");
        assert_eq!(p.sources.text(1), "alpha\nbeta");
    }

    #[test]
    fn a_directive_returned_splice_parses_at_the_cursor_with_its_own_provenance() {
        // The cfg(test) splice directive returns a SpliceRequest built from
        // its input alone; the block-parse loop must insert its lines right
        // after the directive and keep parsing.
        let src =
            "before\n\n.. sphinx-ultra-test-splice:: inc.rst\n\n   alpha\n\n   beta\n\nafter\n";
        let tree = parse_rst(
            src,
            &ParseOptions {
                source_path: "<doc>".into(),
                sphinx: false,
                docname: "index".into(),
                exclude_patterns: Vec::new(),
                py: Default::default(),
                found_docs: None,
            },
        );
        assert_eq!(
            tree.sources,
            vec!["<doc>".to_string(), "inc.rst".to_string()]
        );
        let paras: Vec<(String, u16, u32)> = tree
            .root
            .children
            .iter()
            .map(|n| (n.astext(), n.span.source, n.span.line))
            .collect();
        assert_eq!(
            paras,
            vec![
                ("before".to_string(), 0, 1),
                ("alpha".to_string(), 1, 1),
                ("beta".to_string(), 1, 3),
                ("after".to_string(), 0, 9),
            ],
            "spliced paragraphs carry the pushed source's id and 1-based lines"
        );
    }

    #[test]
    fn a_title_underline_warning_stamps_the_recs_table_path_and_lineno() {
        // "====" is >= 4 chars but shorter than the title: the section forms
        // with a "Title underline too short." WARNING whose source/line come
        // from the underline REC — its table path and lineno — not from a
        // recount of the document text.
        let tree = parse_rst(
            "badly\n====\n",
            &ParseOptions {
                source_path: "<doc>".into(),
                sphinx: false,
                docname: "index".into(),
                exclude_patterns: Vec::new(),
                py: Default::default(),
                found_docs: None,
            },
        );
        let section = &tree.root.children[0];
        let msg = section
            .children
            .iter()
            .find(|n| n.kind == kinds::SYSTEM_MESSAGE)
            .expect("short underline warns");
        assert_eq!(
            msg.get("source"),
            Some(&AttrValue::Str(tree.sources[0].clone())),
            "message source = the rec's table path"
        );
        assert_eq!(
            msg.get("line"),
            Some(&AttrValue::Int(2)),
            "message line = the underline rec's lineno"
        );
    }

    fn pf(src: &str) -> String {
        parse_rst(
            src,
            &ParseOptions {
                source_path: "<snippet>".into(),
                sphinx: false,
                docname: "index".into(),
                exclude_patterns: Vec::new(),
                py: Default::default(),
                found_docs: None,
            },
        )
        .root
        .pformat()
    }

    /// Same, with sphinx's directive set and node overrides enabled.
    fn pf_sphinx(src: &str) -> String {
        parse_rst(
            src,
            &ParseOptions {
                source_path: "<snippet>".into(),
                sphinx: true,
                docname: "index".into(),
                exclude_patterns: Vec::new(),
                py: Default::default(),
                found_docs: None,
            },
        )
        .root
        .pformat()
    }

    /// docutils registers a figure's `:name:` on the *image*
    /// (`Image.run` -> `add_name`); sphinx pops the option first and applies
    /// it to the figure instead (`directives/patches.py:33-56`). The
    /// difference is load-bearing: `numfig` keys figure numbers off
    /// `figure['ids'][0]`, and `:ref:`/`:numref:` resolve to that node.
    ///
    /// The sphinx half was re-verified against the 9.1.0 oracle in wave-4
    /// task 9 (`.. figure:: pic.png` + `:name: myfig` →
    /// `<figure ids="myfig" names="myfig">` over `<image ...>`), but the
    /// case cannot join `tests/fixtures/sphinx_doctree_differential.json`:
    /// a figure needs an `image`, and `ImageCollector.process_doc` stamps
    /// every image with `candidates="{'*': 'pic.png'}"`, one of that
    /// corpus's enumerated excluded divergences. This assertion is the
    /// standing pin until image collection lands.
    #[test]
    fn a_figure_name_lands_on_the_image_in_docutils_and_the_figure_in_sphinx() {
        let src = ".. figure:: pic.png\n   :name: fig one\n\n   Caption.\n";

        let docutils = pf(src);
        assert!(
            docutils.contains(r#"<image ids="fig-one" names="fig\ one" uri="pic.png">"#),
            "{docutils}"
        );
        assert!(docutils.contains("<figure>"), "{docutils}");

        let sphinx = pf_sphinx(src);
        assert!(
            sphinx.contains(r#"<figure ids="fig-one" names="fig\ one">"#),
            "{sphinx}"
        );
        assert!(sphinx.contains(r#"<image uri="pic.png">"#), "{sphinx}");
    }

    /// sphinx returns early — without re-applying the popped `:name:` —
    /// when the figure came back with an error node, so neither node ends
    /// up named.
    #[test]
    fn a_figure_whose_caption_is_malformed_keeps_no_name() {
        let sphinx = pf_sphinx(".. figure:: pic.png\n   :name: fig-bad\n\n   - not a caption\n");
        // (the raw source is echoed inside the error's literal_block, so
        // this checks the attributes, not the text)
        assert!(!sphinx.contains(r#"ids="fig-bad""#), "{sphinx}");
        assert!(sphinx.contains("<figure>\n"), "{sphinx}");
        assert!(sphinx.contains("<system_message"), "{sphinx}");
    }

    /// `EnvVarXRefRole.result_nodes` runs only when `is_ref`, which
    /// `XRefRole` clears for a `!`-prefixed role text. The index entries and
    /// the `index-N` target are the visible half; the invisible half is the
    /// serial, which is document-wide — burning one on a suppressed
    /// reference would renumber every later `index-N` id in the file.
    #[test]
    fn a_suppressed_envvar_reference_consumes_no_index_serial() {
        let live = pf_sphinx("See :envvar:`HOME_A` here.\n\n.. index:: Something\n");
        assert!(live.contains(r#"<target ids="index-0">"#), "{live}");
        assert!(
            live.contains(
                r#"<index entries="('single',\ 'Something',\ 'index-1',\ '',\ None)" inline="0">"#
            ),
            "a live :envvar: takes index-0, so the directive gets index-1:\n{live}"
        );

        let suppressed = pf_sphinx("See :envvar:`!HOME_A` here.\n\n.. index:: Something\n");
        assert!(
            !suppressed.contains("environment variable;"),
            "a suppressed reference emits no index entries:\n{suppressed}"
        );
        assert!(
            suppressed.contains(
                r#"<index entries="('single',\ 'Something',\ 'index-0',\ '',\ None)" inline="0">"#
            ),
            "and no serial, so the directive still gets index-0:\n{suppressed}"
        );
    }

    // ----- task 7: document + paragraphs -----

    #[test]
    fn empty_document() {
        assert_eq!(pf(""), "<document source=\"<snippet>\">\n");
        assert_eq!(pf("   \n\n  \n"), "<document source=\"<snippet>\">\n");
    }

    #[test]
    fn single_paragraph() {
        assert_eq!(
            pf("Just some text."),
            "<document source=\"<snippet>\">\n    <paragraph>\n        Just some text.\n"
        );
    }

    #[test]
    fn multiline_paragraph_keeps_internal_newlines() {
        assert_eq!(
            pf("line one\nline two"),
            "<document source=\"<snippet>\">\n    <paragraph>\n        line one\n        line two\n"
        );
    }

    #[test]
    fn blank_lines_separate_paragraphs() {
        assert_eq!(
            pf("para one\n\n\npara two"),
            "<document source=\"<snippet>\">\n    <paragraph>\n        para one\n    <paragraph>\n        para two\n"
        );
    }

    #[test]
    fn paragraph_spans_cover_source_bytes() {
        let src = "para one\n\npara two";
        let tree = parse_rst(
            src,
            &ParseOptions {
                source_path: "<snippet>".into(),
                sphinx: false,
                docname: "index".into(),
                exclude_patterns: Vec::new(),
                py: Default::default(),
                found_docs: None,
            },
        );
        let second = &tree.root.children[1];
        let text = &src[second.span.start as usize..second.span.end as usize];
        assert_eq!(text, "para two");
    }

    // ----- task 8: sections + transitions -----

    #[test]
    fn nested_sections_no_promotion() {
        assert_eq!(
            pf("Title\n=====\n\nPara under title.\n\nSub\n---\n\nPara under sub."),
            "<document source=\"<snippet>\">\n    <section ids=\"title\" names=\"title\">\n        <title>\n            Title\n        <paragraph>\n            Para under title.\n        <section ids=\"sub\" names=\"sub\">\n            <title>\n                Sub\n            <paragraph>\n                Para under sub.\n"
        );
    }

    #[test]
    fn overline_and_underline_is_a_distinct_style() {
        let out = pf("=====\nOver\n=====\n\nUnder\n=====");
        assert!(out.contains("    <section ids=\"over\" names=\"over\">\n"));
        assert!(out.contains("        <section ids=\"under\" names=\"under\">\n"));
    }

    #[test]
    fn underline_too_short_warns_but_sections() {
        assert_eq!(
            pf("Long Section Title\n======\n"),
            "<document source=\"<snippet>\">\n    <section ids=\"long-section-title\" names=\"long\\ section\\ title\">\n        <title>\n            Long Section Title\n        <system_message level=\"2\" line=\"2\" source=\"<snippet>\" type=\"WARNING\">\n            <paragraph>\n                Title underline too short.\n            <literal_block xml:space=\"preserve\">\n                Long Section Title\n                ======\n"
        );
    }

    #[test]
    fn short_underline_demotes_to_paragraph() {
        let out = pf("Title\n===");
        assert!(out.contains("<system_message level=\"1\" line=\"2\" source=\"<snippet>\" type=\"INFO\">\n        <paragraph>\n            Possible title underline, too short for the title.\n            Treating it as ordinary text because it's so short.\n"));
        assert!(out.contains("<paragraph>\n        Title\n        ===\n"));
        assert!(!out.contains("<section"));
    }

    #[test]
    fn inconsistent_style_skip_is_error_and_drops_section() {
        let out = pf("A\n-\n\nB\n=\n\nC\n-\n\nD\n~\n\nbody\n");
        assert!(out.contains("Inconsistent title style: skip from level 1 to 3.\n"));
        assert!(out.contains("Established title styles: - =\n"));
        assert!(!out.contains("names=\"d\""));
        // D's body attaches inside C, after the error message.
        assert!(out.contains("        <paragraph>\n            body\n"));
    }

    #[test]
    fn duplicate_titles_dupname_both_sections() {
        assert_eq!(
            pf("Duplicate\n=========\n\nx\n\nDuplicate\n=========\n\ny\n"),
            "<document source=\"<snippet>\">\n    <section dupnames=\"duplicate\" ids=\"duplicate\">\n        <title>\n            Duplicate\n        <paragraph>\n            x\n    <section dupnames=\"duplicate\" ids=\"id1\">\n        <title>\n            Duplicate\n        <system_message backrefs=\"id1\" level=\"1\" line=\"7\" source=\"<snippet>\" type=\"INFO\">\n            <paragraph>\n                Duplicate implicit target name: \"duplicate\".\n        <paragraph>\n            y\n"
        );
    }

    #[test]
    fn transitions_parse_clean_everywhere_at_parse_layer() {
        assert_eq!(
            pf("Para.\n\n----\n\nMore."),
            "<document source=\"<snippet>\">\n    <paragraph>\n        Para.\n    <transition>\n    <paragraph>\n        More.\n"
        );
        assert_eq!(
            pf("----\n\npara"),
            "<document source=\"<snippet>\">\n    <transition>\n    <paragraph>\n        para\n"
        );
        assert_eq!(
            pf("para\n\n----"),
            "<document source=\"<snippet>\">\n    <paragraph>\n        para\n    <transition>\n"
        );
        assert_eq!(
            pf("para\n\n----\n\n----\n\nend"),
            "<document source=\"<snippet>\">\n    <paragraph>\n        para\n    <transition>\n    <transition>\n    <paragraph>\n        end\n"
        );
        assert_eq!(
            pf("Head\n====\n\n----\n\npara"),
            "<document source=\"<snippet>\">\n    <section ids=\"head\" names=\"head\">\n        <title>\n            Head\n        <transition>\n        <paragraph>\n            para\n"
        );
        assert_eq!(
            pf("before\n\n---\n\nafter"),
            "<document source=\"<snippet>\">\n    <paragraph>\n        before\n    <paragraph>\n        ---\n    <paragraph>\n        after\n"
        );
    }

    #[test]
    fn single_line_plus_underline_is_title_even_unblanked() {
        assert_eq!(
            pf("para\n----\nafter\n"),
            "<document source=\"<snippet>\">\n    <section ids=\"para\" names=\"para\">\n        <title>\n            para\n        <paragraph>\n            after\n"
        );
    }

    #[test]
    fn multiline_paragraph_absorbs_adornment() {
        assert_eq!(
            pf("line1\nline2\n----\nafter\n"),
            "<document source=\"<snippet>\">\n    <paragraph>\n        line1\n        line2\n        ----\n        after\n"
        );
    }

    // ----- task 9: lists -----

    #[test]
    fn bullet_nesting_and_multi_paragraph_items() {
        assert_eq!(
            pf("- outer one\n\n  * inner a\n\n- first para of item\n\n  second para of item"),
            "<document source=\"<snippet>\">\n    <bullet_list bullet=\"-\">\n        <list_item>\n            <paragraph>\n                outer one\n            <bullet_list bullet=\"*\">\n                <list_item>\n                    <paragraph>\n                        inner a\n        <list_item>\n            <paragraph>\n                first para of item\n            <paragraph>\n                second para of item\n"
        );
    }

    #[test]
    fn tight_and_loose_lists_identical() {
        let tight = pf("- one\n- two");
        let loose = pf("- one\n\n- two");
        assert_eq!(tight, loose);
        assert!(tight.contains("<list_item>\n            <paragraph>\n                one\n"));
    }

    #[test]
    fn enumerated_formats() {
        assert!(pf("a. x\nb. y")
            .contains("<enumerated_list enumtype=\"loweralpha\" prefix=\"\" suffix=\".\">\n"));
        assert!(pf("(1) x\n(2) y")
            .contains("<enumerated_list enumtype=\"arabic\" prefix=\"(\" suffix=\")\">\n"));
        assert!(pf("A) x\nB) y")
            .contains("<enumerated_list enumtype=\"upperalpha\" prefix=\"\" suffix=\")\">\n"));
        assert!(pf("#. x\n#. y")
            .contains("<enumerated_list enumtype=\"arabic\" prefix=\"\" suffix=\".\">\n"));
    }

    #[test]
    fn enumerated_start_and_info_message() {
        let out = pf("3. three\n4. four");
        assert!(out.contains(
            "<enumerated_list enumtype=\"arabic\" prefix=\"\" start=\"3\" suffix=\".\">\n"
        ));
        assert!(out.contains("    <system_message level=\"1\" line=\"1\" source=\"<snippet>\" type=\"INFO\">\n        <paragraph>\n            Enumerated list start value not ordinal-1: \"3\" (ordinal 3)\n"));
    }

    #[test]
    fn non_consecutive_without_blank_aborts_to_paragraph() {
        assert_eq!(
            pf("1. one\n3. three"),
            "<document source=\"<snippet>\">\n    <paragraph>\n        1. one\n        3. three\n"
        );
    }

    #[test]
    fn broken_sequence_mid_list_ends_it_with_warning() {
        assert_eq!(
            pf("1. one\n2. two\n5. five\n"),
            "<document source=\"<snippet>\">\n    <enumerated_list enumtype=\"arabic\" prefix=\"\" suffix=\".\">\n        <list_item>\n            <paragraph>\n                one\n    <system_message level=\"2\" line=\"2\" source=\"<snippet>\" type=\"WARNING\">\n        <paragraph>\n            Enumerated list ends without a blank line; unexpected unindent.\n    <paragraph>\n        2. two\n        5. five\n"
        );
    }

    #[test]
    fn single_letter_ambiguity() {
        assert!(pf("A. Einstein was smart.").contains("enumtype=\"upperalpha\""));
        assert!(pf("i. single").contains("enumtype=\"lowerroman\""));
        let v = pf("v. five");
        assert!(v.contains("enumtype=\"loweralpha\"") && v.contains("start=\"22\""));
        let c = pf("c. see");
        assert!(c.contains("enumtype=\"loweralpha\"") && c.contains("start=\"3\""));
        let ii = pf("ii. two\niii. three");
        assert!(ii.contains("enumtype=\"lowerroman\"") && ii.contains("start=\"2\""));
    }

    #[test]
    fn bullet_list_end_without_blank_warns() {
        assert_eq!(
            pf("- item\nplain\n"),
            "<document source=\"<snippet>\">\n    <bullet_list bullet=\"-\">\n        <list_item>\n            <paragraph>\n                item\n    <system_message level=\"2\" line=\"2\" source=\"<snippet>\" type=\"WARNING\">\n        <paragraph>\n            Bullet list ends without a blank line; unexpected unindent.\n    <paragraph>\n        plain\n"
        );
    }

    #[test]
    fn bullet_marker_alone_takes_indented_body() {
        assert_eq!(
            pf("-\n  body from next line\n"),
            "<document source=\"<snippet>\">\n    <bullet_list bullet=\"-\">\n        <list_item>\n            <paragraph>\n                body from next line\n"
        );
    }

    // ----- task 10: definition lists + block quotes -----

    #[test]
    fn definition_list_with_classifiers() {
        assert_eq!(
            pf("term2 : classifier one : classifier two\n    Definition2."),
            "<document source=\"<snippet>\">\n    <definition_list>\n        <definition_list_item>\n            <term>\n                term2\n            <classifier>\n                classifier one\n            <classifier>\n                classifier two\n            <definition>\n                <paragraph>\n                    Definition2.\n"
        );
    }

    #[test]
    fn no_space_colon_stays_in_term() {
        let out = pf("term:not a classifier\n    Definition.");
        assert!(out.contains("<term>\n                term:not a classifier\n"));
        assert!(!out.contains("<classifier>"));
    }

    #[test]
    fn consecutive_items_merge() {
        let out = pf("term1\n    Def1.\n\nterm2\n    Def2.");
        assert_eq!(out.matches("<definition_list>\n").count(), 1);
        assert_eq!(out.matches("<definition_list_item>\n").count(), 2);
    }

    #[test]
    fn definition_list_end_without_blank_warns() {
        let out = pf("term\n    def\nplain\n");
        assert!(out.contains("Definition list ends without a blank line; unexpected unindent.\n"));
        assert!(out.contains("<system_message level=\"2\" line=\"3\""));
    }

    #[test]
    fn block_quote_with_attribution() {
        assert_eq!(
            pf("Para.\n\n    No matter where you go, there you are.\n\n    -- Buckaroo Banzai"),
            "<document source=\"<snippet>\">\n    <paragraph>\n        Para.\n    <block_quote>\n        <paragraph>\n            No matter where you go, there you are.\n        <attribution>\n            Buckaroo Banzai\n"
        );
    }

    #[test]
    fn attribution_splits_sibling_quotes() {
        let out = pf("Para.\n\n    First quote.\n\n    -- First Author\n\n    Second quote.\n\n    -- Second Author");
        assert_eq!(out.matches("<block_quote>\n").count(), 2);
        assert!(out.contains("First Author") && out.contains("Second Author"));
    }

    #[test]
    fn multiline_attribution_joins_with_newline() {
        let out = pf("Para.\n\n    Quote.\n\n    -- Author Name,\n       Book Title, 1999\n");
        assert!(
            out.contains("<attribution>\n            Author Name,\n            Book Title, 1999\n")
        );
    }

    #[test]
    fn unexpected_indentation_after_multiline_paragraph() {
        assert_eq!(
            pf("line one\nline two\n    Indented without blank line.\n"),
            "<document source=\"<snippet>\">\n    <paragraph>\n        line one\n        line two\n    <system_message level=\"3\" line=\"3\" source=\"<snippet>\" type=\"ERROR\">\n        <paragraph>\n            Unexpected indentation.\n    <block_quote>\n        <paragraph>\n            Indented without blank line.\n"
        );
    }

    #[test]
    fn partial_dedent_nests_inside_quote_with_warning() {
        assert_eq!(
            pf("Para.\n\n    quoted\n  dedented-oddly\n"),
            "<document source=\"<snippet>\">\n    <paragraph>\n        Para.\n    <block_quote>\n        <block_quote>\n            <paragraph>\n                quoted\n        <system_message level=\"2\" line=\"4\" source=\"<snippet>\" type=\"WARNING\">\n            <paragraph>\n                Block quote ends without a blank line; unexpected unindent.\n        <paragraph>\n            dedented-oddly\n"
        );
    }

    // ----- task 11: literal, doctest, line blocks -----

    #[test]
    fn literal_block_expanded_colon() {
        assert_eq!(
            pf("Paragraph introducing::\n\n    literal line one\n    literal line two"),
            "<document source=\"<snippet>\">\n    <paragraph>\n        Paragraph introducing:\n    <literal_block xml:space=\"preserve\">\n        literal line one\n        literal line two\n"
        );
    }

    #[test]
    fn colon_math_variants() {
        assert!(pf("Paragraph ends with ::\n\n    literal here")
            .contains("<paragraph>\n        Paragraph ends with\n"));
        assert!(pf("text:::\n\n    x").contains("<paragraph>\n        text::\n"));
        assert_eq!(
            pf("::\n\n    literal"),
            "<document source=\"<snippet>\">\n    <literal_block xml:space=\"preserve\">\n        literal\n"
        );
    }

    #[test]
    fn quoted_literal_block_keeps_quotes() {
        assert_eq!(
            pf("Next is a quoted literal::\n\n> quoted line one\n> quoted line two"),
            "<document source=\"<snippet>\">\n    <paragraph>\n        Next is a quoted literal:\n    <literal_block xml:space=\"preserve\">\n        > quoted line one\n        > quoted line two\n"
        );
    }

    #[test]
    fn inconsistent_quoted_literal_errors() {
        assert_eq!(
            pf("intro::\n\n> line one\n$ different\n"),
            "<document source=\"<snippet>\">\n    <paragraph>\n        intro:\n    <literal_block xml:space=\"preserve\">\n        > line one\n    <system_message level=\"3\" line=\"4\" source=\"<snippet>\" type=\"ERROR\">\n        <paragraph>\n            Inconsistent literal block quoting.\n    <paragraph>\n        $ different\n"
        );
    }

    #[test]
    fn missing_literal_block_warns() {
        assert_eq!(
            pf("Intro::\n\nNot indented.\n"),
            "<document source=\"<snippet>\">\n    <paragraph>\n        Intro:\n    <system_message level=\"2\" line=\"3\" source=\"<snippet>\" type=\"WARNING\">\n        <paragraph>\n            Literal block expected; none found.\n    <paragraph>\n        Not indented.\n"
        );
    }

    #[test]
    fn literal_block_end_without_blank_warns() {
        let out = pf("para::\n\n    lit\nback\n");
        assert!(out.contains("Literal block ends without a blank line; unexpected unindent.\n"));
        assert!(out.contains("<system_message level=\"2\" line=\"4\""));
    }

    #[test]
    fn doctest_block() {
        assert_eq!(
            pf(">>> print(\"hello\")\nhello\n>>> 1 + 1\n2"),
            "<document source=\"<snippet>\">\n    <doctest_block xml:space=\"preserve\">\n        >>> print(\"hello\")\n        hello\n        >>> 1 + 1\n        2\n"
        );
    }

    #[test]
    fn line_block_nesting_and_empty_line() {
        assert_eq!(
            pf("| top one\n| top two\n|     nested one\n| back\n|\n| after empty"),
            "<document source=\"<snippet>\">\n    <line_block>\n        <line>\n            top one\n        <line>\n            top two\n        <line_block>\n            <line>\n                nested one\n        <line>\n            back\n        <line>\n        <line>\n            after empty\n"
        );
    }

    #[test]
    fn line_block_continuation_joins_line() {
        assert_eq!(
            pf("| A very long line\n  continued here\n| second\n"),
            "<document source=\"<snippet>\">\n    <line_block>\n        <line>\n            A very long line\n            continued here\n        <line>\n            second\n"
        );
    }

    // ----- task 12: comments + targets -----

    #[test]
    fn comment_forms() {
        assert_eq!(
            pf(".. This is a comment\n   that continues on\n   multiple lines."),
            "<document source=\"<snippet>\">\n    <comment xml:space=\"preserve\">\n        This is a comment\n        that continues on\n        multiple lines.\n"
        );
        // Probe-verified: `..` + blank + indented block leaves an EMPTY
        // comment; the block becomes an ordinary block quote.
        assert_eq!(
            pf("..\n\n   Indented block attached\n   to an empty comment start."),
            "<document source=\"<snippet>\">\n    <comment xml:space=\"preserve\">\n    <block_quote>\n        <paragraph>\n            Indented block attached\n            to an empty comment start.\n"
        );
        // Adjacent block IS the body.
        assert_eq!(
            pf("..\n   block line one\n   block line two"),
            "<document source=\"<snippet>\">\n    <comment xml:space=\"preserve\">\n        block line one\n        block line two\n"
        );
        assert_eq!(
            pf(".."),
            "<document source=\"<snippet>\">\n    <comment xml:space=\"preserve\">\n"
        );
    }

    #[test]
    fn comment_ragged_continuation_dedents_by_min() {
        assert_eq!(
            pf(".. first\n      deep\n   shallow\n"),
            "<document source=\"<snippet>\">\n    <comment xml:space=\"preserve\">\n        first\n           deep\n        shallow\n"
        );
    }

    #[test]
    fn comment_vs_target_dispatch() {
        let out = pf(".. _target: http://example.com\n\n.. just a comment::  with weird colons");
        assert!(out
            .contains("<target ids=\"target\" names=\"target\" refuri=\"http://example.com\">\n"));
        assert!(out.contains(
            "<comment xml:space=\"preserve\">\n        just a comment::  with weird colons\n"
        ));
    }

    #[test]
    fn target_forms_keep_ids_and_names_at_parse_layer() {
        let out = pf(".. _para-target:\n\nSome paragraph here.");
        assert!(out.contains("<target ids=\"para-target\" names=\"para-target\">\n    <paragraph>\n        Some paragraph here.\n"));

        let out = pf(".. _docutils: https://docutils.sourceforge.io/\n.. _indirect: docutils_");
        assert!(out.contains(
            "<target ids=\"docutils\" names=\"docutils\" refuri=\"https://docutils.sourceforge.io/\">\n"
        ));
        assert!(out.contains("<target ids=\"indirect\" names=\"indirect\" refname=\"docutils\">\n"));
    }

    #[test]
    fn multiline_refuri_concatenates() {
        assert_eq!(
            pf(".. _long: https://example.com/\n   path/here\n"),
            "<document source=\"<snippet>\">\n    <target ids=\"long\" names=\"long\" refuri=\"https://example.com/path/here\">\n"
        );
    }

    #[test]
    fn uri_with_spaces_strips_whitespace() {
        assert_eq!(
            pf(".. _a: B  Target_\n"),
            "<document source=\"<snippet>\">\n    <target ids=\"a\" names=\"a\" refuri=\"BTarget_\">\n"
        );
    }

    #[test]
    fn backtick_and_escaped_names() {
        assert_eq!(
            pf(".. _`name with: colon`: https://x/\n"),
            "<document source=\"<snippet>\">\n    <target ids=\"name-with-colon\" names=\"name\\ with:\\ colon\" refuri=\"https://x/\">\n"
        );
        assert_eq!(
            pf(".. _a\\: b: https://y/\n"),
            "<document source=\"<snippet>\">\n    <target ids=\"a-b\" names=\"a:\\ b\" refuri=\"https://y/\">\n"
        );
    }

    #[test]
    fn anonymous_targets_both_spellings() {
        assert_eq!(
            pf(".. __: https://example.com/1\n\n__ https://example.com/2"),
            "<document source=\"<snippet>\">\n    <target anonymous=\"1\" ids=\"id1\" refuri=\"https://example.com/1\">\n    <target anonymous=\"1\" ids=\"id2\" refuri=\"https://example.com/2\">\n"
        );
    }

    #[test]
    fn chained_targets_each_keep_own_ids() {
        let out = pf(".. _target1:\n.. _target2:\n\nSection Title\n=============");
        assert!(out.contains("<target ids=\"target1\" names=\"target1\">\n"));
        assert!(out.contains("<target ids=\"target2\" names=\"target2\">\n"));
        assert!(out.contains("<section ids=\"section-title\" names=\"section\\ title\">\n"));
    }

    #[test]
    fn duplicate_explicit_targets_warn_between() {
        assert_eq!(
            pf(".. _dup: https://1/\n\n.. _dup: https://2/\n"),
            "<document source=\"<snippet>\">\n    <target dupnames=\"dup\" ids=\"dup\" refuri=\"https://1/\">\n    <system_message level=\"2\" line=\"3\" source=\"<snippet>\" type=\"WARNING\">\n        <paragraph>\n            Duplicate explicit target name: \"dup\".\n    <target dupnames=\"dup\" ids=\"id1\" refuri=\"https://2/\">\n"
        );
    }

    // ----- nested-context errors -----

    #[test]
    fn nested_transition_and_title_are_errors() {
        assert_eq!(
            pf("Para.\n\n    ----\n\n    quoted\n"),
            "<document source=\"<snippet>\">\n    <paragraph>\n        Para.\n    <block_quote>\n        <system_message level=\"3\" line=\"3\" source=\"<snippet>\" type=\"ERROR\">\n            <paragraph>\n                Unexpected section title or transition.\n            <literal_block xml:space=\"preserve\">\n                ----\n        <paragraph>\n            quoted\n"
        );
        assert_eq!(
            pf("Para.\n\n    Fake\n    ====\n"),
            "<document source=\"<snippet>\">\n    <paragraph>\n        Para.\n    <block_quote>\n        <system_message level=\"3\" line=\"4\" source=\"<snippet>\" type=\"ERROR\">\n            <paragraph>\n                Unexpected section title.\n            <literal_block xml:space=\"preserve\">\n                Fake\n                ====\n"
        );
    }
}

/// The `py:*` directive family (M2 wave 4.5 task 6). Every expected
/// pformat below is pasted verbatim from the Sphinx 9.1.0 oracle probes —
/// the research spec [PY §1.6/1.7] and this task's probe_t6 run (harness3
/// conventions, pinned wheels) — never written from memory.
#[cfg(test)]
mod py_desc_tests {
    use super::*;
    use crate::py::PySigConfig;
    use crate::rst::{parse_rst_full, ParseOptions, ParseOutput};

    fn py_opts(py: PySigConfig) -> ParseOptions {
        ParseOptions {
            source_path: "<snippet>".into(),
            sphinx: true,
            docname: "index".into(),
            exclude_patterns: Vec::new(),
            py,
            found_docs: None,
        }
    }

    fn parse_py(src: &str) -> ParseOutput {
        parse_rst_full(src, &py_opts(PySigConfig::default()))
    }

    fn pf_py(src: &str) -> String {
        parse_py(src).doctree.root.pformat()
    }

    fn pf_py_cfg(src: &str, py: PySigConfig) -> String {
        parse_rst_full(src, &py_opts(py)).doctree.root.pformat()
    }

    /// `(fullname, objtype, node_id, aliased)` of every py object record.
    fn objects(out: &ParseOutput) -> Vec<(String, String, String, bool)> {
        out.registry
            .py_objects
            .iter()
            .map(|r| {
                (
                    r.fullname.clone(),
                    r.objtype.clone(),
                    r.node_id.clone(),
                    r.aliased,
                )
            })
            .collect()
    }

    fn owned(v: &[(&str, &str, &str, bool)]) -> Vec<(String, String, String, bool)> {
        v.iter()
            .map(|(a, b, c, d)| (a.to_string(), b.to_string(), c.to_string(), *d))
            .collect()
    }

    // ---- py_sig_re (checklist row 1) ----------------------------------

    #[test]
    fn py_sig_match_groups_and_spans() {
        let m = py_sig_match("mymod.func(a, b) -> str").unwrap();
        assert_eq!(m.prefix.as_deref(), Some("mymod."));
        assert_eq!(m.name, "func");
        assert_eq!(m.tp_list, None);
        assert_eq!(m.tp_span, (0, 0));
        assert_eq!(m.arglist.as_deref(), Some("a, b"));
        assert_eq!(m.arg_span, (11, 15));
        assert_eq!(m.retann.as_deref(), Some("str"));

        let m = py_sig_match("f[T](x)").unwrap();
        assert_eq!(m.tp_list.as_deref(), Some("T"));
        assert_eq!(m.tp_span, (2, 3));
        assert_eq!(m.arglist.as_deref(), Some("x"));

        // Empty written parens: group 4 participates with '' — falsy, so
        // handle_signature routes it to the needs_arglist branch.
        let m = py_sig_match("f()").unwrap();
        assert_eq!(m.arglist.as_deref(), Some(""));

        // The greedy-arglist edge: the LAST ')' closes the group, so a
        // parenthesized return annotation is swallowed INTO the arglist
        // and group 5 never participates (probe retann_tuple_greedy).
        let m = py_sig_match("f(x) -> (int, str)").unwrap();
        assert_eq!(m.arglist.as_deref(), Some("x) -> (int, str"));
        assert_eq!(m.retann, None);

        assert!(py_sig_match("not a signature!").is_none());
        assert!(py_sig_match("f(x").is_none());
    }

    // ---- baseline shapes (rows 1, 3, 4) --------------------------------

    #[test]
    fn function_plain_args_matches_the_sphinx_probe() {
        let out = parse_py(".. py:function:: func(a, b)\n\n   Body.\n");
        assert_eq!(
            out.doctree.root.pformat(),
            concat!(
                "<document source=\"<snippet>\">\n",
                "    <index entries=\"('pair',\\ 'built-in\\ function;\\ func()',\\ 'func',\\ '',\\ None)\">\n",
                "    <desc classes=\"py function\" desctype=\"function\" domain=\"py\" no-contents-entry=\"0\" no-index=\"0\" no-index-entry=\"0\" no-typesetting=\"0\" nocontentsentry=\"0\" noindex=\"0\" noindexentry=\"0\" objtype=\"function\">\n",
                "        <desc_signature _toc_name=\"func()\" _toc_parts=\"('func',)\" class=\"\" classes=\"sig sig-object\" fullname=\"func\" ids=\"func\" module=\"True\">\n",
                "            <desc_name classes=\"sig-name descname\" xml:space=\"preserve\">\n",
                "                func\n",
                "            <desc_parameterlist multi_line_parameter_list=\"0\" multi_line_trailing_comma=\"1\" xml:space=\"preserve\">\n",
                "                <desc_parameter xml:space=\"preserve\">\n",
                "                    <desc_sig_name classes=\"n\">\n",
                "                        a\n",
                "                <desc_parameter xml:space=\"preserve\">\n",
                "                    <desc_sig_name classes=\"n\">\n",
                "                        b\n",
                "        <desc_content>\n",
                "            <paragraph>\n",
                "                Body.\n",
            )
        );
        assert_eq!(objects(&out), owned(&[("func", "function", "func", false)]));
        assert!(out.registry.log_warnings.is_empty());
    }

    #[test]
    fn function_full_markers_matches_the_sphinx_probe() {
        assert_eq!(
            pf_py(".. py:function:: mymod.func(a, b=1, *args, c: int = 2, **kwargs) -> str\n"),
            concat!(
                "<document source=\"<snippet>\">\n",
                "    <index entries=\"('pair',\\ 'built-in\\ function;\\ mymod.func()',\\ 'mymod.func',\\ '',\\ None)\">\n",
                "    <desc classes=\"py function\" desctype=\"function\" domain=\"py\" no-contents-entry=\"0\" no-index=\"0\" no-index-entry=\"0\" no-typesetting=\"0\" nocontentsentry=\"0\" noindex=\"0\" noindexentry=\"0\" objtype=\"function\">\n",
                // A dotted prefix at top level is a CLASS prefix (trap 12):
                // class="mymod", desc_addname, index still "built-in".
                "        <desc_signature _toc_name=\"mymod.func()\" _toc_parts=\"('mymod', 'func')\" class=\"mymod\" classes=\"sig sig-object\" fullname=\"mymod.func\" ids=\"mymod.func\" module=\"True\">\n",
                "            <desc_addname classes=\"sig-prename descclassname\" xml:space=\"preserve\">\n",
                "                mymod.\n",
                "            <desc_name classes=\"sig-name descname\" xml:space=\"preserve\">\n",
                "                func\n",
                "            <desc_parameterlist multi_line_parameter_list=\"0\" multi_line_trailing_comma=\"1\" xml:space=\"preserve\">\n",
                "                <desc_parameter xml:space=\"preserve\">\n",
                "                    <desc_sig_name classes=\"n\">\n",
                "                        a\n",
                "                <desc_parameter xml:space=\"preserve\">\n",
                "                    <desc_sig_name classes=\"n\">\n",
                "                        b\n",
                "                    <desc_sig_operator classes=\"o\">\n",
                "                        =\n",
                "                    <inline classes=\"default_value\" support_smartquotes=\"0\">\n",
                "                        1\n",
                "                <desc_parameter xml:space=\"preserve\">\n",
                "                    <desc_sig_operator classes=\"o\">\n",
                "                        *\n",
                "                    <desc_sig_name classes=\"n\">\n",
                "                        args\n",
                "                <desc_parameter xml:space=\"preserve\">\n",
                "                    <desc_sig_name classes=\"n\">\n",
                "                        c\n",
                "                    <desc_sig_punctuation classes=\"p\">\n",
                "                        :\n",
                "                    <desc_sig_space classes=\"w\">\n",
                "                         \n",
                "                    <desc_sig_name classes=\"n\">\n",
                "                        <pending_xref py:class=\"True\" py:module=\"True\" refdomain=\"py\" refspecific=\"0\" reftarget=\"int\" reftype=\"class\">\n",
                "                            int\n",
                "                    <desc_sig_space classes=\"w\">\n",
                "                         \n",
                "                    <desc_sig_operator classes=\"o\">\n",
                "                        =\n",
                "                    <desc_sig_space classes=\"w\">\n",
                "                         \n",
                "                    <inline classes=\"default_value\" support_smartquotes=\"0\">\n",
                "                        2\n",
                "                <desc_parameter xml:space=\"preserve\">\n",
                "                    <desc_sig_operator classes=\"o\">\n",
                "                        **\n",
                "                    <desc_sig_name classes=\"n\">\n",
                "                        kwargs\n",
                "            <desc_returns xml:space=\"preserve\">\n",
                "                <pending_xref py:class=\"True\" py:module=\"True\" refdomain=\"py\" refspecific=\"0\" reftarget=\"str\" reftype=\"class\">\n",
                "                    str\n",
                "        <desc_content>\n",
            )
        );
    }

    /// Row 4/trap 1: no written arglist AND empty written `()` both take
    /// the bare attr-less paramlist for needs_arglist kinds; a class
    /// (needs_arglist false) with `()` gets NO paramlist at all.
    #[test]
    fn no_arglist_and_empty_parens_take_the_bare_paramlist() {
        let out = pf_py(".. py:function:: func\n");
        assert!(
            out.contains("            <desc_parameterlist xml:space=\"preserve\">\n"),
            "bare attr-less paramlist: {out}"
        );
        // Empty parens with a return annotation (probe empty_parens_retann):
        // still the bare list, followed by desc_returns.
        let out = pf_py(".. py:function:: f() -> int\n");
        assert!(out.contains(concat!(
            "            <desc_parameterlist xml:space=\"preserve\">\n",
            "            <desc_returns xml:space=\"preserve\">\n",
            "                <pending_xref py:class=\"True\" py:module=\"True\" refdomain=\"py\" refspecific=\"0\" reftarget=\"int\" reftype=\"class\">\n",
            "                    int\n",
        )));
        // Probe class_no_parens: PyClasslike never needs an arglist.
        let out = pf_py(".. py:class:: C()\n");
        assert!(!out.contains("desc_parameterlist"), "{out}");
    }

    /// Row 1/trap 7: a failed py_sig_re match is SILENT — raw sig in one
    /// desc_name, empty toc attrs, no ids, no registration — and drops
    /// option tails with the cleared signode (probe annotation_bad_sig).
    #[test]
    fn a_bad_signature_is_silent_with_empty_toc_and_no_registration() {
        let out = parse_py(".. py:function:: not a signature!\n   :annotation: tail\n");
        assert_eq!(
            out.doctree.root.pformat(),
            concat!(
                "<document source=\"<snippet>\">\n",
                "    <index entries=\"\">\n",
                "    <desc classes=\"py function\" desctype=\"function\" domain=\"py\" no-contents-entry=\"0\" no-index=\"0\" no-index-entry=\"0\" no-typesetting=\"0\" nocontentsentry=\"0\" noindex=\"0\" noindexentry=\"0\" objtype=\"function\">\n",
                "        <desc_signature _toc_name=\"\" _toc_parts=\"()\" classes=\"sig sig-object\">\n",
                "            <desc_name classes=\"sig-name descname\" xml:space=\"preserve\">\n",
                "                not a signature!\n",
                "        <desc_content>\n",
            )
        );
        assert!(objects(&out).is_empty());
        assert!(out.registry.log_warnings.is_empty(), "no warning (trap 7)");
    }

    /// Row 4: `:async:` prefix annotation and the `:annotation:` tail
    /// (probes function_async / function_annotation_option).
    #[test]
    fn async_prefix_and_annotation_option_tail() {
        let out = pf_py(".. py:function:: coro(x)\n   :async:\n");
        assert!(out.contains(concat!(
            "            <desc_annotation xml:space=\"preserve\">\n",
            "                <desc_sig_keyword classes=\"k\">\n",
            "                    async\n",
            "                <desc_sig_space classes=\"w\">\n",
            "                     \n",
            "            <desc_name classes=\"sig-name descname\" xml:space=\"preserve\">\n",
            "                coro\n",
        )));
        let out = pf_py(".. py:function:: f(x)\n   :annotation: something extra\n");
        assert!(out.ends_with(concat!(
            "            <desc_annotation xml:space=\"preserve\">\n",
            "                <desc_sig_space classes=\"w\">\n",
            "                     \n",
            "                something extra\n",
            "        <desc_content>\n",
        )));
    }

    // ---- module resolution (rows 2, 9, 10) -----------------------------

    /// Row 3/10: the `:module:` option qualifies ids/index/registration and
    /// pushes/pops the module scope around the content — the NEXT directive
    /// is back under the surrounding module (probe method_module_pop).
    #[test]
    fn the_module_option_qualifies_and_pops() {
        let out = parse_py(
            ".. py:module:: outer\n\n.. py:function:: g(x)\n   :module: inner\n\n.. py:function:: h(x)\n",
        );
        let pf = out.doctree.root.pformat();
        assert!(pf.contains(
            "    <index entries=\"('single',\\ 'g()\\ (in\\ module\\ inner)',\\ 'inner.g',\\ '',\\ None)\">\n"
        ));
        assert!(pf.contains(
            "        <desc_signature _toc_name=\"g()\" _toc_parts=\"('inner', 'g')\" class=\"\" classes=\"sig sig-object\" fullname=\"g\" ids=\"inner.g\" module=\"inner\">\n"
        ));
        assert!(pf.contains(
            "    <index entries=\"('single',\\ 'h()\\ (in\\ module\\ outer)',\\ 'outer.h',\\ '',\\ None)\">\n"
        ));
        assert!(pf.contains(
            "        <desc_signature _toc_name=\"h()\" _toc_parts=\"('outer', 'h')\" class=\"\" classes=\"sig sig-object\" fullname=\"h\" ids=\"outer.h\" module=\"outer\">\n"
        ));
        assert_eq!(
            objects(&out),
            owned(&[
                ("outer", "module", "module-outer", false),
                ("inner.g", "function", "inner.g", false),
                ("outer.h", "function", "outer.h", false),
            ])
        );
    }

    /// Row 2: prefix resolution inside a class — the class's own prefix is
    /// stripped from display; a DIFFERENT prefix nests (fullname
    /// `C.D.meth`, desc_addname `D.`, index `meth() (C.D method)`) —
    /// probes method_class_prefix_given / method_other_prefix.
    #[test]
    fn class_prefixes_strip_or_nest() {
        let out = parse_py(
            ".. py:class:: C\n\n   .. py:method:: C.meth(x)\n\n   .. py:method:: D.meth(x)\n",
        );
        let pf = out.doctree.root.pformat();
        assert!(pf.contains(
            "                <desc_signature _toc_name=\"C.meth()\" _toc_parts=\"('C', 'meth')\" class=\"C\" classes=\"sig sig-object\" fullname=\"C.meth\" ids=\"C.meth\" module=\"True\">\n"
        ));
        // The stripped prefix leaves no desc_addname on C.meth.
        let c_meth_sig = pf
            .split("fullname=\"C.meth\"")
            .nth(1)
            .unwrap()
            .split("desc_signature")
            .next()
            .unwrap();
        assert!(!c_meth_sig.contains("desc_addname"), "{c_meth_sig}");
        assert!(pf.contains(
            "                <desc_signature _toc_name=\"C.D.meth()\" _toc_parts=\"('C', 'D', 'meth')\" class=\"C\" classes=\"sig sig-object\" fullname=\"C.D.meth\" ids=\"C.D.meth\" module=\"True\">\n"
        ));
        assert!(pf.contains(concat!(
            "                    <desc_addname classes=\"sig-prename descclassname\" xml:space=\"preserve\">\n",
            "                        D.\n",
        )));
        assert!(pf.contains(
            "            <index entries=\"('single',\\ 'meth()\\ (C.D\\ method)',\\ 'C.D.meth',\\ '',\\ None)\">\n"
        ));
    }

    // ---- multi-signature (row 11) --------------------------------------

    #[test]
    fn multiple_signatures_share_one_desc_and_register_each_unique_name() {
        let out = parse_py(".. py:function:: f(x)\n                  g(y)\n\n   Shared body.\n");
        let pf = out.doctree.root.pformat();
        assert!(pf.contains(
            "    <index entries=\"('pair',\\ 'built-in\\ function;\\ f()',\\ 'f',\\ '',\\ None) ('pair',\\ 'built-in\\ function;\\ g()',\\ 'g',\\ '',\\ None)\">\n"
        ));
        assert!(pf.contains(
            "        <desc_signature _toc_name=\"f()\" _toc_parts=\"('f',)\" class=\"\" classes=\"sig sig-object\" fullname=\"f\" ids=\"f\" module=\"True\">\n"
        ));
        assert!(pf.contains(
            "        <desc_signature _toc_name=\"g()\" _toc_parts=\"('g',)\" class=\"\" classes=\"sig sig-object\" fullname=\"g\" ids=\"g\" module=\"True\">\n"
        ));
        assert_eq!(pf.matches("<desc_content>").count(), 1, "one shared body");
        assert_eq!(
            objects(&out),
            owned(&[("f", "function", "f", false), ("g", "function", "g", false),]),
            "every UNIQUE name registers ([PY §1.6 function_multi_sig])"
        );
        // Identical signatures dedupe via `if name not in self.names`.
        let out = parse_py(".. py:function:: f(x)\n                  f(x)\n");
        assert_eq!(objects(&out), owned(&[("f", "function", "f", false)]));
    }

    // ---- class + nesting (rows 2, 10) ----------------------------------

    #[test]
    fn class_with_bases_nests_the_method_scope() {
        let out = parse_py(
            ".. py:class:: MyClass(Base1, Base2)\n\n   .. py:method:: meth(self, arg)\n\n      Body.\n",
        );
        assert_eq!(
            out.doctree.root.pformat(),
            concat!(
                "<document source=\"<snippet>\">\n",
                "    <index entries=\"('single',\\ 'MyClass\\ (built-in\\ class)',\\ 'MyClass',\\ '',\\ None)\">\n",
                "    <desc classes=\"py class\" desctype=\"class\" domain=\"py\" no-contents-entry=\"0\" no-index=\"0\" no-index-entry=\"0\" no-typesetting=\"0\" nocontentsentry=\"0\" noindex=\"0\" noindexentry=\"0\" objtype=\"class\">\n",
                "        <desc_signature _toc_name=\"MyClass\" _toc_parts=\"('MyClass',)\" class=\"\" classes=\"sig sig-object\" fullname=\"MyClass\" ids=\"MyClass\" module=\"True\">\n",
                "            <desc_annotation xml:space=\"preserve\">\n",
                "                <desc_sig_keyword classes=\"k\">\n",
                "                    class\n",
                "                <desc_sig_space classes=\"w\">\n",
                "                     \n",
                "            <desc_name classes=\"sig-name descname\" xml:space=\"preserve\">\n",
                "                MyClass\n",
                // Class bases parse as an ordinary arglist — names only.
                "            <desc_parameterlist multi_line_parameter_list=\"0\" multi_line_trailing_comma=\"1\" xml:space=\"preserve\">\n",
                "                <desc_parameter xml:space=\"preserve\">\n",
                "                    <desc_sig_name classes=\"n\">\n",
                "                        Base1\n",
                "                <desc_parameter xml:space=\"preserve\">\n",
                "                    <desc_sig_name classes=\"n\">\n",
                "                        Base2\n",
                "        <desc_content>\n",
                "            <index entries=\"('single',\\ 'meth()\\ (MyClass\\ method)',\\ 'MyClass.meth',\\ '',\\ None)\">\n",
                "            <desc classes=\"py method\" desctype=\"method\" domain=\"py\" no-contents-entry=\"0\" no-index=\"0\" no-index-entry=\"0\" no-typesetting=\"0\" nocontentsentry=\"0\" noindex=\"0\" noindexentry=\"0\" objtype=\"method\">\n",
                "                <desc_signature _toc_name=\"MyClass.meth()\" _toc_parts=\"('MyClass', 'meth')\" class=\"MyClass\" classes=\"sig sig-object\" fullname=\"MyClass.meth\" ids=\"MyClass.meth\" module=\"True\">\n",
                "                    <desc_name classes=\"sig-name descname\" xml:space=\"preserve\">\n",
                "                        meth\n",
                "                    <desc_parameterlist multi_line_parameter_list=\"0\" multi_line_trailing_comma=\"1\" xml:space=\"preserve\">\n",
                "                        <desc_parameter xml:space=\"preserve\">\n",
                "                            <desc_sig_name classes=\"n\">\n",
                "                                self\n",
                "                        <desc_parameter xml:space=\"preserve\">\n",
                "                            <desc_sig_name classes=\"n\">\n",
                "                                arg\n",
                "                <desc_content>\n",
                "                    <paragraph>\n",
                "                        Body.\n",
            )
        );
        assert_eq!(
            objects(&out),
            owned(&[
                ("MyClass", "class", "MyClass", false),
                ("MyClass.meth", "method", "MyClass.meth", false),
            ])
        );
    }

    /// Row 10: nested classes stack and unwind — after the inner class's
    /// content, the OUTER class scope is restored (probe nested_classes:
    /// inner signode class="Outer.Inner", _toc_parts ('Outer','Inner','m')).
    #[test]
    fn nested_classes_stack_and_unwind() {
        let out = parse_py(concat!(
            ".. py:class:: Outer\n",
            "\n",
            "   .. py:class:: Inner\n",
            "\n",
            "      .. py:method:: m(x)\n",
            "\n",
            "   .. py:method:: back(x)\n",
        ));
        let pf = out.doctree.root.pformat();
        assert!(pf.contains(
            "class=\"Outer.Inner\" classes=\"sig sig-object\" fullname=\"Outer.Inner.m\" ids=\"Outer.Inner.m\""
        ));
        assert!(pf.contains("_toc_parts=\"('Outer', 'Inner', 'm')\""));
        assert!(
            pf.contains("class=\"Outer\" classes=\"sig sig-object\" fullname=\"Outer.back\""),
            "the inner class popped back to Outer: {pf}"
        );
    }

    // ---- method options / aliasing directives (rows 4, 7; trap 13) -----

    #[test]
    fn method_option_trio_prefixes_and_index_texts() {
        let out = parse_py(concat!(
            ".. py:class:: C\n",
            "\n",
            "   .. py:method:: m1(x)\n",
            "      :classmethod:\n",
            "\n",
            "   .. py:method:: m2(x)\n",
            "      :staticmethod:\n",
            "\n",
            "   .. py:method:: m3(x)\n",
            "      :abstractmethod:\n",
            "      :async:\n",
            "      :final:\n",
        ));
        let pf = out.doctree.root.pformat();
        let kw = |words: &[&str]| {
            let mut s =
                String::from("                    <desc_annotation xml:space=\"preserve\">\n");
            for w in words {
                s.push_str(&format!(
                    "                        <desc_sig_keyword classes=\"k\">\n                            {w}\n                        <desc_sig_space classes=\"w\">\n                             \n"
                ));
            }
            s
        };
        assert!(pf.contains(&kw(&["classmethod"])), "{pf}");
        // `:staticmethod:` prints keyword `static`.
        assert!(pf.contains(&kw(&["static"])), "{pf}");
        assert!(
            pf.contains(&kw(&["final", "abstractmethod", "async"])),
            "{pf}"
        );
        assert!(pf.contains("('single',\\ 'm1()\\ (C\\ class\\ method)',\\ 'C.m1',\\ '',\\ None)"));
        assert!(pf.contains("('single',\\ 'm2()\\ (C\\ static\\ method)',\\ 'C.m2',\\ '',\\ None)"));
        assert!(pf.contains("('single',\\ 'm3()\\ (C\\ method)',\\ 'C.m3',\\ '',\\ None)"));
    }

    /// Trap 13: `py:classmethod`/`py:staticmethod`/`py:decoratormethod`
    /// rewrite `self.name`, so their descs carry objtype `method` with the
    /// injected flag driving prefix and index text.
    #[test]
    fn aliasing_directives_register_the_aliased_objtype() {
        let out = parse_py(concat!(
            ".. py:class:: C\n",
            "\n",
            "   .. py:classmethod:: cm(x)\n",
            "\n",
            "   .. py:staticmethod:: sm(x)\n",
            "\n",
            "   .. py:decoratormethod:: dm\n",
        ));
        let pf = out.doctree.root.pformat();
        assert_eq!(
            pf.matches("desctype=\"method\" domain=\"py\"").count(),
            3,
            "{pf}"
        );
        assert!(pf.contains("('single',\\ 'cm()\\ (C\\ class\\ method)',\\ 'C.cm',\\ '',\\ None)"));
        assert!(pf.contains("('single',\\ 'sm()\\ (C\\ static\\ method)',\\ 'C.sm',\\ '',\\ None)"));
        assert!(pf.contains("('single',\\ 'dm()\\ (C\\ method)',\\ 'C.dm',\\ '',\\ None)"));
        // The decorator method: @ addname first, no forced parens.
        assert!(pf.contains(concat!(
            "                    <desc_addname classes=\"sig-prename descclassname\" xml:space=\"preserve\">\n",
            "                        @\n",
            "                    <desc_name classes=\"sig-name descname\" xml:space=\"preserve\">\n",
            "                        dm\n",
        )));
        assert_eq!(
            objects(&out),
            owned(&[
                ("C", "class", "C", false),
                ("C.cm", "method", "C.cm", false),
                ("C.sm", "method", "C.sm", false),
                ("C.dm", "method", "C.dm", false),
            ])
        );
    }

    // ---- attribute / property / data (rows 5, 7; trap 3) ---------------

    #[test]
    fn attribute_typed_matches_the_sphinx_probe() {
        let out = parse_py(concat!(
            ".. py:class:: C\n",
            "\n",
            "   .. py:attribute:: attr\n",
            "      :type: int\n",
            "      :value: 42\n",
        ));
        let pf = out.doctree.root.pformat();
        assert!(pf.contains(
            "            <index entries=\"('single',\\ 'attr\\ (C\\ attribute)',\\ 'C.attr',\\ '',\\ None)\">\n"
        ));
        assert!(pf.contains(concat!(
            "                <desc_signature _toc_name=\"C.attr\" _toc_parts=\"('C', 'attr')\" class=\"C\" classes=\"sig sig-object\" fullname=\"C.attr\" ids=\"C.attr\" module=\"True\">\n",
            "                    <desc_name classes=\"sig-name descname\" xml:space=\"preserve\">\n",
            "                        attr\n",
            // `:` is desc_sig_punctuation here (trap 3), and the xref
            // carries the enclosing class in py:class.
            "                    <desc_annotation xml:space=\"preserve\">\n",
            "                        <desc_sig_punctuation classes=\"p\">\n",
            "                            :\n",
            "                        <desc_sig_space classes=\"w\">\n",
            "                             \n",
            "                        <pending_xref py:class=\"C\" py:module=\"True\" refdomain=\"py\" refspecific=\"0\" reftarget=\"int\" reftype=\"class\">\n",
            "                            int\n",
            // `=` in `:value:` is desc_sig_punctuation too, unlike
            // parameter defaults (desc_sig_operator).
            "                    <desc_annotation xml:space=\"preserve\">\n",
            "                        <desc_sig_space classes=\"w\">\n",
            "                             \n",
            "                        <desc_sig_punctuation classes=\"p\">\n",
            "                            =\n",
            "                        <desc_sig_space classes=\"w\">\n",
            "                             \n",
            "                        42\n",
        )));
        // Bare-name attribute outside any scope: index text is the name
        // itself (probe value_only_attr).
        let pf = pf_py(".. py:attribute:: a\n   :value: 42\n");
        assert!(pf.contains("    <index entries=\"('single',\\ 'a',\\ 'a',\\ '',\\ None)\">\n"));
    }

    #[test]
    fn property_typed_prefix_and_type_only() {
        let out = parse_py(concat!(
            ".. py:class:: C\n",
            "\n",
            "   .. py:property:: prop\n",
            "      :type: str\n",
            "      :abstractmethod:\n",
            "      :classmethod:\n",
        ));
        let pf = out.doctree.root.pformat();
        // Prefix: abstract ␣ class ␣ property ␣ (three keyword+space pairs).
        assert!(pf.contains(concat!(
            "                    <desc_annotation xml:space=\"preserve\">\n",
            "                        <desc_sig_keyword classes=\"k\">\n",
            "                            abstract\n",
            "                        <desc_sig_space classes=\"w\">\n",
            "                             \n",
            "                        <desc_sig_keyword classes=\"k\">\n",
            "                            class\n",
            "                        <desc_sig_space classes=\"w\">\n",
            "                             \n",
            "                        <desc_sig_keyword classes=\"k\">\n",
            "                            property\n",
            "                        <desc_sig_space classes=\"w\">\n",
            "                             \n",
        )));
        assert!(pf.contains(concat!(
            "                    <desc_annotation xml:space=\"preserve\">\n",
            "                        <desc_sig_punctuation classes=\"p\">\n",
            "                            :\n",
            "                        <desc_sig_space classes=\"w\">\n",
            "                             \n",
            "                        <pending_xref py:class=\"C\" py:module=\"True\" refdomain=\"py\" refspecific=\"0\" reftarget=\"str\" reftype=\"class\">\n",
            "                            str\n",
        )));
        assert!(pf.contains("('single',\\ 'prop\\ (C\\ property)',\\ 'C.prop',\\ '',\\ None)"));
        assert_eq!(
            objects(&out),
            owned(&[
                ("C", "class", "C", false),
                ("C.prop", "property", "C.prop", false),
            ])
        );
    }

    #[test]
    fn data_typed_renders_type_and_value_tails() {
        let out = parse_py(".. py:data:: CONST\n   :type: dict[str, int]\n   :value: {}\n");
        let pf = out.doctree.root.pformat();
        assert!(pf.contains(
            "    <index entries=\"('single',\\ 'CONST\\ (built-in\\ variable)',\\ 'CONST',\\ '',\\ None)\">\n"
        ));
        // dict [ str , int ] — three xrefs with punctuation between.
        for target in ["dict", "str", "int"] {
            assert!(pf.contains(&format!(
                "<pending_xref py:class=\"True\" py:module=\"True\" refdomain=\"py\" refspecific=\"0\" reftarget=\"{target}\" reftype=\"class\">"
            )));
        }
        assert!(pf.contains(concat!(
            "            <desc_annotation xml:space=\"preserve\">\n",
            "                <desc_sig_space classes=\"w\">\n",
            "                     \n",
            "                <desc_sig_punctuation classes=\"p\">\n",
            "                    =\n",
            "                <desc_sig_space classes=\"w\">\n",
            "                     \n",
            "                {}\n",
        )));
        assert_eq!(objects(&out), owned(&[("CONST", "data", "CONST", false)]));
    }

    // ---- decorator / type alias / exception (rows 5, 6, 7) -------------

    #[test]
    fn decorator_basic_matches_the_sphinx_probe() {
        let out = parse_py(".. py:decorator:: mydeco\n");
        assert_eq!(
            out.doctree.root.pformat(),
            concat!(
                "<document source=\"<snippet>\">\n",
                "    <index entries=\"('pair',\\ 'built-in\\ function;\\ mydeco()',\\ 'mydeco',\\ '',\\ None)\">\n",
                "    <desc classes=\"py function\" desctype=\"function\" domain=\"py\" no-contents-entry=\"0\" no-index=\"0\" no-index-entry=\"0\" no-typesetting=\"0\" nocontentsentry=\"0\" noindex=\"0\" noindexentry=\"0\" objtype=\"function\">\n",
                "        <desc_signature _toc_name=\"mydeco()\" _toc_parts=\"('mydeco',)\" class=\"\" classes=\"sig sig-object\" fullname=\"mydeco\" ids=\"mydeco\" module=\"True\">\n",
                "            <desc_addname classes=\"sig-prename descclassname\" xml:space=\"preserve\">\n",
                "                @\n",
                "            <desc_name classes=\"sig-name descname\" xml:space=\"preserve\">\n",
                "                mydeco\n",
                "        <desc_content>\n",
            )
        );
        assert_eq!(
            objects(&out),
            owned(&[("mydeco", "function", "mydeco", false)])
        );
    }

    #[test]
    fn type_alias_canonical_is_display_only() {
        let out = parse_py(".. py:type:: MyAlias\n   :canonical: list[int]\n");
        let pf = out.doctree.root.pformat();
        assert!(pf.contains(
            "    <index entries=\"('single',\\ 'MyAlias',\\ 'MyAlias',\\ '',\\ None)\">\n"
        ));
        assert!(pf.contains(concat!(
            "            <desc_annotation xml:space=\"preserve\">\n",
            "                <desc_sig_keyword classes=\"k\">\n",
            "                    type\n",
            "                <desc_sig_space classes=\"w\">\n",
            "                     \n",
            "            <desc_name classes=\"sig-name descname\" xml:space=\"preserve\">\n",
            "                MyAlias\n",
            "            <desc_annotation xml:space=\"preserve\">\n",
            "                <desc_sig_space classes=\"w\">\n",
            "                     \n",
            "                <desc_sig_punctuation classes=\"p\">\n",
            "                    =\n",
            "                <desc_sig_space classes=\"w\">\n",
            "                     \n",
            "                <pending_xref py:class=\"True\" py:module=\"True\" refdomain=\"py\" refspecific=\"0\" reftarget=\"list\" reftype=\"class\">\n",
            "                    list\n",
            "                <desc_sig_punctuation classes=\"p\">\n",
            "                    [\n",
            "                <pending_xref py:class=\"True\" py:module=\"True\" refdomain=\"py\" refspecific=\"0\" reftarget=\"int\" reftype=\"class\">\n",
            "                    int\n",
            "                <desc_sig_punctuation classes=\"p\">\n",
            "                    ]\n",
        )));
        // NO alias registration on py:type (§6).
        assert_eq!(
            objects(&out),
            owned(&[("MyAlias", "type", "MyAlias", false)])
        );
    }

    #[test]
    fn exception_basic_matches_the_sphinx_probe() {
        assert_eq!(
            pf_py(".. py:exception:: MyError\n"),
            concat!(
                "<document source=\"<snippet>\">\n",
                // Exception index entries are the bare name (trap 10).
                "    <index entries=\"('single',\\ 'MyError',\\ 'MyError',\\ '',\\ None)\">\n",
                "    <desc classes=\"py exception\" desctype=\"exception\" domain=\"py\" no-contents-entry=\"0\" no-index=\"0\" no-index-entry=\"0\" no-typesetting=\"0\" nocontentsentry=\"0\" noindex=\"0\" noindexentry=\"0\" objtype=\"exception\">\n",
                "        <desc_signature _toc_name=\"MyError\" _toc_parts=\"('MyError',)\" class=\"\" classes=\"sig sig-object\" fullname=\"MyError\" ids=\"MyError\" module=\"True\">\n",
                "            <desc_annotation xml:space=\"preserve\">\n",
                "                <desc_sig_keyword classes=\"k\">\n",
                "                    exception\n",
                "                <desc_sig_space classes=\"w\">\n",
                "                     \n",
                "            <desc_name classes=\"sig-name descname\" xml:space=\"preserve\">\n",
                "                MyError\n",
                "        <desc_content>\n",
            )
        );
    }

    // ---- py:module / py:currentmodule (row 9; traps 5, 6) --------------

    /// Row 9 — OUR pre-propagation shape (§Scope-3 sanctioned divergence):
    /// sphinx's recorded doctree has docutils PropagateTargets move the
    /// module target's id onto the next body node (`<target ismod="1"
    /// refid="module-mymod">` + desc `ids="module-mymod"`); this parse
    /// layer runs no transforms, so the target KEEPS its ids and the desc
    /// gains none. Everything else is the probe's bytes.
    #[test]
    fn module_basic_pre_propagation_shape() {
        let out = parse_py(concat!(
            ".. py:module:: mymod\n",
            "   :synopsis: A module.\n",
            "   :platform: Unix\n",
            "\n",
            ".. py:function:: f(x)\n",
            "\n",
            "   Body.\n",
        ));
        assert_eq!(
            out.doctree.root.pformat(),
            concat!(
                "<document source=\"<snippet>\">\n",
                "    <index entries=\"('pair',\\ 'module;\\ mymod',\\ 'module-mymod',\\ '',\\ None)\">\n",
                "    <target ids=\"module-mymod\" ismod=\"1\">\n",
                "    <index entries=\"('single',\\ 'f()\\ (in\\ module\\ mymod)',\\ 'mymod.f',\\ '',\\ None)\">\n",
                "    <desc classes=\"py function\" desctype=\"function\" domain=\"py\" no-contents-entry=\"0\" no-index=\"0\" no-index-entry=\"0\" no-typesetting=\"0\" nocontentsentry=\"0\" noindex=\"0\" noindexentry=\"0\" objtype=\"function\">\n",
                "        <desc_signature _toc_name=\"f()\" _toc_parts=\"('mymod', 'f')\" class=\"\" classes=\"sig sig-object\" fullname=\"f\" ids=\"mymod.f\" module=\"mymod\">\n",
                "            <desc_addname classes=\"sig-prename descclassname\" xml:space=\"preserve\">\n",
                "                mymod.\n",
                "            <desc_name classes=\"sig-name descname\" xml:space=\"preserve\">\n",
                "                f\n",
                "            <desc_parameterlist multi_line_parameter_list=\"0\" multi_line_trailing_comma=\"1\" xml:space=\"preserve\">\n",
                "                <desc_parameter xml:space=\"preserve\">\n",
                "                    <desc_sig_name classes=\"n\">\n",
                "                        x\n",
                "        <desc_content>\n",
                "            <paragraph>\n",
                "                Body.\n",
            )
        );
        assert_eq!(
            objects(&out),
            owned(&[
                ("mymod", "module", "module-mymod", false),
                ("mymod.f", "function", "mymod.f", false),
            ])
        );
        assert_eq!(out.registry.py_modules.len(), 1);
        let m = &out.registry.py_modules[0];
        assert_eq!(
            (
                m.name.as_str(),
                m.node_id.as_str(),
                m.synopsis.as_str(),
                m.platform.as_str(),
                m.deprecated,
                m.lineno
            ),
            ("mymod", "module-mymod", "A module.", "Unix", false, 1)
        );
    }

    /// Module content stays in place (the id-propagation onto it is the
    /// same excluded transform); `:deprecated:` reaches the record.
    #[test]
    fn module_content_and_deprecated() {
        let out = parse_py(".. py:module:: secmod\n\n   Module body content.\n");
        assert_eq!(
            out.doctree.root.pformat(),
            concat!(
                "<document source=\"<snippet>\">\n",
                "    <index entries=\"('pair',\\ 'module;\\ secmod',\\ 'module-secmod',\\ '',\\ None)\">\n",
                "    <target ids=\"module-secmod\" ismod=\"1\">\n",
                "    <paragraph>\n",
                "        Module body content.\n",
            )
        );
        let out = parse_py(".. py:module:: oldmod\n   :deprecated:\n");
        assert!(out.registry.py_modules[0].deprecated);
        assert_eq!(out.registry.py_modules[0].synopsis, "");
    }

    /// Trap 6: `:no-index:` on py:module still sets the module scope —
    /// nothing is emitted or registered for the module itself, but the
    /// following function is module-qualified. `:no-index-entry:` keeps
    /// target + registration and drops only the index node.
    #[test]
    fn module_noindex_and_noindexentry() {
        let out = parse_py(".. py:module:: quietmod\n   :no-index:\n\n.. py:function:: f(x)\n");
        let pf = out.doctree.root.pformat();
        assert!(!pf.contains("module-quietmod"), "{pf}");
        assert!(pf.contains(
            "    <index entries=\"('single',\\ 'f()\\ (in\\ module\\ quietmod)',\\ 'quietmod.f',\\ '',\\ None)\">\n"
        ));
        assert!(out.registry.py_modules.is_empty());
        assert_eq!(
            objects(&out),
            owned(&[("quietmod.f", "function", "quietmod.f", false)])
        );

        let out = parse_py(".. py:module:: halfmod\n   :no-index-entry:\n");
        assert_eq!(
            out.doctree.root.pformat(),
            concat!(
                "<document source=\"<snippet>\">\n",
                "    <target ids=\"module-halfmod\" ismod=\"1\">\n",
            )
        );
        assert_eq!(
            objects(&out),
            owned(&[("halfmod", "module", "module-halfmod", false)])
        );
        assert_eq!(out.registry.py_modules.len(), 1);
    }

    /// Row 9 option-spec quirks: PyModule's spec lacks the old
    /// `noindexentry` spelling → docutils unknown-option ERROR (probe
    /// module_bad_option: nothing runs, no scope set); `no-typesetting`
    /// is accepted but unused by PyModule.run (probe module_no_typesetting).
    #[test]
    fn module_option_spec_quirks() {
        let out = parse_py(".. py:module:: m\n   :noindexentry:\n\n.. py:function:: f(x)\n");
        let pf = out.doctree.root.pformat();
        assert!(pf.contains(concat!(
            "    <system_message level=\"3\" line=\"1\" source=\"<snippet>\" type=\"ERROR\">\n",
            "        <paragraph>\n",
            "            Error in \"py:module\" directive:\n",
            "            unknown option: \"noindexentry\".\n",
        )));
        // The directive never ran: no module scope for the function.
        assert!(pf.contains("('pair',\\ 'built-in\\ function;\\ f()',\\ 'f',\\ '',\\ None)"));
        assert!(out.registry.py_modules.is_empty());

        let out = parse_py(".. py:module:: m2\n   :no-typesetting:\n");
        assert_eq!(
            out.doctree.root.pformat(),
            concat!(
                "<document source=\"<snippet>\">\n",
                "    <index entries=\"('pair',\\ 'module;\\ m2',\\ 'module-m2',\\ '',\\ None)\">\n",
                "    <target ids=\"module-m2\" ismod=\"1\">\n",
            ),
            "accepted-and-inert"
        );
    }

    /// Row 9: `py:currentmodule` sets the scope with no nodes and no
    /// registration; the literal argument `None` pops it (probes
    /// currentmodule / currentmodule_pop).
    #[test]
    fn currentmodule_sets_and_pops() {
        let out = parse_py(".. py:currentmodule:: curmod\n\n.. py:function:: f(x)\n");
        let pf = out.doctree.root.pformat();
        assert!(pf.starts_with(concat!(
            "<document source=\"<snippet>\">\n",
            "    <index entries=\"('single',\\ 'f()\\ (in\\ module\\ curmod)',\\ 'curmod.f',\\ '',\\ None)\">\n",
        )));
        assert!(pf.contains(concat!(
            "            <desc_addname classes=\"sig-prename descclassname\" xml:space=\"preserve\">\n",
            "                curmod.\n",
        )));
        assert_eq!(
            objects(&out),
            owned(&[("curmod.f", "function", "curmod.f", false)])
        );
        assert!(out.registry.py_modules.is_empty());

        let out = parse_py(
            ".. py:currentmodule:: curmod\n\n.. py:currentmodule:: None\n\n.. py:function:: f(x)\n",
        );
        assert!(out
            .doctree
            .root
            .pformat()
            .contains("('pair',\\ 'built-in\\ function;\\ f()',\\ 'f',\\ '',\\ None)"));
    }

    // ---- registration edges (row 8) ------------------------------------

    #[test]
    fn canonical_function_adds_an_aliased_record() {
        let out = parse_py(".. py:function:: new_name()\n   :canonical: old.name\n");
        assert_eq!(
            objects(&out),
            owned(&[
                ("new_name", "function", "new_name", false),
                ("old.name", "function", "new_name", true),
            ])
        );
    }

    /// Row 8/[PY §1.5]: the empty-prefix make_id path — the id IS the
    /// fullname; the second definition collides and takes the `id0`
    /// serial (probe duplicate_functions; the duplicate WARNING itself is
    /// the env layer's job, T9).
    #[test]
    fn duplicate_definitions_take_the_id0_serial() {
        let out = parse_py(".. py:function:: dup()\n\n.. py:function:: dup()\n");
        let pf = out.doctree.root.pformat();
        assert!(pf.contains("fullname=\"dup\" ids=\"dup\" module=\"True\""));
        assert!(pf.contains("fullname=\"dup\" ids=\"id0\" module=\"True\""));
        assert!(pf.contains("('pair',\\ 'built-in\\ function;\\ dup()',\\ 'id0',\\ '',\\ None)"));
        assert_eq!(
            objects(&out),
            owned(&[
                ("dup", "function", "dup", false),
                ("dup", "function", "id0", false),
            ])
        );
    }

    // ---- no-* family (row 7/8, [PY §1.7]) ------------------------------

    #[test]
    fn no_star_option_quartet() {
        // :no-index:: empty index node, no ids, nothing registered.
        let out = parse_py(".. py:function:: hidden()\n   :no-index:\n");
        let pf = out.doctree.root.pformat();
        assert!(pf.contains("    <index entries=\"\">\n"));
        assert!(pf.contains("no-index=\"1\""));
        assert!(pf.contains("noindex=\"1\""));
        assert!(pf.contains(
            "        <desc_signature _toc_name=\"hidden()\" _toc_parts=\"('hidden',)\" class=\"\" classes=\"sig sig-object\" fullname=\"hidden\" module=\"True\">\n"
        ));
        assert!(objects(&out).is_empty());

        // :noindex: old spelling behaves identically (both attrs 1).
        let out = parse_py(".. py:function:: hidden()\n   :noindex:\n");
        let pf2 = out.doctree.root.pformat();
        assert!(pf2.contains("no-index=\"1\"") && pf2.contains("noindex=\"1\""));
        assert!(objects(&out).is_empty());

        // :no-index-entry:: registered with ids, no index entry.
        let out = parse_py(".. py:function:: quiet()\n   :no-index-entry:\n");
        let pf = out.doctree.root.pformat();
        assert!(pf.contains("    <index entries=\"\">\n"));
        assert!(pf.contains("fullname=\"quiet\" ids=\"quiet\" module=\"True\""));
        assert_eq!(
            objects(&out),
            owned(&[("quiet", "function", "quiet", false)])
        );

        // :no-typesetting:: desc collapses to a bare target carrying the
        // collected ids; index entry + registration survive.
        let out = parse_py(".. py:function:: invisible()\n   :no-typesetting:\n");
        assert_eq!(
            out.doctree.root.pformat(),
            concat!(
                "<document source=\"<snippet>\">\n",
                "    <index entries=\"('pair',\\ 'built-in\\ function;\\ invisible()',\\ 'invisible',\\ '',\\ None)\">\n",
                "    <target ids=\"invisible\">\n",
            )
        );
        assert_eq!(
            objects(&out),
            owned(&[("invisible", "function", "invisible", false)])
        );
    }

    // ---- strip_signature_backslash (row 12) ----------------------------

    #[test]
    fn strip_signature_backslash_strips_before_parsing() {
        let cfg = PySigConfig {
            strip_signature_backslash: true,
            ..PySigConfig::default()
        };
        let pf = pf_py_cfg(".. py:function:: f(a\\_b)\n", cfg);
        assert!(pf.contains(concat!(
            "                    <desc_sig_name classes=\"n\">\n",
            "                        a_b\n",
        )));
        // Default off: the backslash survives into the (pseudo-parsed)
        // parameter (probe strip_backslash_off).
        let pf = pf_py(".. py:function:: f(a\\_b)\n");
        assert!(pf.contains(concat!(
            "                    <desc_sig_name classes=\"n\">\n",
            "                        a\\_b\n",
        )));
    }

    // ---- error channels (row 13) ---------------------------------------

    /// Row 13: duplicate parameter names WARN (`could not parse arglist`)
    /// with the pseudo fallback; tp-list failures WARN (`could not parse
    /// tp_list`) with the exception text interpolated — bytes pinned by
    /// probes arglist_dup_warning / tp_list_warning / tp_list_tokerror.
    #[test]
    fn arglist_and_tp_list_error_paths_warn() {
        let out = parse_py(".. py:function:: f(a, a)\n");
        assert_eq!(
            out.registry
                .log_warnings
                .iter()
                .map(|w| (w.message.as_str(), w.line))
                .collect::<Vec<_>>(),
            vec![(
                "could not parse arglist ('a, a'): duplicate parameter name: 'a'",
                1
            )]
        );
        // Pseudo fallback still renders both parameters.
        let pf = out.doctree.root.pformat();
        assert_eq!(
            pf.matches(concat!(
                "                <desc_parameter xml:space=\"preserve\">\n",
                "                    <desc_sig_name classes=\"n\">\n",
                "                        a\n",
            ))
            .count(),
            2
        );

        let out = parse_py(".. py:function:: f[*Ts: int](x)\n");
        assert_eq!(
            out.registry.log_warnings[0].message,
            "could not parse tp_list ('*Ts: int'): type parameter bound or constraint is not allowed for variadic positional parameters"
        );
        // The failed tp list is simply absent; the signature continues.
        let pf = out.doctree.root.pformat();
        assert!(!pf.contains("desc_type_parameter_list"));
        assert!(pf.contains("fullname=\"f\" ids=\"f\""));

        let out = parse_py(".. py:function:: f[(T](x)\n");
        assert_eq!(
            out.registry.log_warnings[0].message,
            "could not parse tp_list ('(T'): ('unexpected EOF in multi-line statement', (1, 0))"
        );

        // The SyntaxError channel stays SILENT (debug level): brackets
        // fall back to the pseudo parser with no warning.
        let out = parse_py(".. py:function:: func(a[, b])\n");
        assert!(out.registry.log_warnings.is_empty());
        assert!(out.doctree.root.pformat().contains("<desc_optional"));
    }

    /// The greedy-arglist edge stays TOTAL but diverges from sphinx:
    /// sphinx's `_parse_arglist` wraps the captured `x) -> (int, str` in
    /// `def func(...): pass`, where the stray `)` closes the def and the
    /// tuple parses as a (discarded) def-level return annotation — params
    /// [x]. Our arglist grammar rejects the stray `)` (SyntaxError channel,
    /// silent) and pseudo-parses instead. Known divergence, excluded from
    /// the T8 corpus; this pin is a totality guard, not an oracle match.
    #[test]
    fn greedy_arglist_edge_is_total_and_silent() {
        let out = parse_py(".. py:function:: f(x) -> (int, str)\n");
        assert!(out.registry.log_warnings.is_empty());
        assert_eq!(objects(&out), owned(&[("f", "function", "f", false)]));
        assert!(out.doctree.root.pformat().contains("x) -> (int"));
    }

    // ---- toc config variants (row 3, [SIG §2.2]) -----------------------

    #[test]
    fn toc_entry_config_variants() {
        // hide: last part only (probe toc_hide: _toc_name "m()").
        let cfg = PySigConfig {
            toc_object_entries_show_parents: "hide".to_string(),
            ..PySigConfig::default()
        };
        let pf = pf_py_cfg(".. py:method:: C.m(x)\n", cfg);
        assert!(
            pf.contains("_toc_name=\"m()\" _toc_parts=\"('C', 'm')\""),
            "{pf}"
        );

        // all: every hierarchy part joined — the module joins the parts.
        let cfg = PySigConfig {
            toc_object_entries_show_parents: "all".to_string(),
            ..PySigConfig::default()
        };
        let pf = pf_py_cfg(".. py:module:: pkg\n\n.. py:method:: C.m(x)\n", cfg);
        assert!(
            pf.contains("_toc_name=\"pkg.C.m()\" _toc_parts=\"('pkg', 'C', 'm')\""),
            "{pf}"
        );

        // add_function_parentheses=false drops the parens from _toc_name
        // (and only functions/methods ever get them).
        let cfg = PySigConfig {
            add_function_parentheses: false,
            ..PySigConfig::default()
        };
        let pf = pf_py_cfg(".. py:method:: C.m(x)\n", cfg);
        assert!(
            pf.contains("_toc_name=\"C.m\" _toc_parts=\"('C', 'm')\""),
            "{pf}"
        );

        // toc_object_entries=false: empty toc attrs, everything else kept
        // (probe toc_off).
        let cfg = PySigConfig {
            toc_object_entries: false,
            ..PySigConfig::default()
        };
        let pf = pf_py_cfg(".. py:function:: f(x)\n", cfg);
        assert!(pf.contains(
            "        <desc_signature _toc_name=\"\" _toc_parts=\"()\" class=\"\" classes=\"sig sig-object\" fullname=\"f\" ids=\"f\" module=\"True\">\n"
        ));
    }

    // ---- multi-line signature wrapping ([SIG §2.5] probes) -------------

    #[test]
    fn long_signatures_wrap_and_single_line_options_suppress() {
        let cfg = || PySigConfig {
            maximum_signature_line_length: Some(20),
            ..PySigConfig::default()
        };
        let pf = pf_py_cfg(
            ".. py:function:: really_long_function_name(argument_one, argument_two)\n",
            cfg(),
        );
        assert!(pf.contains(
            "<desc_parameterlist multi_line_parameter_list=\"1\" multi_line_trailing_comma=\"1\" xml:space=\"preserve\">"
        ));
        let pf = pf_py_cfg(
            ".. py:function:: really_long_function_name(argument_one, argument_two)\n   :single-line-parameter-list:\n",
            cfg(),
        );
        assert!(pf.contains(
            "<desc_parameterlist multi_line_parameter_list=\"0\" multi_line_trailing_comma=\"1\" xml:space=\"preserve\">"
        ));
        let pf = pf_py_cfg(
            ".. py:class:: LongName[TypeParamOne, TypeParamTwo]\n",
            cfg(),
        );
        assert!(pf.contains(
            "<desc_type_parameter_list multi_line_parameter_list=\"1\" multi_line_trailing_comma=\"1\" xml:space=\"preserve\">"
        ));
        let pf = pf_py_cfg(
            ".. py:class:: LongName[TypeParamOne, TypeParamTwo]\n   :single-line-type-parameter-list:\n",
            cfg(),
        );
        assert!(pf.contains(
            "<desc_type_parameter_list multi_line_parameter_list=\"0\" multi_line_trailing_comma=\"1\" xml:space=\"preserve\">"
        ));
    }

    // ---- misc: add_module_names off ------------------------------------

    #[test]
    fn add_module_names_off_drops_the_module_addname() {
        let cfg = PySigConfig {
            add_module_names: false,
            ..PySigConfig::default()
        };
        let pf = pf_py_cfg(".. py:module:: mymod\n\n.. py:function:: f(x)\n", cfg);
        assert!(!pf.contains("desc_addname"), "{pf}");
        // Registration and index stay module-qualified regardless.
        assert!(pf.contains("ids=\"mymod.f\" module=\"mymod\""));
        assert!(
            pf.contains("('single',\\ 'f()\\ (in\\ module\\ mymod)',\\ 'mymod.f',\\ '',\\ None)")
        );
    }

    /// Row 7: the method-family index clsname is module-qualified iff
    /// `add_module_names` (probes method_module_qualified /
    /// method_module_qualified_off: `meth() (mymod.C method)` vs
    /// `meth() (C method)` — the node id stays qualified either way).
    #[test]
    fn method_index_clsname_qualification_follows_add_module_names() {
        let src = ".. py:module:: mymod\n\n.. py:class:: C\n\n   .. py:method:: meth(x)\n";
        let pf = pf_py(src);
        assert!(pf.contains(
            "('single',\\ 'meth()\\ (mymod.C\\ method)',\\ 'mymod.C.meth',\\ '',\\ None)"
        ));
        let cfg = PySigConfig {
            add_module_names: false,
            ..PySigConfig::default()
        };
        let pf = pf_py_cfg(src, cfg);
        assert!(
            pf.contains("('single',\\ 'meth()\\ (C\\ method)',\\ 'mymod.C.meth',\\ '',\\ None)")
        );
    }

    /// Row 10: a NON-nesting kind with a written prefix scopes its own
    /// content to that prefix — `before_content`'s `name_prefix.strip('.')`
    /// branch (probe method_prefix_scope: the xref inside carries
    /// `py:class="D"`), and the scope pops after the content.
    #[test]
    fn a_prefixed_method_scopes_its_content_without_nesting() {
        let out = parse_py(
            ".. py:method:: D.meth(x)\n\n   :py:func:`target`\n\n.. py:function:: after(x)\n",
        );
        let pf = out.doctree.root.pformat();
        assert!(pf.contains(
            "                <pending_xref py:class=\"D\" py:module=\"True\" refdoc=\"index\" refdomain=\"py\" refexplicit=\"0\" reftarget=\"target\" reftype=\"func\" refwarn=\"0\">\n"
        ));
        assert!(pf.contains("('single',\\ 'meth()\\ (D\\ method)',\\ 'D.meth',\\ '',\\ None)"));
        // after_content restored the empty scope for the next directive.
        assert!(pf.contains("fullname=\"after\" ids=\"after\" module=\"True\""));
    }

    /// [PY §3.1] probe role_in_module_scope: a py role inside a class's
    /// content carries the enclosing ref_context on the pending_xref —
    /// `py:class="C" py:module="mymod"` instead of the None sentinels.
    #[test]
    fn a_py_role_inside_a_scope_stamps_the_ref_context() {
        let pf = pf_py(".. py:module:: mymod\n\n.. py:class:: C\n\n   :py:func:`target`\n");
        assert!(
            pf.contains(concat!(
                "                <pending_xref py:class=\"C\" py:module=\"mymod\" refdoc=\"index\" refdomain=\"py\" refexplicit=\"0\" reftarget=\"target\" reftype=\"func\" refwarn=\"0\">\n",
                "                    <literal classes=\"xref py py-func\">\n",
                "                        target()\n",
            )),
            "{pf}"
        );
    }
}

/// Doc-field transformation (M2 wave 4.5 task 7). Every expected pformat
/// below is pasted verbatim from the Sphinx 9.1.0 oracle — this task's
/// probe_t7 run (harness3 conventions, pinned wheels) over the research
/// specs [PY §1.6 "Doc fields"] and [SIG §4.2 item 2 / A.5] — never
/// written from memory.
#[cfg(test)]
mod py_docfield_tests {
    use super::*;
    use crate::py::PySigConfig;
    use crate::rst::{parse_rst_full, ParseOptions};

    fn pf_cfg(src: &str, py: PySigConfig) -> String {
        let opts = ParseOptions {
            source_path: "<snippet>".into(),
            sphinx: true,
            docname: "index".into(),
            exclude_patterns: Vec::new(),
            py,
            found_docs: None,
        };
        parse_rst_full(src, &opts).doctree.root.pformat()
    }

    fn pf(src: &str) -> String {
        pf_cfg(src, PySigConfig::default())
    }

    fn unqual() -> PySigConfig {
        PySigConfig {
            python_use_unqualified_type_names: true,
            ..PySigConfig::default()
        }
    }

    /// `PyXrefMixin._delimiters_re` split parity, pinned against the
    /// Python `re.split` outputs (delimiters kept, empties dropped).
    #[test]
    fn the_delimiter_split_matches_python_re_split() {
        let split = |t: &str| split_type_delimiters(t);
        let owned = |v: &[(&str, bool)]| -> Vec<(String, bool)> {
            v.iter().map(|(s, d)| (s.to_string(), *d)).collect()
        };
        assert_eq!(
            split("int or str"),
            owned(&[("int", false), (" or ", true), ("str", false)])
        );
        assert_eq!(
            split("Literal[1, 2]"),
            owned(&[
                ("Literal", false),
                ("[", true),
                ("1", false),
                (", ", true),
                ("2", false),
                ("]", true),
            ])
        );
        assert_eq!(
            split("list[int]"),
            owned(&[("list", false), ("[", true), ("int", false), ("]", true)])
        );
        assert_eq!(
            split("a | b"),
            owned(&[("a", false), (" | ", true), ("b", false)])
        );
        assert_eq!(split("int..."), owned(&[("int", false), ("...", true)]));
        assert_eq!(
            split("dict of str"),
            owned(&[("dict", false), (" of ", true), ("str", false)])
        );
        // No trailing whitespace after `or` -> the whole thing is text.
        assert_eq!(split("x or"), owned(&[("x or", false)]));
        assert_eq!(split("of or"), owned(&[("of or", false)]));
        assert_eq!(split(" or "), owned(&[(" or ", true)]));
        // A bracket delimiter swallows a following `or `.
        assert_eq!(
            split("int, or str"),
            owned(&[("int", false), (", or ", true), ("str", false)])
        );
        assert_eq!(
            split("tuple(int)"),
            owned(&[("tuple", false), ("(", true), ("int", false), (")", true)])
        );
        assert_eq!(
            split("a|b"),
            owned(&[("a", false), ("|", true), ("b", false)])
        );
    }

    /// std kinds run the transformer with an EMPTY typemap
    /// (`directives/__init__.py:295`; no std kind declares
    /// `doc_field_types`): every field takes the unknown branch —
    /// `Param x`, body untouched, fresh field_list (probe envvar_param).
    #[test]
    fn a_std_field_is_capitalized_and_passed_through() {
        assert_eq!(
            pf(".. envvar:: HOME_X\n\n   :param x: not transformed\n"),
            concat!(
                "<document source=\"<snippet>\">\n",
                "    <index entries=\"('single',\\ 'environment\\ variable;\\ HOME_X',\\ 'envvar-HOME_X',\\ '',\\ None)\">\n",
                "    <desc classes=\"std envvar\" desctype=\"envvar\" domain=\"std\" no-contents-entry=\"0\" no-index=\"0\" no-index-entry=\"0\" no-typesetting=\"0\" nocontentsentry=\"0\" noindex=\"0\" noindexentry=\"0\" objtype=\"envvar\">\n",
                "        <desc_signature _toc_name=\"\" _toc_parts=\"()\" classes=\"sig sig-object\" ids=\"envvar-HOME_X\">\n",
                "            <desc_name classes=\"sig-name descname\" xml:space=\"preserve\">\n",
                "                HOME_X\n",
                "        <desc_content>\n",
                "            <field_list>\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Param x\n",
                "                    <field_body>\n",
                "                        <paragraph>\n",
                "                            not transformed\n",
            )
        );
    }

    /// `filter_meta_fields` guards `domain == 'py'`
    /// (`domains/python/__init__.py:610-611`), so a std `:meta private:`
    /// SURVIVES and renders renamed `Meta private` with its empty body
    /// (probe envvar_meta).
    #[test]
    fn a_std_meta_field_is_not_removed() {
        assert_eq!(
            pf(".. envvar:: HOME_Y\n\n   :meta private:\n"),
            concat!(
                "<document source=\"<snippet>\">\n",
                "    <index entries=\"('single',\\ 'environment\\ variable;\\ HOME_Y',\\ 'envvar-HOME_Y',\\ '',\\ None)\">\n",
                "    <desc classes=\"std envvar\" desctype=\"envvar\" domain=\"std\" no-contents-entry=\"0\" no-index=\"0\" no-index-entry=\"0\" no-typesetting=\"0\" nocontentsentry=\"0\" noindex=\"0\" noindexentry=\"0\" objtype=\"envvar\">\n",
                "        <desc_signature _toc_name=\"\" _toc_parts=\"()\" classes=\"sig sig-object\" ids=\"envvar-HOME_Y\">\n",
                "            <desc_name classes=\"sig-name descname\" xml:space=\"preserve\">\n",
                "                HOME_Y\n",
                "        <desc_content>\n",
                "            <field_list>\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Meta private\n",
                "                    <field_body>\n",
            )
        );
    }

    /// Confval's `transform_content` inserts its own field_list BEFORE the
    /// transformer runs; both it and the body's field_list are direct
    /// desc_content children and both transform — pass-through for the
    /// generated `Type` field (same name, body untouched) and the
    /// unknown rename for `:param y:` (probe confval_type_and_field).
    #[test]
    fn the_confval_generated_field_list_transforms_too() {
        assert_eq!(
            pf(".. confval:: s\n   :type: text with *emphasis*\n\n   :param y: field in body\n"),
            concat!(
                "<document source=\"<snippet>\">\n",
                "    <index entries=\"('pair',\\ 's;\\ configuration\\ value',\\ 'confval-s',\\ '',\\ None)\">\n",
                "    <desc classes=\"std confval\" desctype=\"confval\" domain=\"std\" no-contents-entry=\"0\" no-index=\"0\" no-index-entry=\"0\" no-typesetting=\"0\" nocontentsentry=\"0\" noindex=\"0\" noindexentry=\"0\" objtype=\"confval\">\n",
                "        <desc_signature _toc_name=\"s\" _toc_parts=\"('s',)\" classes=\"sig sig-object\" fullname=\"s\" ids=\"confval-s\">\n",
                "            <desc_name classes=\"sig-name descname\" xml:space=\"preserve\">\n",
                "                s\n",
                "        <desc_content>\n",
                "            <field_list>\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Type\n",
                "                    <field_body>\n",
                "                        text with \n",
                "                        <emphasis>\n",
                "                            emphasis\n",
                "            <field_list>\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Param y\n",
                "                    <field_body>\n",
                "                        <paragraph>\n",
                "                            field in body\n",
            )
        );
    }

    /// `describe` (bare docutils registration, domain='') transforms with
    /// the empty map too (probe describe_param).
    #[test]
    fn describe_fields_take_the_unknown_branch() {
        let pf = pf(".. describe:: foo\n\n   :param x: desc\n");
        assert!(
            pf.contains(concat!(
                "        <desc_content>\n",
                "            <field_list>\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Param x\n",
                "                    <field_body>\n",
                "                        <paragraph>\n",
                "                            desc\n",
            )),
            "{pf}"
        );
    }

    /// `option` transforms with the empty map (probe option_param).
    #[test]
    fn option_fields_take_the_unknown_branch() {
        let pf = pf(".. option:: --x\n\n   :param x: desc\n");
        assert!(
            pf.contains(concat!(
                "        <desc_content>\n",
                "            <field_list>\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Param x\n",
                "                    <field_body>\n",
                "                        <paragraph>\n",
                "                            desc\n",
            )),
            "{pf}"
        );
    }

    /// The py field names mean nothing to a std kind: `:returns:` /
    /// `:rtype:` are renamed `Returns`/`Rtype` and left as plain fields
    /// (probe envvar_multi_fields).
    #[test]
    fn py_field_names_are_unknown_on_std_kinds() {
        let pf = pf(".. envvar:: HOME_W\n\n   :param a: one\n   :returns: two\n   :rtype: bool\n");
        assert!(
            pf.contains(concat!(
                "        <desc_content>\n",
                "            <field_list>\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Param a\n",
                "                    <field_body>\n",
                "                        <paragraph>\n",
                "                            one\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Returns\n",
                "                    <field_body>\n",
                "                        <paragraph>\n",
                "                            two\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Rtype\n",
                "                    <field_body>\n",
                "                        <paragraph>\n",
                "                            bool\n",
            )),
            "{pf}"
        );
    }

    /// The [PY §1.6] `function_fields` probe: grouped Parameters with
    /// `:param int a:` inline-type and `:type b:` merge, Returns, Return
    /// type body role, Raises `exc` xref.
    #[test]
    fn function_fields_probe_byte_for_byte() {
        assert_eq!(
            pf(".. py:function:: f(a, b)\n\n   :param int a: first\n   :param b: second\n   :type b: str\n   :returns: something\n   :rtype: bool\n   :raises ValueError: when bad\n"),
            concat!(
                "<document source=\"<snippet>\">\n",
                "    <index entries=\"('pair',\\ 'built-in\\ function;\\ f()',\\ 'f',\\ '',\\ None)\">\n",
                "    <desc classes=\"py function\" desctype=\"function\" domain=\"py\" no-contents-entry=\"0\" no-index=\"0\" no-index-entry=\"0\" no-typesetting=\"0\" nocontentsentry=\"0\" noindex=\"0\" noindexentry=\"0\" objtype=\"function\">\n",
                "        <desc_signature _toc_name=\"f()\" _toc_parts=\"('f',)\" class=\"\" classes=\"sig sig-object\" fullname=\"f\" ids=\"f\" module=\"True\">\n",
                "            <desc_name classes=\"sig-name descname\" xml:space=\"preserve\">\n",
                "                f\n",
                "            <desc_parameterlist multi_line_parameter_list=\"0\" multi_line_trailing_comma=\"1\" xml:space=\"preserve\">\n",
                "                <desc_parameter xml:space=\"preserve\">\n",
                "                    <desc_sig_name classes=\"n\">\n",
                "                        a\n",
                "                <desc_parameter xml:space=\"preserve\">\n",
                "                    <desc_sig_name classes=\"n\">\n",
                "                        b\n",
                "        <desc_content>\n",
                "            <field_list>\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Parameters\n",
                "                    <field_body>\n",
                "                        <bullet_list>\n",
                "                            <list_item>\n",
                "                                <paragraph>\n",
                "                                    <literal_strong>\n",
                "                                        a\n",
                "                                     (\n",
                "                                    <pending_xref py:class=\"True\" py:module=\"True\" refdomain=\"py\" refexplicit=\"0\" refspecific=\"1\" reftarget=\"int\" reftype=\"class\">\n",
                "                                        <literal_emphasis>\n",
                "                                            int\n",
                "                                    )\n",
                "                                     -- \n",
                "                                    first\n",
                "                            <list_item>\n",
                "                                <paragraph>\n",
                "                                    <literal_strong>\n",
                "                                        b\n",
                "                                     (\n",
                "                                    <pending_xref py:class=\"True\" py:module=\"True\" refdomain=\"py\" refexplicit=\"0\" refspecific=\"1\" reftarget=\"str\" reftype=\"class\">\n",
                "                                        <literal_emphasis>\n",
                "                                            str\n",
                "                                    )\n",
                "                                     -- \n",
                "                                    second\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Returns\n",
                "                    <field_body>\n",
                "                        <paragraph>\n",
                "                            something\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Return type\n",
                "                    <field_body>\n",
                "                        <paragraph>\n",
                "                            <pending_xref py:class=\"True\" py:module=\"True\" refdomain=\"py\" refexplicit=\"0\" refspecific=\"1\" reftarget=\"bool\" reftype=\"class\">\n",
                "                                bool\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Raises\n",
                "                    <field_body>\n",
                "                        <paragraph>\n",
                "                            <pending_xref py:class=\"True\" py:module=\"True\" refdomain=\"py\" refexplicit=\"0\" refspecific=\"1\" reftarget=\"ValueError\" reftype=\"exc\">\n",
                "                                <literal_strong>\n",
                "                                    ValueError\n",
                "                             -- \n",
                "                            when bad\n",
            )
        );
    }

    /// TypedField `can_collapse`: one item is a bare paragraph in the
    /// field_body — no bullet_list (probe param_single; F-U1 shape).
    #[test]
    fn a_single_param_collapses_to_a_paragraph() {
        assert_eq!(
            pf(".. py:function:: f(x)\n\n   :param x: only one\n"),
            concat!(
                "<document source=\"<snippet>\">\n",
                "    <index entries=\"('pair',\\ 'built-in\\ function;\\ f()',\\ 'f',\\ '',\\ None)\">\n",
                "    <desc classes=\"py function\" desctype=\"function\" domain=\"py\" no-contents-entry=\"0\" no-index=\"0\" no-index-entry=\"0\" no-typesetting=\"0\" nocontentsentry=\"0\" noindex=\"0\" noindexentry=\"0\" objtype=\"function\">\n",
                "        <desc_signature _toc_name=\"f()\" _toc_parts=\"('f',)\" class=\"\" classes=\"sig sig-object\" fullname=\"f\" ids=\"f\" module=\"True\">\n",
                "            <desc_name classes=\"sig-name descname\" xml:space=\"preserve\">\n",
                "                f\n",
                "            <desc_parameterlist multi_line_parameter_list=\"0\" multi_line_trailing_comma=\"1\" xml:space=\"preserve\">\n",
                "                <desc_parameter xml:space=\"preserve\">\n",
                "                    <desc_sig_name classes=\"n\">\n",
                "                        x\n",
                "        <desc_content>\n",
                "            <field_list>\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Parameters\n",
                "                    <field_body>\n",
                "                        <paragraph>\n",
                "                            <literal_strong>\n",
                "                                x\n",
                "                             -- \n",
                "                            only one\n",
            )
        );
    }

    /// `:type x:` content lands as ` ( <xref> )` inside the single
    /// collapsed param entry (probe type_merge).
    #[test]
    fn a_type_field_merges_into_the_param_entry() {
        assert_eq!(
            pf(".. py:function:: f(x)\n\n   :param x: thing\n   :type x: str\n"),
            concat!(
                "<document source=\"<snippet>\">\n",
                "    <index entries=\"('pair',\\ 'built-in\\ function;\\ f()',\\ 'f',\\ '',\\ None)\">\n",
                "    <desc classes=\"py function\" desctype=\"function\" domain=\"py\" no-contents-entry=\"0\" no-index=\"0\" no-index-entry=\"0\" no-typesetting=\"0\" nocontentsentry=\"0\" noindex=\"0\" noindexentry=\"0\" objtype=\"function\">\n",
                "        <desc_signature _toc_name=\"f()\" _toc_parts=\"('f',)\" class=\"\" classes=\"sig sig-object\" fullname=\"f\" ids=\"f\" module=\"True\">\n",
                "            <desc_name classes=\"sig-name descname\" xml:space=\"preserve\">\n",
                "                f\n",
                "            <desc_parameterlist multi_line_parameter_list=\"0\" multi_line_trailing_comma=\"1\" xml:space=\"preserve\">\n",
                "                <desc_parameter xml:space=\"preserve\">\n",
                "                    <desc_sig_name classes=\"n\">\n",
                "                        x\n",
                "        <desc_content>\n",
                "            <field_list>\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Parameters\n",
                "                    <field_body>\n",
                "                        <paragraph>\n",
                "                            <literal_strong>\n",
                "                                x\n",
                "                             (\n",
                "                            <pending_xref py:class=\"True\" py:module=\"True\" refdomain=\"py\" refexplicit=\"0\" refspecific=\"1\" reftarget=\"str\" reftype=\"class\">\n",
                "                                <literal_emphasis>\n",
                "                                    str\n",
                "                            )\n",
                "                             -- \n",
                "                            thing\n",
            )
        );
    }

    /// `filter_meta_fields` removes the `:meta:` field BEFORE the
    /// transformer runs; the emptied `<field_list>` remains [PY §1.6
    /// meta probe].
    #[test]
    fn meta_fields_are_removed_but_the_field_list_remains() {
        assert_eq!(
            pf(".. py:function:: f()\n\n   :meta private:\n"),
            concat!(
                "<document source=\"<snippet>\">\n",
                "    <index entries=\"('pair',\\ 'built-in\\ function;\\ f()',\\ 'f',\\ '',\\ None)\">\n",
                "    <desc classes=\"py function\" desctype=\"function\" domain=\"py\" no-contents-entry=\"0\" no-index=\"0\" no-index-entry=\"0\" no-typesetting=\"0\" nocontentsentry=\"0\" noindex=\"0\" noindexentry=\"0\" objtype=\"function\">\n",
                "        <desc_signature _toc_name=\"f()\" _toc_parts=\"('f',)\" class=\"\" classes=\"sig sig-object\" fullname=\"f\" ids=\"f\" module=\"True\">\n",
                "            <desc_name classes=\"sig-name descname\" xml:space=\"preserve\">\n",
                "                f\n",
                "            <desc_parameterlist xml:space=\"preserve\">\n",
                "        <desc_content>\n",
                "            <field_list>\n",
            )
        );
    }

    /// `PyXrefMixin.make_xrefs` splits `int or str` into two xrefs around
    /// a `literal_emphasis` ` or ` delimiter (probe multi_type_or).
    #[test]
    fn a_multi_type_field_splits_on_or() {
        assert_eq!(
            pf(".. py:function:: f(x)\n\n   :param x: thing\n   :type x: int or str\n"),
            concat!(
                "<document source=\"<snippet>\">\n",
                "    <index entries=\"('pair',\\ 'built-in\\ function;\\ f()',\\ 'f',\\ '',\\ None)\">\n",
                "    <desc classes=\"py function\" desctype=\"function\" domain=\"py\" no-contents-entry=\"0\" no-index=\"0\" no-index-entry=\"0\" no-typesetting=\"0\" nocontentsentry=\"0\" noindex=\"0\" noindexentry=\"0\" objtype=\"function\">\n",
                "        <desc_signature _toc_name=\"f()\" _toc_parts=\"('f',)\" class=\"\" classes=\"sig sig-object\" fullname=\"f\" ids=\"f\" module=\"True\">\n",
                "            <desc_name classes=\"sig-name descname\" xml:space=\"preserve\">\n",
                "                f\n",
                "            <desc_parameterlist multi_line_parameter_list=\"0\" multi_line_trailing_comma=\"1\" xml:space=\"preserve\">\n",
                "                <desc_parameter xml:space=\"preserve\">\n",
                "                    <desc_sig_name classes=\"n\">\n",
                "                        x\n",
                "        <desc_content>\n",
                "            <field_list>\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Parameters\n",
                "                    <field_body>\n",
                "                        <paragraph>\n",
                "                            <literal_strong>\n",
                "                                x\n",
                "                             (\n",
                "                            <pending_xref py:class=\"True\" py:module=\"True\" refdomain=\"py\" refexplicit=\"0\" refspecific=\"1\" reftarget=\"int\" reftype=\"class\">\n",
                "                                <literal_emphasis>\n",
                "                                    int\n",
                "                            <literal_emphasis>\n",
                "                                 or \n",
                "                            <pending_xref py:class=\"True\" py:module=\"True\" refdomain=\"py\" refexplicit=\"0\" refspecific=\"1\" reftarget=\"str\" reftype=\"class\">\n",
                "                                <literal_emphasis>\n",
                "                                    str\n",
                "                            )\n",
                "                             -- \n",
                "                            thing\n",
            )
        );
    }

    /// [SIG §4.2 item 2, probe F-U2]: under
    /// `python_use_unqualified_type_names`, the typed-field xref gets the
    /// two `pending_xref_condition` children with the `literal_emphasis`
    /// innernode wrapped INSIDE each condition.
    #[test]
    fn unqualified_type_names_wrap_field_xrefs_in_conditions() {
        assert_eq!(
            pf_cfg(".. py:function:: f(x)\n\n   :param x: thing\n   :type x: pkg.Cls\n", unqual()),
            concat!(
                "<document source=\"<snippet>\">\n",
                "    <index entries=\"('pair',\\ 'built-in\\ function;\\ f()',\\ 'f',\\ '',\\ None)\">\n",
                "    <desc classes=\"py function\" desctype=\"function\" domain=\"py\" no-contents-entry=\"0\" no-index=\"0\" no-index-entry=\"0\" no-typesetting=\"0\" nocontentsentry=\"0\" noindex=\"0\" noindexentry=\"0\" objtype=\"function\">\n",
                "        <desc_signature _toc_name=\"f()\" _toc_parts=\"('f',)\" class=\"\" classes=\"sig sig-object\" fullname=\"f\" ids=\"f\" module=\"True\">\n",
                "            <desc_name classes=\"sig-name descname\" xml:space=\"preserve\">\n",
                "                f\n",
                "            <desc_parameterlist multi_line_parameter_list=\"0\" multi_line_trailing_comma=\"1\" xml:space=\"preserve\">\n",
                "                <desc_parameter xml:space=\"preserve\">\n",
                "                    <desc_sig_name classes=\"n\">\n",
                "                        x\n",
                "        <desc_content>\n",
                "            <field_list>\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Parameters\n",
                "                    <field_body>\n",
                "                        <paragraph>\n",
                "                            <literal_strong>\n",
                "                                x\n",
                "                             (\n",
                "                            <pending_xref py:class=\"True\" py:module=\"True\" refdomain=\"py\" refexplicit=\"0\" refspecific=\"1\" reftarget=\"pkg.Cls\" reftype=\"class\">\n",
                "                                <pending_xref_condition condition=\"resolved\">\n",
                "                                    <literal_emphasis>\n",
                "                                        Cls\n",
                "                                <pending_xref_condition condition=\"*\">\n",
                "                                    <literal_emphasis>\n",
                "                                        pkg.Cls\n",
                "                            )\n",
                "                             -- \n",
                "                            thing\n",
            )
        );
    }

    /// GroupedField collapse applies only to a single item; two `:raises:`
    /// build a bullet_list (probe raises_two).
    #[test]
    fn multiple_raises_stay_a_bullet_list() {
        let pf =
            pf(".. py:function:: f()\n\n   :raises ValueError: bad\n   :raises TypeError: worse\n");
        assert!(
            pf.contains(concat!(
                "        <desc_content>\n",
                "            <field_list>\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Raises\n",
                "                    <field_body>\n",
                "                        <bullet_list>\n",
                "                            <list_item>\n",
                "                                <paragraph>\n",
                "                                    <pending_xref py:class=\"True\" py:module=\"True\" refdomain=\"py\" refexplicit=\"0\" refspecific=\"1\" reftarget=\"ValueError\" reftype=\"exc\">\n",
                "                                        <literal_strong>\n",
                "                                            ValueError\n",
                "                                     -- \n",
                "                                    bad\n",
                "                            <list_item>\n",
                "                                <paragraph>\n",
                "                                    <pending_xref py:class=\"True\" py:module=\"True\" refdomain=\"py\" refexplicit=\"0\" refspecific=\"1\" reftarget=\"TypeError\" reftype=\"exc\">\n",
                "                                        <literal_strong>\n",
                "                                            TypeError\n",
                "                                     -- \n",
                "                                    worse\n",
            )),
            "{pf}"
        );
    }

    /// `:ivar:`/`:vartype:` render under Variables, and the xref reads the
    /// enclosing class scope: `py:class=\"C\"` (probe ivar_vartype).
    #[test]
    fn variables_fields_read_the_class_ref_context() {
        let pf = pf(".. py:class:: C\n\n   :ivar x: doc\n   :vartype x: int\n");
        assert!(
            pf.contains(concat!(
                "        <desc_content>\n",
                "            <field_list>\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Variables\n",
                "                    <field_body>\n",
                "                        <paragraph>\n",
                "                            <literal_strong>\n",
                "                                x\n",
                "                             (\n",
                "                            <pending_xref py:class=\"C\" py:module=\"True\" refdomain=\"py\" refexplicit=\"0\" refspecific=\"1\" reftarget=\"int\" reftype=\"class\">\n",
                "                                <literal_emphasis>\n",
                "                                    int\n",
                "                            )\n",
                "                             -- \n",
                "                            doc\n",
            )),
            "{pf}"
        );
    }

    /// An unknown field name is capitalized and the field passed through
    /// untouched (probe unknown_field).
    #[test]
    fn an_unknown_field_is_capitalized_and_passed_through() {
        let pf = pf(".. py:function:: f()\n\n   :custom foo: bar\n");
        assert!(
            pf.contains(concat!(
                "        <desc_content>\n",
                "            <field_list>\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Custom foo\n",
                "                    <field_body>\n",
                "                        <paragraph>\n",
                "                            bar\n",
            )),
            "{pf}"
        );
    }

    /// A `:type x:` with no matching `:param x:` is consumed into the
    /// types map and never re-emitted — empty field_list (probe
    /// orphan_type).
    #[test]
    fn an_orphan_type_field_is_consumed_silently() {
        let pf = pf(".. py:function:: f(x)\n\n   :type x: int\n");
        assert!(
            pf.contains(concat!(
                "        <desc_content>\n",
                "            <field_list>\n",
            )),
            "{pf}"
        );
    }

    /// `:param:` and `:keyword:` are the same `parameter` group (probe
    /// param_keyword_group).
    #[test]
    fn param_and_keyword_share_one_parameters_group() {
        let pf = pf(".. py:function:: f(a, b)\n\n   :param a: pos\n   :keyword b: kw\n");
        assert!(
            pf.contains(concat!(
                "        <desc_content>\n",
                "            <field_list>\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Parameters\n",
                "                    <field_body>\n",
                "                        <bullet_list>\n",
                "                            <list_item>\n",
                "                                <paragraph>\n",
                "                                    <literal_strong>\n",
                "                                        a\n",
                "                                     -- \n",
                "                                    pos\n",
                "                            <list_item>\n",
                "                                <paragraph>\n",
                "                                    <literal_strong>\n",
                "                                        b\n",
                "                                     -- \n",
                "                                    kw\n",
            )),
            "{pf}"
        );
    }

    /// `~pkg.Cls` in a type field: full reftarget, short title in the
    /// `literal_emphasis` innernode (probe tilde_type).
    #[test]
    fn a_tilde_type_takes_the_short_title() {
        let pf = pf(".. py:function:: f(x)\n\n   :param x: thing\n   :type x: ~pkg.Cls\n");
        assert!(
            pf.contains(concat!(
                "        <desc_content>\n",
                "            <field_list>\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Parameters\n",
                "                    <field_body>\n",
                "                        <paragraph>\n",
                "                            <literal_strong>\n",
                "                                x\n",
                "                             (\n",
                "                            <pending_xref py:class=\"True\" py:module=\"True\" refdomain=\"py\" refexplicit=\"0\" refspecific=\"1\" reftarget=\"pkg.Cls\" reftype=\"class\">\n",
                "                                <literal_emphasis>\n",
                "                                    Cls\n",
                "                            )\n",
                "                             -- \n",
                "                            thing\n",
            )),
            "{pf}"
        );
    }

    /// `Literal[...]` suppression is sticky: everything after the
    /// `Literal` xref renders as plain `literal_emphasis` (probe
    /// literal_type).
    #[test]
    fn literal_bracket_types_suppress_inner_xrefs() {
        let pf = pf(".. py:function:: f(x)\n\n   :param x: thing\n   :type x: Literal[1, 2]\n");
        assert!(
            pf.contains(concat!(
                "        <desc_content>\n",
                "            <field_list>\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Parameters\n",
                "                    <field_body>\n",
                "                        <paragraph>\n",
                "                            <literal_strong>\n",
                "                                x\n",
                "                             (\n",
                "                            <pending_xref py:class=\"True\" py:module=\"True\" refdomain=\"py\" refexplicit=\"0\" refspecific=\"1\" reftarget=\"Literal\" reftype=\"class\">\n",
                "                                <literal_emphasis>\n",
                "                                    Literal\n",
                "                            <literal_emphasis>\n",
                "                                [\n",
                "                            <literal_emphasis>\n",
                "                                1\n",
                "                            <literal_emphasis>\n",
                "                                , \n",
                "                            <literal_emphasis>\n",
                "                                2\n",
                "                            <literal_emphasis>\n",
                "                                ]\n",
                "                            )\n",
                "                             -- \n",
                "                            thing\n",
            )),
            "{pf}"
        );
    }

    /// The rtype body role splits too, and the split contnode keeps BARE
    /// Text children — no literal_emphasis (probe rtype_or_split).
    #[test]
    fn rtype_bodies_split_and_keep_bare_text() {
        let pf = pf(".. py:function:: f()\n\n   :rtype: int or str\n");
        assert!(
            pf.contains(concat!(
                "        <desc_content>\n",
                "            <field_list>\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Return type\n",
                "                    <field_body>\n",
                "                        <paragraph>\n",
                "                            <pending_xref py:class=\"True\" py:module=\"True\" refdomain=\"py\" refexplicit=\"0\" refspecific=\"1\" reftarget=\"int\" reftype=\"class\">\n",
                "                                int\n",
                "                             or \n",
                "                            <pending_xref py:class=\"True\" py:module=\"True\" refdomain=\"py\" refexplicit=\"0\" refspecific=\"1\" reftarget=\"str\" reftype=\"class\">\n",
                "                                str\n",
            )),
            "{pf}"
        );
    }

    /// `types.pop()` semantics: a doubled `:param x:` gets the type on the
    /// first entry only (probe dup_param_typed).
    #[test]
    fn a_doubled_param_consumes_its_type_once() {
        let pf = pf(".. py:function:: f(x)\n\n   :param x: a\n   :param x: b\n   :type x: int\n");
        assert!(
            pf.contains(concat!(
                "        <desc_content>\n",
                "            <field_list>\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Parameters\n",
                "                    <field_body>\n",
                "                        <bullet_list>\n",
                "                            <list_item>\n",
                "                                <paragraph>\n",
                "                                    <literal_strong>\n",
                "                                        x\n",
                "                                     (\n",
                "                                    <pending_xref py:class=\"True\" py:module=\"True\" refdomain=\"py\" refexplicit=\"0\" refspecific=\"1\" reftarget=\"int\" reftype=\"class\">\n",
                "                                        <literal_emphasis>\n",
                "                                            int\n",
                "                                    )\n",
                "                                     -- \n",
                "                                    a\n",
                "                            <list_item>\n",
                "                                <paragraph>\n",
                "                                    <literal_strong>\n",
                "                                        x\n",
                "                                     -- \n",
                "                                    b\n",
            )),
            "{pf}"
        );
    }

    /// Field xrefs carry `py:module` from the ref_context (probe
    /// currentmodule_ctx).
    #[test]
    fn field_xrefs_read_the_module_ref_context() {
        let pf = pf(".. py:currentmodule:: curmod\n\n.. py:function:: f(x)\n\n   :param x: thing\n   :type x: str\n");
        assert!(
            pf.contains(concat!(
                "        <desc_content>\n",
                "            <field_list>\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Parameters\n",
                "                    <field_body>\n",
                "                        <paragraph>\n",
                "                            <literal_strong>\n",
                "                                x\n",
                "                             (\n",
                "                            <pending_xref py:class=\"True\" py:module=\"curmod\" refdomain=\"py\" refexplicit=\"0\" refspecific=\"1\" reftarget=\"str\" reftype=\"class\">\n",
                "                                <literal_emphasis>\n",
                "                                    str\n",
                "                            )\n",
                "                             -- \n",
                "                            thing\n",
            )),
            "{pf}"
        );
    }

    /// `returnvalue` is a plain `Field` (no PyXrefMixin): `int or str`
    /// stays one Text (probe returns_or_not_split).
    #[test]
    fn returns_bodies_are_never_split() {
        let pf = pf(".. py:function:: f()\n\n   :returns: int or str\n");
        assert!(
            pf.contains(concat!(
                "        <desc_content>\n",
                "            <field_list>\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Returns\n",
                "                    <field_body>\n",
                "                        <paragraph>\n",
                "                            int or str\n",
            )),
            "{pf}"
        );
    }

    /// Removing a `:meta:` field keeps its siblings transforming (probe
    /// meta_then_param).
    #[test]
    fn meta_removal_keeps_sibling_fields() {
        let pf = pf(".. py:function:: f(x)\n\n   :param x: kept\n   :meta private:\n");
        assert!(
            pf.contains(concat!(
                "        <desc_content>\n",
                "            <field_list>\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Parameters\n",
                "                    <field_body>\n",
                "                        <paragraph>\n",
                "                            <literal_strong>\n",
                "                                x\n",
                "                             -- \n",
                "                            kept\n",
            )),
            "{pf}"
        );
    }

    /// `:type:` with no argument mismatches `has_arg` and passes through
    /// renamed `Type` — but a lone-Text body is still type-linked
    /// (`docfields.py:409-432`; probe type_no_arg).
    #[test]
    fn a_bare_type_field_is_unknown_but_type_linked() {
        let pf = pf(".. py:function:: f(x)\n\n   :type: int\n");
        assert!(
            pf.contains(concat!(
                "        <desc_content>\n",
                "            <field_list>\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Type\n",
                "                    <field_body>\n",
                "                        <paragraph>\n",
                "                            <pending_xref py:class=\"True\" py:module=\"True\" refdomain=\"py\" refexplicit=\"0\" refspecific=\"1\" reftarget=\"int\" reftype=\"class\">\n",
                "                                int\n",
            )),
            "{pf}"
        );
    }

    /// `:returns foo:` mismatches `has_arg=False` and passes through as
    /// `Returns foo` (probe returns_with_arg_mismatch).
    #[test]
    fn an_arg_on_returns_demotes_it_to_unknown() {
        let pf = pf(".. py:function:: f()\n\n   :returns foo: x\n");
        assert!(
            pf.contains(concat!(
                "        <desc_content>\n",
                "            <field_list>\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Returns foo\n",
                "                    <field_body>\n",
                "                        <paragraph>\n",
                "                            x\n",
            )),
            "{pf}"
        );
    }

    /// `:raises:` with no argument is the unknown path — capitalized name,
    /// body untouched, no xref (probe raises_no_arg_mismatch).
    #[test]
    fn raises_without_arg_is_passed_through() {
        let pf = pf(".. py:function:: f()\n\n   :raises: something\n");
        assert!(
            pf.contains(concat!(
                "        <desc_content>\n",
                "            <field_list>\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Raises\n",
                "                    <field_body>\n",
                "                        <paragraph>\n",
                "                            something\n",
            )),
            "{pf}"
        );
    }

    /// TypedField adds ` -- ` only when the description has content (probe
    /// param_no_desc).
    #[test]
    fn an_empty_description_omits_the_dashes() {
        let pf = pf(".. py:function:: f(x)\n\n   :param x:\n");
        assert!(
            pf.contains(concat!(
                "        <desc_content>\n",
                "            <field_list>\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Parameters\n",
                "                    <field_body>\n",
                "                        <paragraph>\n",
                "                            <literal_strong>\n",
                "                                x\n",
            )),
            "{pf}"
        );
    }

    /// A multi-paragraph field body keeps its paragraphs, nested inside
    /// the item paragraph after ` -- ` (probe param_multipara).
    #[test]
    fn multi_paragraph_content_nests_in_the_item() {
        let pf = pf(".. py:function:: f(x)\n\n   :param x: first para\n\n      second para\n");
        assert!(
            pf.contains(concat!(
                "        <desc_content>\n",
                "            <field_list>\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Parameters\n",
                "                    <field_body>\n",
                "                        <paragraph>\n",
                "                            <literal_strong>\n",
                "                                x\n",
                "                             -- \n",
                "                            <paragraph>\n",
                "                                first para\n",
                "                            <paragraph>\n",
                "                                second para\n",
            )),
            "{pf}"
        );
    }

    /// A `:type x:` body with markup is not a single Text: the parsed
    /// nodes (here a role-generated pending_xref) are spliced verbatim
    /// between the parens (probe type_markup_body).
    #[test]
    fn a_markup_type_body_is_spliced_verbatim() {
        let pf = pf(".. py:function:: f(x)\n\n   :param x: thing\n   :type x: :class:`Foo`\n");
        assert!(
            pf.contains(concat!(
                "        <desc_content>\n",
                "            <field_list>\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Parameters\n",
                "                    <field_body>\n",
                "                        <paragraph>\n",
                "                            <literal_strong>\n",
                "                                x\n",
                "                             (\n",
                "                            <pending_xref py:class=\"True\" py:module=\"True\" refdoc=\"index\" refdomain=\"py\" refexplicit=\"0\" reftarget=\"Foo\" reftype=\"class\" refwarn=\"0\">\n",
                "                                <literal classes=\"xref py py-class\">\n",
                "                                    Foo\n",
                "                            )\n",
                "                             -- \n",
                "                            thing\n",
            )),
            "{pf}"
        );
    }

    /// Description inline markup rides along into the transformed entry
    /// (probe param_markup_desc).
    #[test]
    fn markup_in_descriptions_is_spliced() {
        let pf = pf(".. py:function:: f(x)\n\n   :param x: has *emphasis* here\n");
        assert!(
            pf.contains(concat!(
                "        <desc_content>\n",
                "            <field_list>\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Parameters\n",
                "                    <field_body>\n",
                "                        <paragraph>\n",
                "                            <literal_strong>\n",
                "                                x\n",
                "                             -- \n",
                "                            has \n",
                "                            <emphasis>\n",
                "                                emphasis\n",
                "                             here\n",
            )),
            "{pf}"
        );
    }

    /// Only immediate field_list children transform — and each one does,
    /// independently (probe two_field_lists).
    #[test]
    fn each_field_list_transforms_independently() {
        let pf = pf(
            ".. py:function:: f(x)\n\n   :param x: one\n\n   Body between.\n\n   :returns: two\n",
        );
        assert!(
            pf.contains(concat!(
                "        <desc_content>\n",
                "            <field_list>\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Parameters\n",
                "                    <field_body>\n",
                "                        <paragraph>\n",
                "                            <literal_strong>\n",
                "                                x\n",
                "                             -- \n",
                "                            one\n",
                "            <paragraph>\n",
                "                Body between.\n",
                "            <field_list>\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Returns\n",
                "                    <field_body>\n",
                "                        <paragraph>\n",
                "                            two\n",
            )),
            "{pf}"
        );
    }

    /// `:param pkg.Cls x:` rsplits into type + name (probe
    /// param_type_name_syntax_dotted).
    #[test]
    fn param_type_name_syntax_takes_the_last_token() {
        let pf = pf(".. py:function:: f(x)\n\n   :param pkg.Cls x: doc\n");
        assert!(
            pf.contains(concat!(
                "        <desc_content>\n",
                "            <field_list>\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Parameters\n",
                "                    <field_body>\n",
                "                        <paragraph>\n",
                "                            <literal_strong>\n",
                "                                x\n",
                "                             (\n",
                "                            <pending_xref py:class=\"True\" py:module=\"True\" refdomain=\"py\" refexplicit=\"0\" refspecific=\"1\" reftarget=\"pkg.Cls\" reftype=\"class\">\n",
                "                                <literal_emphasis>\n",
                "                                    pkg.Cls\n",
                "                            )\n",
                "                             -- \n",
                "                            doc\n",
            )),
            "{pf}"
        );
    }

    /// `~pkg.Cls` under unqualified names: the title-rewrite branch wins
    /// and NO condition nodes appear (probe FU2_tilde_precedence).
    #[test]
    fn the_title_rewrite_beats_unqualified_conditions() {
        let pf = pf_cfg(
            ".. py:function:: f(x)\n\n   :param x: thing\n   :type x: ~pkg.Cls\n",
            unqual(),
        );
        assert!(
            pf.contains(concat!(
                "        <desc_content>\n",
                "            <field_list>\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Parameters\n",
                "                    <field_body>\n",
                "                        <paragraph>\n",
                "                            <literal_strong>\n",
                "                                x\n",
                "                             (\n",
                "                            <pending_xref py:class=\"True\" py:module=\"True\" refdomain=\"py\" refexplicit=\"0\" refspecific=\"1\" reftarget=\"pkg.Cls\" reftype=\"class\">\n",
                "                                <literal_emphasis>\n",
                "                                    Cls\n",
                "                            )\n",
                "                             -- \n",
                "                            thing\n",
            )),
            "{pf}"
        );
    }

    /// Raises xrefs wrap in conditions too, with their `literal_strong`
    /// innernode inside each (probe FU2_raises).
    #[test]
    fn unqualified_wraps_raises_xrefs_too() {
        let pf = pf_cfg(
            ".. py:function:: f()\n\n   :raises pkg.Err: bad\n",
            unqual(),
        );
        assert!(
            pf.contains(concat!(
                "        <desc_content>\n",
                "            <field_list>\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Raises\n",
                "                    <field_body>\n",
                "                        <paragraph>\n",
                "                            <pending_xref py:class=\"True\" py:module=\"True\" refdomain=\"py\" refexplicit=\"0\" refspecific=\"1\" reftarget=\"pkg.Err\" reftype=\"exc\">\n",
                "                                <pending_xref_condition condition=\"resolved\">\n",
                "                                    <literal_strong>\n",
                "                                        Err\n",
                "                                <pending_xref_condition condition=\"*\">\n",
                "                                    <literal_strong>\n",
                "                                        pkg.Err\n",
                "                             -- \n",
                "                            bad\n",
            )),
            "{pf}"
        );
    }

    /// The rtype body path: `resolved` holds the default `emphasis`
    /// innernode, `*` keeps the original bare-Text contnode (probe
    /// FU2_rtype).
    #[test]
    fn unqualified_rtype_conditions_use_emphasis_innernode() {
        let pf = pf_cfg(".. py:function:: f()\n\n   :rtype: pkg.Cls\n", unqual());
        assert!(
            pf.contains(concat!(
                "        <desc_content>\n",
                "            <field_list>\n",
                "                <field>\n",
                "                    <field_name>\n",
                "                        Return type\n",
                "                    <field_body>\n",
                "                        <paragraph>\n",
                "                            <pending_xref py:class=\"True\" py:module=\"True\" refdomain=\"py\" refexplicit=\"0\" refspecific=\"1\" reftarget=\"pkg.Cls\" reftype=\"class\">\n",
                "                                <pending_xref_condition condition=\"resolved\">\n",
                "                                    <emphasis>\n",
                "                                        Cls\n",
                "                                <pending_xref_condition condition=\"*\">\n",
                "                                    pkg.Cls\n",
            )),
            "{pf}"
        );
    }
}

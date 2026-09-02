//! The `py` domain's *collection* half: object and module registration
//! with Sphinx's duplicate semantics — `PythonDomain.note_object` /
//! `note_module` / `clear_doc` / `merge_domaindata`
//! (`sphinx/domains/python/__init__.py:780-832` and `:744-757`). The
//! *resolution* half (`find_obj`/`resolve_xref`) is Task 10's.
//!
//! Registrations replay from the parse layer's records
//! ([`crate::rst::RegistryExport::py_objects`]/[`py_modules`]) inside
//! [`crate::env::std_domain::process_doc`]'s parse-time pass: in Sphinx
//! every one of these calls fires *while the directive runs*, so a
//! document's py duplicate warnings interleave with its std
//! description/term duplicates in document order — probe-verified against
//! sphinx 9.1.0 (a doc with an envvar duplicate at line 8, a py duplicate
//! at line 15 and a term duplicate at line 18 warns 8 → 15 → 18).
//! `PythonDomain` defines **no** `process_doc` hook at all, so the `py`
//! slot of `_DomainsContainer._process_doc` (dispatch order `c, changeset,
//! citation, cpp, index, js, math, py, rst, std`) contributes nothing of
//! its own.
//!
//! [`py_modules`]: crate::rst::RegistryExport::py_modules

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::env::std_domain::{source_path_of, DocumentIds, DocumentSource};
use crate::env::BuildEnvironment;
use crate::error::{BuildWarning, WarningType};

/// Sphinx's `ObjectEntry` (`__init__.py:60-65`), keyed by fullname in
/// [`PyDomainData::objects`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PyObjectEntry {
    pub docname: String,
    pub node_id: String,
    pub objtype: String,
    /// `:canonical:` alias registrations carry `true`; resolve-time
    /// disambiguation prefers non-aliased entries, and the duplicate rules
    /// below treat aliased entries as overridable.
    pub aliased: bool,
}

/// Sphinx's `ModuleEntry` (`__init__.py:67-73`), keyed by module name in
/// [`PyDomainData::modules`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PyModuleEntry {
    pub docname: String,
    pub node_id: String,
    pub synopsis: String,
    pub platform: String,
    pub deprecated: bool,
}

/// Python-domain (`py`) registries: `domaindata['py']['objects']` and
/// `['modules']`.
///
/// INSERTION-ORDERED, not a plain `BTreeMap`: Sphinx's fuzzy resolution
/// pass iterates the objects dict in **insertion order**
/// (`__init__.py:901-908`) and ambiguity takes the FIRST match, with the
/// candidates listed in match order — lexicographic iteration would
/// diverge on both the resolved target and the warning bytes whenever
/// registration order isn't alphabetical. Registration order is the
/// docname-ordered merge, record order within a document — and Python
/// dict assignment on an existing key keeps the original insertion slot,
/// so every overwrite here is **in place** (probe: a `:canonical:` alias
/// registered between two real definitions keeps its middle slot after
/// the second definition overwrites it).
///
/// The side `*_index` maps give O(log n) exact lookup; they always name
/// the entry's position in the paired `Vec` and carry no information of
/// their own.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct PyDomainData {
    /// fullname -> entry, in registration order.
    pub objects: Vec<(String, PyObjectEntry)>,
    /// fullname -> index into [`Self::objects`].
    pub objects_index: BTreeMap<String, usize>,
    /// modname -> entry, in registration order.
    pub modules: Vec<(String, PyModuleEntry)>,
    /// modname -> index into [`Self::modules`].
    pub modules_index: BTreeMap<String, usize>,
}

impl PyDomainData {
    /// `PythonDomain.note_object` (`__init__.py:780-813`). Returns the
    /// docname of the entry the caller must warn about — `Some` exactly
    /// when Sphinx's `logger.warning` fires. The aliased-vs-real matrix
    /// (each cell probe-verified against sphinx 9.1.0):
    ///
    /// | existing \ new | real                    | aliased                 |
    /// |----------------|-------------------------|-------------------------|
    /// | real           | warn + overwrite        | silent keep (no write)  |
    /// | aliased        | silent overwrite        | warn + overwrite        |
    ///
    /// Overwrites land **in place** (Python dict assignment keeps the
    /// original insertion slot).
    pub fn note_object(&mut self, name: &str, entry: PyObjectEntry) -> Option<String> {
        if let Some(&index) = self.objects_index.get(name) {
            let other = &self.objects[index].1;
            if !other.aliased && entry.aliased {
                // "The original definition is already registered" — the
                // alias is dropped without touching the real entry.
                return None;
            }
            // `other.aliased && !entry.aliased`: "The original definition
            // found. Override it!" — silently. Every other combination
            // falls through Sphinx's `else` and warns; both overwrite.
            let warn = (other.aliased == entry.aliased).then(|| other.docname.clone());
            self.objects[index].1 = entry;
            warn
        } else {
            self.objects_index
                .insert(name.to_string(), self.objects.len());
            self.objects.push((name.to_string(), entry));
            None
        }
    }

    /// `PythonDomain.note_module` (`__init__.py:819-832`) — an
    /// unconditional dict assignment: never warns, last value wins, an
    /// existing name keeps its insertion slot. (The duplicate-module
    /// *warning* comes from the `note_object(modname, 'module', ...)` call
    /// `PyModule.run` makes alongside this one.)
    pub fn note_module(&mut self, name: &str, entry: PyModuleEntry) {
        if let Some(&index) = self.modules_index.get(name) {
            self.modules[index].1 = entry;
        } else {
            self.modules_index
                .insert(name.to_string(), self.modules.len());
            self.modules.push((name.to_string(), entry));
        }
    }

    /// `PythonDomain.clear_doc` (`__init__.py:744-751`): drop every entry
    /// the document owns. Survivors keep their relative order — deleting
    /// from a Python dict never reorders what stays — and the indices are
    /// rebuilt to match.
    pub fn clear_doc(&mut self, docname: &str) {
        self.objects.retain(|(_, entry)| entry.docname != docname);
        self.modules.retain(|(_, entry)| entry.docname != docname);
        self.rebuild_indices();
    }

    /// `PythonDomain.merge_domaindata` (`__init__.py:753-757`): fold in
    /// `other`'s entries whose docname is in `docnames`, in `other`'s
    /// registration order. Like Sphinx's, this is a plain dict assignment
    /// per entry — no duplicate checks ("XXX check duplicates?"), an
    /// existing name is overwritten in place, a new one appended.
    pub fn merge(&mut self, other: &PyDomainData, docnames: &BTreeSet<String>) {
        for (name, entry) in &other.objects {
            if !docnames.contains(&entry.docname) {
                continue;
            }
            if let Some(&index) = self.objects_index.get(name) {
                self.objects[index].1 = entry.clone();
            } else {
                self.objects_index.insert(name.clone(), self.objects.len());
                self.objects.push((name.clone(), entry.clone()));
            }
        }
        for (name, entry) in &other.modules {
            if !docnames.contains(&entry.docname) {
                continue;
            }
            if let Some(&index) = self.modules_index.get(name) {
                self.modules[index].1 = entry.clone();
            } else {
                self.modules_index.insert(name.clone(), self.modules.len());
                self.modules.push((name.clone(), entry.clone()));
            }
        }
    }

    fn rebuild_indices(&mut self) {
        self.objects_index = self
            .objects
            .iter()
            .enumerate()
            .map(|(index, (name, _))| (name.clone(), index))
            .collect();
        self.modules_index = self
            .modules
            .iter()
            .enumerate()
            .map(|(index, (name, _))| (name.clone(), index))
            .collect();
    }
}

/// Replay one document's py registrations from the parse layer's records —
/// the `note_module` + `note_object` calls `PyModule.run` and
/// `PyObject.add_target_and_index` made while the directives ran, which
/// our parse layer records instead (the module scope they read lives in
/// the parser's ref_context, and a `:no-typesetting:` object registers
/// itself and then vanishes from the tree).
///
/// Duplicate warnings join `out` keyed by the registered node's position
/// in the doctree — the same document-order merge key
/// [`crate::env::std_domain::process_doc`] uses for its glossary and
/// description passes, because in Sphinx all three warning streams are
/// parse-time and interleave in document order (see the module comment).
///
/// `note_module` runs before `note_object` for the whole record stream
/// where Sphinx alternates per directive; the two registries are disjoint
/// maps and `note_module` never warns, so the difference is unobservable.
pub(crate) fn collect_registrations(
    env: &mut BuildEnvironment,
    doc: &DocumentSource<'_>,
    ids: &DocumentIds<'_>,
    warnings: &mut Vec<(usize, BuildWarning)>,
) {
    for record in &doc.registry.py_modules {
        env.py.note_module(
            &record.name,
            PyModuleEntry {
                docname: doc.docname.to_string(),
                node_id: record.node_id.clone(),
                synopsis: record.synopsis.clone(),
                platform: record.platform.clone(),
                deprecated: record.deprecated,
            },
        );
    }
    for record in &doc.registry.py_objects {
        let Some(other) = env.py.note_object(
            &record.fullname,
            PyObjectEntry {
                docname: doc.docname.to_string(),
                node_id: record.node_id.clone(),
                objtype: record.objtype.clone(),
                aliased: record.aliased,
            },
        ) else {
            continue;
        };
        let order = ids
            .get(&record.node_id)
            .map(|(order, _)| order)
            .unwrap_or(usize::MAX);
        warnings.push((
            order,
            // [PY §5]: plain `logger.warning` with no type/subtype — no
            // `[category]` suffix, and no objtype in the text (unlike the
            // std domain's `duplicate {objtype} description`).
            BuildWarning::new(
                source_path_of(doc, record.source),
                Some(record.lineno as usize),
                format!(
                    "duplicate object description of {}, other instance in {}, \
                     use :no-index: for one of them",
                    record.fullname, other
                ),
                WarningType::DuplicateLabel,
            )
            .with_category(None),
        ));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::env::std_domain;
    use crate::rst::{parse_rst_full, ParseOptions};
    use std::path::PathBuf;

    fn entry(docname: &str, node_id: &str, objtype: &str, aliased: bool) -> PyObjectEntry {
        PyObjectEntry {
            docname: docname.to_string(),
            node_id: node_id.to_string(),
            objtype: objtype.to_string(),
            aliased,
        }
    }

    fn module_entry(docname: &str, node_id: &str) -> PyModuleEntry {
        PyModuleEntry {
            docname: docname.to_string(),
            node_id: node_id.to_string(),
            synopsis: String::new(),
            platform: String::new(),
            deprecated: false,
        }
    }

    /// `(fullname, docname, aliased)` of every object, in iteration order.
    fn object_rows(data: &PyDomainData) -> Vec<(&str, &str, bool)> {
        data.objects
            .iter()
            .map(|(name, e)| (name.as_str(), e.docname.as_str(), e.aliased))
            .collect()
    }

    fn assert_indices_consistent(data: &PyDomainData) {
        assert_eq!(data.objects_index.len(), data.objects.len());
        for (name, &index) in &data.objects_index {
            assert_eq!(&data.objects[index].0, name, "objects_index[{name}]");
        }
        assert_eq!(data.modules_index.len(), data.modules.len());
        for (name, &index) in &data.modules_index {
            assert_eq!(&data.modules[index].0, name, "modules_index[{name}]");
        }
    }

    // ---- note_object matrix ([PY §5], each cell probe-verified) --------

    #[test]
    fn real_over_real_warns_and_the_last_definition_wins_in_place() {
        let mut py = PyDomainData::default();
        py.note_object("other", entry("a", "other", "function", false));
        assert_eq!(
            py.note_object("dup", entry("a", "dup", "function", false)),
            None
        );
        assert_eq!(
            py.note_object("dup", entry("b", "id0", "function", false)),
            Some("a".to_string()),
            "the second real definition warns naming the first's docname"
        );
        assert_eq!(
            object_rows(&py),
            vec![("other", "a", false), ("dup", "b", false)],
            "the overwrite lands in the original insertion slot"
        );
        assert_eq!(py.objects[py.objects_index["dup"]].1.node_id, "id0");
        assert_indices_consistent(&py);
    }

    #[test]
    fn an_alias_never_replaces_a_real_definition_and_stays_silent() {
        let mut py = PyDomainData::default();
        py.note_object("name", entry("a", "name", "function", false));
        assert_eq!(
            py.note_object("name", entry("b", "alias-id", "function", true)),
            None
        );
        assert_eq!(
            py.objects[py.objects_index["name"]].1,
            entry("a", "name", "function", false),
            "the real entry is untouched"
        );
    }

    #[test]
    fn a_real_definition_silently_overrides_an_alias_in_place() {
        let mut py = PyDomainData::default();
        py.note_object("first", entry("a", "first", "function", false));
        py.note_object("name", entry("a", "alias-id", "function", true));
        py.note_object("last", entry("a", "last", "function", false));
        assert_eq!(
            py.note_object("name", entry("b", "name", "function", false)),
            None,
            "\"The original definition found. Override it!\" — no warning"
        );
        assert_eq!(
            object_rows(&py),
            vec![
                ("first", "a", false),
                ("name", "b", false),
                ("last", "a", false)
            ],
            "the override keeps the alias's insertion slot"
        );
    }

    /// The fourth cell, probe-verified against sphinx 9.1.0: two
    /// `:canonical: shared.alias` registrations warn (`duplicate object
    /// description of shared.alias, other instance in index, use
    /// :no-index: for one of them`) and the later alias wins, keeping the
    /// original slot — `note_object` falls through to the warn+overwrite
    /// `else` whenever the aliased flags are equal.
    #[test]
    fn an_alias_over_an_alias_warns_and_overwrites_in_place() {
        let mut py = PyDomainData::default();
        py.note_object("new_a", entry("index", "new_a", "function", false));
        py.note_object("shared.alias", entry("index", "new_a", "function", true));
        py.note_object("new_b", entry("index", "new_b", "function", false));
        assert_eq!(
            py.note_object("shared.alias", entry("index", "new_b", "function", true)),
            Some("index".to_string())
        );
        assert_eq!(
            object_rows(&py),
            vec![
                ("new_a", "index", false),
                ("shared.alias", "index", true),
                ("new_b", "index", false),
            ]
        );
        assert_eq!(
            py.objects[py.objects_index["shared.alias"]].1.node_id,
            "new_b"
        );
    }

    // ---- ordering, clear_doc, merge, note_module -----------------------

    /// The registration-order contract T10's fuzzy pass builds on:
    /// iteration yields entries in the order they were first registered,
    /// never alphabetized.
    #[test]
    fn iteration_preserves_registration_order_not_lexicographic_order() {
        let mut py = PyDomainData::default();
        py.note_object("zeta.same", entry("a", "zeta.same", "function", false));
        py.note_object("alpha.same", entry("a", "alpha.same", "function", false));
        assert_eq!(
            py.objects
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>(),
            vec!["zeta.same", "alpha.same"]
        );
        assert_eq!(py.objects_index["zeta.same"], 0);
        assert_eq!(py.objects_index["alpha.same"], 1);
    }

    #[test]
    fn clear_doc_preserves_the_relative_order_of_survivors() {
        let mut py = PyDomainData::default();
        py.note_object("one", entry("a", "one", "function", false));
        py.note_object("two", entry("b", "two", "function", false));
        py.note_object("three", entry("a", "three", "class", false));
        py.note_object("four", entry("b", "four", "function", false));
        py.note_module("amod", module_entry("a", "module-amod"));
        py.note_module("bmod", module_entry("b", "module-bmod"));

        py.clear_doc("a");

        assert_eq!(
            object_rows(&py),
            vec![("two", "b", false), ("four", "b", false)]
        );
        assert_eq!(
            py.modules
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>(),
            vec!["bmod"]
        );
        assert_indices_consistent(&py);

        py.clear_doc("b");
        assert!(py.objects.is_empty() && py.modules.is_empty());
        assert!(py.objects_index.is_empty() && py.modules_index.is_empty());
    }

    #[test]
    fn merge_folds_only_the_named_docnames_in_registration_order() {
        let mut ours = PyDomainData::default();
        ours.note_object("kept", entry("a", "kept", "function", false));
        ours.note_object("both", entry("a", "both", "function", false));

        let mut theirs = PyDomainData::default();
        theirs.note_object("zeta", entry("b", "zeta", "function", false));
        theirs.note_object("both", entry("b", "id0", "function", false));
        theirs.note_object("skipped", entry("c", "skipped", "function", false));
        theirs.note_module("bmod", module_entry("b", "module-bmod"));
        theirs.note_module("cmod", module_entry("c", "module-cmod"));

        ours.merge(&theirs, &BTreeSet::from(["b".to_string()]));

        assert_eq!(
            object_rows(&ours),
            vec![
                ("kept", "a", false),
                // Dict assignment: the existing key keeps its slot, the
                // value is theirs. No duplicate warning — sphinx's
                // merge_domaindata performs none.
                ("both", "b", false),
                ("zeta", "b", false),
            ]
        );
        assert_eq!(
            ours.modules
                .iter()
                .map(|(n, _)| n.as_str())
                .collect::<Vec<_>>(),
            vec!["bmod"]
        );
        assert_indices_consistent(&ours);
    }

    #[test]
    fn note_module_never_warns_and_the_last_entry_wins_in_place() {
        let mut py = PyDomainData::default();
        py.note_module("mod", module_entry("a", "module-mod"));
        py.note_module("other", module_entry("a", "module-other"));
        py.note_module(
            "mod",
            PyModuleEntry {
                docname: "b".to_string(),
                node_id: "module-0".to_string(),
                synopsis: "S".to_string(),
                platform: "P".to_string(),
                deprecated: true,
            },
        );
        assert_eq!(
            py.modules
                .iter()
                .map(|(n, e)| (n.as_str(), e.docname.as_str()))
                .collect::<Vec<_>>(),
            vec![("mod", "b"), ("other", "a")]
        );
        assert!(py.modules[py.modules_index["mod"]].1.deprecated);
    }

    // ---- the replay through std_domain::process_doc --------------------

    fn parse(source: &str, docname: &str) -> crate::rst::ParseOutput {
        parse_rst_full(
            source,
            &ParseOptions {
                source_path: format!("<{docname}>"),
                sphinx: true,
                docname: docname.to_string(),
                found_docs: None,
                exclude_patterns: Vec::new(),
                py: Default::default(),
            },
        )
    }

    /// Fold sources into a fresh environment through the real per-document
    /// orchestration ([`std_domain::process_doc`], which replays the py
    /// records) and return it with the warnings.
    fn read(sources: &[(&str, &str)]) -> (BuildEnvironment, Vec<BuildWarning>) {
        let mut env = BuildEnvironment::default();
        let mut warnings = Vec::new();
        let doc2path = |docname: &str| PathBuf::from(format!("/src/{docname}.rst"));
        for (docname, source) in sources {
            let parsed = parse(source, docname);
            let path = PathBuf::from(format!("/src/{docname}.rst"));
            std_domain::process_doc(
                &mut env,
                &DocumentSource {
                    docname,
                    doctree: &parsed.doctree,
                    registry: &parsed.registry,
                    path: &path,
                },
                &doc2path,
                &mut warnings,
            );
        }
        (env, warnings)
    }

    /// [PY §5] `duplicate_functions` probe: the second definition's id
    /// falls back to `id0`, the warning names the document's own docname
    /// with the `:no-index:` hint and no category suffix, and the objects
    /// table keeps the LAST definition in the FIRST definition's slot.
    #[test]
    fn a_py_object_defined_twice_in_one_document_warns_with_the_sphinx_bytes() {
        let (env, warnings) = read(&[(
            "index",
            ".. py:function:: dup()\n\n.. py:function:: dup()\n",
        )]);
        assert_eq!(
            warnings.iter().map(|w| w.render()).collect::<Vec<_>>(),
            vec![
                "<index>:3: WARNING: duplicate object description of dup, \
                 other instance in index, use :no-index: for one of them"
            ]
        );
        assert_eq!(
            object_rows(&env.py),
            vec![("dup", "index", false)],
            "last definition wins"
        );
        assert_eq!(env.py.objects[0].1.node_id, "id0");
    }

    /// [PY §5] `duplicate_modules` probe: the module duplicate warns via
    /// its `note_object` half (line = the directive's own), while
    /// `note_module` silently records the second entry — both tables end
    /// on `module-0`.
    #[test]
    fn a_module_defined_twice_warns_once_and_both_tables_keep_the_second() {
        let (env, warnings) =
            read(&[("index", ".. py:module:: dupmod\n\n.. py:module:: dupmod\n")]);
        assert_eq!(
            warnings.iter().map(|w| w.render()).collect::<Vec<_>>(),
            vec![
                "<index>:3: WARNING: duplicate object description of dupmod, \
                 other instance in index, use :no-index: for one of them"
            ]
        );
        assert_eq!(
            env.py.objects[env.py.objects_index["dupmod"]].1,
            entry("index", "module-0", "module", false)
        );
        assert_eq!(
            env.py.modules[env.py.modules_index["dupmod"]].1,
            module_entry("index", "module-0")
        );
    }

    /// Cross-document duplicate: the warning fires from the second
    /// document, naming the first — byte-checked against a sphinx 9.1.0
    /// dummy build of this pair.
    #[test]
    fn a_py_duplicate_across_documents_names_the_other_docname() {
        let (env, warnings) = read(&[
            ("a", ".. py:function:: dup()\n"),
            ("b", "B\n=\n\n.. py:function:: dup()\n"),
        ]);
        assert_eq!(
            warnings.iter().map(|w| w.render()).collect::<Vec<_>>(),
            vec![
                "<b>:4: WARNING: duplicate object description of dup, \
                 other instance in a, use :no-index: for one of them"
            ]
        );
        assert_eq!(object_rows(&env.py), vec![("dup", "b", false)]);
    }

    /// §6 `canonical_function` probe: `:canonical:` registers a second
    /// entry under the canonical name with `aliased=True` and the same
    /// node id.
    #[test]
    fn canonical_registers_an_aliased_entry_with_the_same_node_id() {
        let (env, warnings) = read(&[(
            "index",
            ".. py:function:: new_name()\n   :canonical: old.name\n",
        )]);
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(
            env.py.objects,
            vec![
                (
                    "new_name".to_string(),
                    entry("index", "new_name", "function", false)
                ),
                (
                    "old.name".to_string(),
                    entry("index", "new_name", "function", true)
                ),
            ]
        );
    }

    /// Cross-domain interleaving, probe-verified against sphinx 9.1.0 on
    /// this exact document built twice: an envvar duplicate (line 8), a py
    /// duplicate (line 15) and a term duplicate (line 18) warn in
    /// DOCUMENT order — all three registrations are parse-time in Sphinx,
    /// so no domain's stream comes out grouped.
    #[test]
    fn py_duplicate_warnings_interleave_with_std_s_in_document_order() {
        let document = "Probe\n=====\n\n\
                        .. envvar:: STDDUP\n\n\
                        .. py:function:: pydup()\n\n\
                        .. envvar:: STDDUP\n\n\
                        .. glossary::\n\n   \
                        gterm\n      First.\n\n\
                        .. py:function:: pydup()\n\n\
                        .. glossary::\n\n   \
                        gterm\n      Second.\n";
        let (_, warnings) = read(&[("index", document)]);
        assert_eq!(
            warnings
                .iter()
                .map(|warning| (warning.line, warning.message.as_str()))
                .collect::<Vec<_>>(),
            vec![
                (
                    Some(8),
                    "duplicate envvar description of STDDUP, other instance in index"
                ),
                (
                    Some(15),
                    "duplicate object description of pydup, other instance in index, \
                     use :no-index: for one of them"
                ),
                (
                    Some(18),
                    "duplicate term description of gterm, other instance in index"
                ),
            ],
            "{warnings:?}"
        );
    }

    /// The std domain must not see any of this: a py-only document adds
    /// nothing to `env.std`, and a std-only document adds nothing to
    /// `env.py` — the guard for "no std behavior change" alongside the
    /// wiring this task added to `process_doc`.
    #[test]
    fn py_and_std_registrations_stay_in_their_own_registries() {
        let (env, warnings) = read(&[("index", ".. py:function:: func()\n\n.. envvar:: HOME\n")]);
        assert!(warnings.is_empty(), "{warnings:?}");
        assert_eq!(object_rows(&env.py), vec![("func", "index", false)]);
        assert_eq!(
            env.std.objects.keys().collect::<Vec<_>>(),
            vec![&("envvar".to_string(), "HOME".to_string())]
        );
        assert!(env
            .std
            .objects
            .keys()
            .all(|(objtype, _)| objtype != "function"));
    }
}

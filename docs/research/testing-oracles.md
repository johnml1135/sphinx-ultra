# HTML Testing Oracle Research

## Finding

There is no single exhaustive upstream golden corpus for Sphinx HTML output.
Upstream behavior is distributed across pytest cases, fixture projects,
builder-specific tests, extension tests, parser tests, and generated artifacts.
Those sources exercise different phases and do not provide one authoritative,
machine-readable set of input trees, output trees, provenance, and exclusions.

The repository already contains strong local evidence, but it is intentionally
split by layer:

| Existing source | Evidence | What it covers |
|---|---:|---|
| tests/fixtures/doctree_differential.json and tools/gen_doctree_fixture.py | 735 cases | Docutils parse-layer behavior |
| tests/fixtures/sphinx_doctree_differential.json and tools/gen_sphinx_fixture.py | 489 cases | Sphinx 9.1.0 read-phase snippets |
| tests/fixtures/env_differential.json and tools/gen_env_fixture.py | 29 projects / 84 documents | Build-environment state, resolved doctrees, warnings, toctrees, domains, and indexes |
| tests/fixtures/{basic,basic_missing_ref,deps_image,intersphinx,literalinclude,toctree_forms,toctree_glob} | 7 projects | Checked-in HTML-ish end-to-end fixture projects |
| tools/gen_inventory_fixture.py and tests/fixtures/inventories/manifest.json | 4 Sphinx-built projects | Real HTML-builder objects.inv records; the committed inventory directory also has handcrafted valid and malformed byte cases |
| tests/fixtures/pattern_differential.json and tools/gen_pattern_fixture.py | 881 cases | Parser-only Sphinx pattern semantics |
| Local sibling sphinx-needs doc_test projects | 142 projects | Extension-originated pytest expectations and project-level behavior |

The counts above are floors for the future HTML corpus. The source-set
discovery check must also prove exact set equality for every case currently
present, so a silently truncated or renamed source cannot pass merely because
the floor still passes. The existing status table in
docs/IMPLEMENTATION_STATUS.md is the repository's corroborating inventory for
the parser, Sphinx read-phase, environment, pattern, and inventory counts.

## Oracle policy

Sphinx snippets should be promoted through real HTML builds whenever the input
can be materialized as a project. A direct read-phase harness is useful for
isolating parser behavior, but it is not enough to pin page templates, copied
assets, search data, inventory emission, relative links, warnings, or builder
finish behavior. The HTML oracle should therefore materialize each eligible
snippet or project, run the real Sphinx HTML builder, and retain the complete
logical output tree. Cases that genuinely require a different builder or an
external service stay in the ledger with a reason instead of being silently
dropped.

Every upstream case must occur exactly once in one machine-readable ledger.
The ledger's status is one of these values and no other value is accepted:

| Status | Meaning |
|---|---|
| active | The reference and Ultra build are runnable and comparable. |
| alias | The case has a unique source identity but reuses one canonical case's input and reference. |
| excluded-network | The upstream case needs network access; it is recorded but never run by the offline harness. |
| excluded-plantuml | The case requires PlantUML or another unavailable diagram renderer. |
| excluded-external-test-fixture | The test depends on data outside the checked-in source set. |
| unsupported-builder | The case targets a builder other than the HTML builder or cannot produce an HTML contract. |
| reference-crash | The pinned reference process crashes; its failure, traceback summary, and provenance are retained as reference evidence. |

expectation is separate from status and is one of match, expected-failure, or
reference-only. This allows the harness to record a known Ultra limitation
without treating it as an upstream omission. In particular, local
sphinx-needs cases retain their originating pytest node ID, assertion or
regression expectation, and source revision. A case containing an unsupported
sphinx-needs directive is reference-only or expected-failure; it is never
blindly passed to native Ultra as if the extension were implemented.

## Pinned reference profiles

The core profile is exactly Sphinx 9.1.0 with Docutils 0.22.4 and has its own
committed lock. The local-needs profile is exactly sphinx-needs 8.5.0 at
commit 58bcb59d861da95f2aca79f343e8bae6ec5c1250, with subtree tree
958172a89defcec69704f6b9d61e482e7c4e8409, and uses Sphinx 9.1.0 with
Docutils 0.21.2 from its own committed lock. The generator must verify all
four version values plus the sphinx-needs commit and subtree tree before it
generates anything. It must fail closed if the local checkout is missing or
does not match those pins.

The two profiles are independent. A core fixture must never be regenerated
under the local-needs dependency graph, and a local-needs case must retain its
extension provenance even when its source happens to be plain RST. The ledger
records the profile, lock digest, source revision, originating file, pytest
node ID, builder, and expectation for every case.

## Implications for the harness

The new oracle needs to add a complete build contract around the existing
layered fixtures:

1. tools/gen_html_oracle.py must discover every declared source set, verify
   the exact source-set equality and count floors, build eligible inputs with
   the pinned profile, and write the fixture atomically and deterministically.
2. tools/html_oracle_cases.toml must be the checked-in discovery and policy
   schema. It must name the seven statuses, the three expectation modes, the
   source roots, the exact profile locks, the normalization boundaries, the
   output policies, and the licensing records.
3. tests/fixtures/html_oracle/index.json must be the generated ledger. It must
   contain one record per upstream case, sorted by source set and case ID, with
   no duplicate IDs and no unreferenced input, reference, or blob.
4. tests/fixtures/html_oracle/inputs must retain the materialized source trees,
   refs must retain structured and textual reference records, and blobs must
   deduplicate static assets by content hash without deleting any logical
   output-tree entry. NOTICE.md must identify source, revision, license, and
   redistribution scope for every imported corpus.
5. Rust comparison must use the actual CARGO_BIN_EXE_sphinx-ultra binary.
   Cache directories must be outside the output directory. The same case must
   be built under two different absolute source and output roots, then
   compared after only the declared path-field and CRLF normalizations.
6. JSON and searchindex.js must be compared through explicit canonical
   structured policies. objects.inv must be decoded and compared as semantic
   header and record data. Opaque files such as images, fonts, and other
   binary assets must be compared byte-for-byte.
7. The default suite must stay green while Ultra is incomplete. One ignored,
   opt-in exhaustive differential test must run every runnable case, aggregate
   every mismatch, cap each diagnostic, and exit nonzero until all active cases
   pass. Excluded, alias, expected-failure, and reference-only cases must be
   reported according to their ledger records rather than hidden.

These constraints make the HTML oracle a compatibility ledger and reproducible
artifact set, not a sample-only smoke corpus.

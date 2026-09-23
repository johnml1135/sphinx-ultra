# HTML Oracle Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

## Goal

Create one comprehensive first PR that discovers the complete in-scope corpus, builds committed references with real Sphinx HTML builds, and runs the actual sphinx-ultra CLI against every runnable case. The default cargo test suite stays green. One ignored exhaustive test runs the whole runnable set, aggregates every mismatch, writes target/html-oracle/report.md and target/html-oracle/report.json, and exits nonzero while Ultra is incomplete.

This is one first PR because the acceptance contract is end-to-end: committed references, deterministic generation, a runnable comparator, and a complete known-red report. It is divided into bounded commits so each schema, discovery, runner, storage, comparator, process, and report task is independently reviewable.

No production or test implementation is written until this plan is executed.

## Architecture

Reference generation is Python and has three layers:

1. Discovery reads the existing JSON fixtures and project directories, materializes each document-shaped input as an HTML project, and records provenance.
2. A child runner installs a socket guard and calls sphinx.cmd.build.main(argv) with the pinned Sphinx environment.
3. The generator captures output, warnings, status, hashes, and static assets into an atomic fixture tree.

Rust tests have three layers:

1. tests/support/html_oracle.rs loads and validates index.json, reconstructs logical trees, applies the fixed path policy, and compares outputs.
2. tests/support/diagnostics.rs executes child processes with bounded pipes and deadlines.
3. tests/html_differential.rs invokes env!("CARGO_BIN_EXE_sphinx-ultra") for every runnable ledger case, then writes the aggregate report.

Required implementation paths:

~~~text
tools/gen_html_oracle.py
tools/html_oracle_runner.py
tools/html_oracle_cases.toml
tools/oracle_profiles/core/pyproject.toml
tools/oracle_profiles/core/uv.lock
tools/oracle_profiles/local_needs/pyproject.toml
tools/oracle_profiles/local_needs/uv.lock
tools/test_gen_html_oracle.py
tests/html_differential.rs
tests/support/html_oracle.rs
tests/support/diagnostics.rs
tests/fixtures/html_oracle/index.json
tests/fixtures/html_oracle/inputs
tests/fixtures/html_oracle/refs
tests/fixtures/html_oracle/blobs
tests/fixtures/html_oracle/NOTICE.md
~~~

Each case directory is tests/fixtures/html_oracle/inputs/profile/source_set/case_id and tests/fixtures/html_oracle/refs/profile/source_set/case_id. Sanitize case IDs to A-Za-z0-9_.-, limit the sanitized portion to 48 characters, and append a hyphen plus the first eight hex characters of sha256 of the unsanitized identifier on truncation or collision.

### Corpus and scope

The verified source counts are:

| Source set | Profile | Source | Contract |
| --- | --- | --- | --- |
| docutils_snippets | core | tests/fixtures/doctree_differential.json | at least 735 one-document projects |
| sphinx_read_snippets | core | tests/fixtures/sphinx_doctree_differential.json | at least 489 one-document projects |
| environment_projects | core | tests/fixtures/env_differential.json | exactly 29 projects and at least 84 documents |
| html_projects | core | seven named directories in tests/fixtures | exactly 7 projects |
| inventory_projects | core | SPHINX_PROJECTS in tools/gen_inventory_fixture.py | exactly 4 Sphinx-built projects |
| sphinx_needs_doc_tests | local_needs | packages/sphinx-needs/tests/doc_test | exactly 142 directories containing conf.py |

The 881 records in tests/fixtures/pattern_differential.json are parser-only pattern cases, not documents, and are outside the HTML oracle. They are not materialized or ledgered. The handcrafted .inv files in tests/fixtures/inventories remain parser fixtures; only the four Sphinx-built projects from SPHINX_PROJECTS become cases.

The seven existing HTML-ish directories are basic, basic_missing_ref, deps_image, intersphinx, literalinclude, toctree_forms, and toctree_glob. They are read-only inputs. The Docutils, Sphinx read-phase, and environment JSON fixtures are also read-only.

Each Docutils or Sphinx snippet becomes an index.rst plus this exact conf.py:

~~~python
project = "html-oracle"
extensions = []
master_doc = "index"
exclude_patterns = ["_build"]
smartquotes = False
keep_warnings = True
~~~

Environment projects are reconstructed with all documents and built with the real html builder. Existing HTML projects keep their own conf.py. Inventory projects use the real html builder and the conventions in tools/gen_inventory_fixture.py.

Local-needs discovery is restricted to needs-root/packages/sphinx-needs/tests/doc_test. The checked sibling checkout has source at packages/sphinx-needs/src/sphinx_needs and exactly 142 direct doc_test directories with conf.py. One case is one project. The ledger records every statically found pytest node ID referencing that project and sets variants_not_captured when a matching test scope contains confoverrides or a non-html builder. The scanner reads AST and source paths only; it does not import tests or extract assertions.

### Status model

Use exactly these statuses:

| Status | Reference handling | Ultra handling |
| --- | --- | --- |
| built | exit code 0; capture complete tree | run and compare complete tree |
| build-error | nonzero exit without a Python traceback; capture warnings and partial tree | run and require build-error class, warnings, and files |
| reference-crash | traceback; capture record and do not run Ultra | not scheduled |
| excluded-network | static remote fetch requirement | not scheduled |
| excluded-plantuml | static conf.py load of sphinxcontrib.plantuml | not scheduled |

Excluded records have null exit_code, empty warnings, a required excluded_reason, and no files. A traceback is the exact marker Traceback (most recent call last): in combined child output. A nonzero result without that marker is build-error. A build-error is the supported cannot-generate reference case and is still run through Ultra.

The local-needs cases are not downgraded because Ultra lacks extension directives. They are built with their own conf.py, committed, and sent to Ultra whenever their reference status is built or build-error.

### Fixed per-path comparison policy

TOML has no normalization settings. Both Python and Rust implement this exact table:

| Path or stream | Policy |
| --- | --- |
| warnings stream | UTF-8 with replacement, CRLF to LF, replace the absolute source root with <SRCDIR>, exact text comparison |
| searchindex.js | CRLF to LF, remove Search.setIndex( and the final );, parse JSON, compare values with object-key order ignored and array order preserved |
| objects.inv | exact four-line header, zlib-decode records, parse name/domain-role/priority/URI/display name, sort by all five fields, compare header plus canonical record list |
| *.buildinfo | exact bytes |
| *.html, _sources/**, *.css, *.js except searchindex.js, *.json, *.xml, *.txt | CRLF to LF, then exact bytes |
| every other path, including images | exact bytes |

There is no HTML DOM rewrite or field-specific normalizer. Source-root replacement is permitted only in warnings. The generator fails if an absolute source, build, or cache root occurs in captured reference bytes outside warnings.

### Storage and determinism

HTML, searchindex.js, objects.inv, .buildinfo, _sources, and other non-static outputs are stored under refs. Every logical path beginning _static/ or _images/ is stored under blobs using its content SHA-256 as the filename. The ledger retains every logical path, so deduplicating bytes never removes logical output.

For sorted relative path and content digest pairs, use:

~~~python
def canonical_hash(entries: list[tuple[str, str]]) -> str:
    payload = "".join(
        f"{path}\0{digest}\n"
        for path, digest in sorted(entries)
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
~~~

Use this formula for input_sha256 and tree_sha256. Generation uses a process pool controlled by -j N, default os.cpu_count(), but collects and serializes results by profile, source_set, case_id, then logical path.

Each profile owns a subtree. Generate into a complete sibling staging tree, validate it, rename the old tree to tests/fixtures/html_oracle.old, rename staging to tests/fixtures/html_oracle, and remove .old only after the new tree is visible. A failure before the second rename preserves the old tree. Run one failure-injection test.

## Tech Stack

Python 3.12 stdlib, Sphinx 9.1.0, Docutils 0.22.4 for core, Docutils 0.21.2 for local_needs, sphinx-needs 8.5.0 loaded by explicit PYTHONPATH, and uv locked projects. Lock creation may resolve packages through the package index; generation uses uv run --locked and the child socket guard.

Rust uses serde, serde_json, sha2, tempfile, std::process, std::thread, std::sync, and std::time. Add sha2 as a direct Cargo dependency for exact SHA-256. Do not add a runtime network service or OS sandbox.

## Full index.json schema

The generator writes sorted UTF-8 JSON at tests/fixtures/html_oracle/index.json. These Python types are the complete writer schema:

~~~python
from typing import Literal, TypedDict

CaseStatus = Literal[
    "built",
    "build-error",
    "reference-crash",
    "excluded-network",
    "excluded-plantuml",
]
FileStorage = Literal["ref", "blob"]

class ProfileRecord(TypedDict):
    sphinx: str
    docutils: str
    needs_version: str | None
    needs_commit: str | None
    needs_tree: str | None
    lock_path: str
    lock_sha256: str

class OriginRecord(TypedDict):
    source_set: str
    origin_path: str
    pytest_node_ids: list[str]
    variants_not_captured: bool

class FileRecord(TypedDict):
    logical_path: str
    storage: FileStorage
    storage_path: str
    sha256: str
    size: int

class CaseRecord(TypedDict):
    profile: str
    source_set: str
    case_id: str
    status: CaseStatus
    exit_code: int | None
    warnings: str
    excluded_reason: str | None
    origin: OriginRecord
    input_sha256: str
    tree_sha256: str
    files: list[FileRecord]

class IndexDocument(TypedDict):
    schema_version: int
    generator: str
    profiles: dict[str, ProfileRecord]
    cases: list[CaseRecord]
~~~

The complete JSON shape is:

~~~json
{
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
      "lock_sha256": "0000000000000000000000000000000000000000000000000000000000000000"
    },
    "local_needs": {
      "sphinx": "9.1.0",
      "docutils": "0.21.2",
      "needs_version": "8.5.0",
      "needs_commit": "58bcb59d861da95f2aca79f343e8bae6ec5c1250",
      "needs_tree": "958172a89defcec69704f6b9d61e482e7c4e8409",
      "lock_path": "tools/oracle_profiles/local_needs/uv.lock",
      "lock_sha256": "0000000000000000000000000000000000000000000000000000000000000000"
    }
  },
  "cases": [
    {
      "profile": "core",
      "source_set": "docutils_snippets",
      "case_id": "docutils-0001",
      "status": "built",
      "exit_code": 0,
      "warnings": "",
      "excluded_reason": null,
      "origin": {
        "source_set": "docutils_snippets",
        "origin_path": "tests/fixtures/doctree_differential.json[0]",
        "pytest_node_ids": [],
        "variants_not_captured": false
      },
      "input_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
      "tree_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
      "files": [
        {
          "logical_path": "index.html",
          "storage": "ref",
          "storage_path": "refs/core/docutils_snippets/docutils-0001/index.html",
          "sha256": "0000000000000000000000000000000000000000000000000000000000000000",
          "size": 1032
        }
      ]
    }
  ]
}
~~~

The Rust serde model is:

~~~rust
use serde::Deserialize;
use std::collections::BTreeMap;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IndexDocument {
    pub schema_version: u32,
    pub generator: String,
    pub profiles: BTreeMap<String, ProfileRecord>,
    pub cases: Vec<CaseRecord>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ProfileRecord {
    pub sphinx: String,
    pub docutils: String,
    pub needs_version: Option<String>,
    pub needs_commit: Option<String>,
    pub needs_tree: Option<String>,
    pub lock_path: String,
    pub lock_sha256: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OriginRecord {
    pub source_set: String,
    pub origin_path: String,
    pub pytest_node_ids: Vec<String>,
    pub variants_not_captured: bool,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum FileStorage {
    Ref,
    Blob,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FileRecord {
    pub logical_path: String,
    pub storage: FileStorage,
    pub storage_path: String,
    pub sha256: String,
    pub size: u64,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CaseStatus {
    Built,
    BuildError,
    ReferenceCrash,
    ExcludedNetwork,
    ExcludedPlantuml,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CaseRecord {
    pub profile: String,
    pub source_set: String,
    pub case_id: String,
    pub status: CaseStatus,
    pub exit_code: Option<i32>,
    pub warnings: String,
    pub excluded_reason: Option<String>,
    pub origin: OriginRecord,
    pub input_sha256: String,
    pub tree_sha256: String,
    pub files: Vec<FileRecord>,
}
~~~

Validation rejects unknown fields, missing fields, unknown statuses, absolute paths, drive prefixes, parent components, symlinks, wrong hashes, wrong sizes, duplicate case keys, unsorted case keys, unknown profiles, and status-specific nullability errors. Every file reference must resolve below tests/fixtures/html_oracle.

## TDD implementation tasks

### Task 1: Profiles, configuration, and schema

- [ ] Write red tests in tools/test_gen_html_oracle.py for unknown status, missing input_sha256, path escape, hash mismatch, duplicate key, null exit_code for built, and files on an excluded case.
- [ ] Create tools/oracle_profiles/core/pyproject.toml with Sphinx 9.1.0, Docutils 0.22.4, and pytest 8 through 9.
- [ ] Create tools/oracle_profiles/local_needs/pyproject.toml with Sphinx 9.1.0, Docutils 0.21.2, sphinx-needs 8.5.0, and pytest 8 through 9. Do not add tool.uv.sources.
- [ ] Create tools/html_oracle_cases.toml with the six source sets and exact counts above. There is no pattern source set.
- [ ] Create both locks with uv lock --project tools/oracle_profiles/core and uv lock --project tools/oracle_profiles/local_needs. All later commands use uv run --locked.
- [ ] Implement the Python schema and validator in tools/gen_html_oracle.py using the complete types above.
- [ ] Implement the Rust serde model in tests/support/html_oracle.rs and a default schema test in tests/html_differential.rs.
- [ ] Run uv run --locked --project tools/oracle_profiles/core python -m pytest tools/test_gen_html_oracle.py -q; cargo fmt --all -- --check; cargo test --test html_differential schema. Expected green result: all schema tests pass.
- [ ] Commit test: define HTML oracle schema and profiles.

### Task 2: Discovery, materialization, provenance, and licensing

- [ ] Write red tests for 735 Docutils cases, 489 Sphinx cases, 29 projects, 84 documents, 7 HTML projects, 4 inventory projects, 142 needs projects, duplicate detection, ledger/discovery set equality, and needs-root escape.
- [ ] Implement core discovery in tools/gen_html_oracle.py. Preserve bytes, reject symlinks, create one-document snippet projects, rebuild environment projects as HTML, copy the seven named projects, and select only SPHINX_PROJECTS.
- [ ] Implement direct-child needs discovery at packages/sphinx-needs/tests/doc_test. Scan packages/sphinx-needs/tests/**/*.py with ast.parse. Emit repository-relative node IDs from file, class, and function scopes. Detect confoverrides and non-html builder values only to set variants_not_captured. Do not import tests or inspect assertions.
- [ ] Implement static inspection. Mark excluded-network for a remote URL that a build would fetch. A remote intersphinx target with local inventory remains allowed. Mark excluded-plantuml when conf.py loads sphinxcontrib.plantuml. If both match, excluded-network wins.
- [ ] Create NOTICE.md with exactly one licensing row per source set and one row for Sphinx and alabaster theme assets. Use BSD-2-Clause for Docutils and Sphinx-derived sets, the repository license for checked-in fixture projects, and MIT for sphinx-needs. State that the 881 pattern records are outside this HTML corpus and that licensing is not recorded per file.
- [ ] Run uv run --locked --project tools/oracle_profiles/core python -m pytest tools/test_gen_html_oracle.py -q -k discovery. Expected green result: source floors, exact project counts, and the 142 count pass.
- [ ] Commit feat: discover complete HTML oracle corpus.

### Task 3: Child runner, provenance, and network denial

- [ ] Write red tests for a conf.py socket attempt, core versions, wrong needs import location, wrong version, wrong commit, wrong tree, dirty package subtree, and harmless dirt outside that subtree.
- [ ] Create tools/html_oracle_runner.py. Parse --sourcedir, --outputdir, --doctree-dir, --builder, --warnings-file, --needs-root, and --verify-needs.
- [ ] Install this guard before importing Sphinx:

~~~python
import socket

def reject_network(*args, **kwargs):
    raise RuntimeError("network disabled by html oracle")

socket.socket.connect = reject_network
socket.create_connection = reject_network
socket.getaddrinfo = reject_network
~~~

- [ ] Call sphinx.cmd.build.main with argv ["-b", builder, "-d", str(doctree_dir), str(sourcedir), str(outputdir)]. The parent captures both streams and sets PYTHONNOUSERSITE=1.
- [ ] For local_needs, prepend needs-root/packages/sphinx-needs/src to PYTHONPATH and verify inside the child that sphinx_needs.__file__ is inside that path, __version__ is 8.5.0, git HEAD is 58bcb59d861da95f2aca79f343e8bae6ec5c1250, git HEAD:packages/sphinx-needs is 958172a89defcec69704f6b9d61e482e7c4e8409, and git status --porcelain -- packages/sphinx-needs is empty. There is no bypass flag.
- [ ] Normalize combined output in the fixed order stdout, bytes, stderr, bytes, then CRLF and source-root replacement. Classify the traceback marker as reference-crash and other nonzero output as build-error.
- [ ] Run uv run --locked --project tools/oracle_profiles/core python -m pytest tools/test_gen_html_oracle.py -q -k runner. Expected green result: child network access is rejected and all provenance checks are enforced.
- [ ] Commit feat: add guarded Sphinx oracle runner.

### Task 4: Complete trees, hashes, blobs, and atomic replacement

- [ ] Write red tests for symlink rejection, parent path rejection, repeated static bytes producing one blob, complete logical paths, hash changes, absolute-root leaks, and failure injection.
- [ ] Walk output with sorted relative paths and check is_symlink before is_file. Capture partial output for build-error and reference-crash. Store non-static output under refs and _static and _images bytes under blobs by content SHA-256.
- [ ] Write warnings to refs/profile/source_set/case_id/warnings.txt and validate them against the warnings field. Keep warnings outside the logical files list.
- [ ] Compute input_sha256 and tree_sha256 with canonical_hash. The empty entry list is used for excluded cases.
- [ ] Stage a complete replacement in tests/fixtures/html_oracle.staging. Validate every ledger hash, file, path, source-set key, lock digest, and root-leak rule. Rename old to .old, staging to final, and remove .old only after the final rename. Make HTML_ORACLE_INJECT_FAILURE_AFTER=case-count fail before rename and preserve the old tree.
- [ ] Run uv run --locked --project tools/oracle_profiles/core python -m pytest tools/test_gen_html_oracle.py -q -k "storage or hash or atomic". Expected green result: no symlink escapes, blobs deduplicate, logical trees remain complete, and failure is recoverable.
- [ ] Commit feat: store deterministic oracle trees atomically.

### Task 5: Fixed normalizers and comparator

- [ ] Write red Rust tests for CRLF text equality, exact .buildinfo bytes, warnings-only source-root replacement, searchindex key ordering, searchindex array ordering, malformed wrappers, canonical objects.inv records, opaque bytes, missing files, and unexpected files.
- [ ] Add Policy values Warnings, SearchIndex, ObjectsInventory, TextCrlf, and ExactBytes to tests/support/html_oracle.rs. Dispatch searchindex.js and objects.inv before general text extensions.
- [ ] Implement searchindex.js parsing of Search.setIndex( JSON ); with JSON object key order ignored, array order preserved, and scalar types exact.
- [ ] Implement objects.inv parsing of four exact header lines plus zlib records into five fields sorted by the complete tuple. Use invalid-objects-inventory and objects-inventory-value categories.
- [ ] Reconstruct blob files, compare the union of logical paths, compare normalized warnings, and compare status class. built requires success; build-error requires build-error plus exact warnings and files; reference-crash and excluded records are not scheduled.
- [ ] Return missing-file, unexpected-file, bytes-value, text-value, searchindex-value, objects-inventory-value, invalid-searchindex, invalid-objects-inventory, status, warning, spawn, io, and timeout categories.
- [ ] Run cargo fmt --all and cargo test --test html_differential normalizer comparator schema. Expected green result: every table policy test passes.
- [ ] Commit feat: compare HTML oracle trees by fixed policy.

### Task 6: Actual CLI execution and bounded diagnostics

- [ ] Write red tests that re-enter std::env::current_exe through an ignored helper selected by HTML_ORACLE_DIAGNOSTICS_HELPER. Test a timeout, more than 512 KiB on both streams, and a read or kill error.
- [ ] Define ExitStatusKind with Success, BuildError(i32), Timeout, SpawnError(String), and IoError(String) in tests/support/diagnostics.rs.
- [ ] Spawn with piped stdout and stderr, drain both pipes to EOF on reader threads, retain at most 256 KiB per stream plus [output truncated], poll every 20 milliseconds to a 60-second deadline, kill and drain on timeout, and map wait/read/kill failures to IoError.
- [ ] In tests/html_differential.rs invoke env!("CARGO_BIN_EXE_sphinx-ultra") with positional input and output paths, -b html, -d cache path, and -q. The output and cache directories are siblings under target/html-oracle/runs/profile/source_set/case_id. Do not use -M.
- [ ] Verify against src/main.rs: positional SOURCEDIR OUTPUTDIR, -b/--builder, -c/--conf-dir, -d/--doctree-dir, -D, -A, -t/--tag, -n/--nitpicky, -q/--quiet, -E/--fresh-env, -a/--write-all, and -T/--show-traceback.
- [ ] Run cargo test --test html_differential diagnostics smoke. Expected green result: CARGO_BIN_EXE_sphinx-ultra executes, cache is outside output, and no pipe deadlock occurs.
- [ ] Commit test: run Ultra with bounded diagnostics.

### Task 7: Two-root determinism and generator ordering

- [ ] Write red tests that generate one small complete source set below two different absolute roots and compare index.json, inputs, refs, blobs, order, and root-leak behavior.
- [ ] Implement sorted source-set, origin, case, file, and JSON-key ordering; UTF-8 JSON with indent 2 and final LF; process-pool -j N with canonical result collection; distinct absolute source, output, cache, and warnings paths per child.
- [ ] Implement the focused core command: uv run --locked --project tools/oracle_profiles/core python tools/gen_html_oracle.py --config tools/html_oracle_cases.toml --out tests/fixtures/html_oracle --profile core -j 2.
- [ ] Implement the focused local-needs command: uv run --locked --project tools/oracle_profiles/local_needs python tools/gen_html_oracle.py --config tools/html_oracle_cases.toml --out tests/fixtures/html_oracle --profile local_needs --needs-root C:\Users\johnm\Documents\repos\sphinx-needs -j 4.
- [ ] Ensure a focused profile run cannot remove or rewrite an unselected profile. The complete workflow stages both profiles before replacement.
- [ ] Run uv run --locked --project tools/oracle_profiles/core python -m pytest tools/test_gen_html_oracle.py -q -k deterministic. Expected green result: different absolute roots produce identical trees.
- [ ] Commit test: prove two-root oracle determinism.

### Task 8: Exhaustive differential run and reports

- [ ] Write red report tests for ordering by profile, source_set, case_id, logical_path, category; per-case pass/fail; counts per source set and category; excluded and reference-crash counts; and a 64 KiB assertion cap.
- [ ] Load every ledger case. Apply HTML_ORACLE_FILTER as a substring over profile/source_set/case_id. Schedule built and build-error. Do not schedule reference-crash, excluded-network, or excluded-plantuml. A filter with no matches is an error.
- [ ] Use available_parallelism as a bounded thread pool. Each worker runs CARGO_BIN_EXE_sphinx-ultra and returns an owned result. The main thread sorts all results before reporting.
- [ ] Write target/html-oracle/report.json with total_cases, scheduled_cases, passed_cases, failed_cases, excluded_cases, reference_crash_cases, counts_by_source_set, counts_by_category, and cases. Write equivalent Markdown summary, source-set, category, and per-failure sections.
- [ ] Cap each diff at 64 KiB and combined stdout plus assertion text at 64 KiB. The assertion points to both report files and states that the files contain every result.
- [ ] Mark the test #[ignore] as html_oracle_exhaustive. Run cargo test --test html_differential html_oracle_exhaustive -- --ignored --nocapture. Expected result while Ultra is incomplete: every runnable case is attempted, all failures are aggregated, and the command exits nonzero.
- [ ] Run cargo test --test html_differential report and cargo test --test html_differential -- --list. Expected green result: report tests pass and exhaustive is ignored by default.
- [ ] Commit test: add exhaustive HTML oracle report.

### Task 9: Generate artifacts, verify integrity, and hand off

- [ ] Run uv run --locked --project tools/oracle_profiles/core python -m pytest tools/test_gen_html_oracle.py -q; cargo fmt --all; cargo clippy --all-targets --all-features -- -D warnings; cargo test. Expected green result: all default checks pass before generation.
- [ ] Generate core with uv run --locked --project tools/oracle_profiles/core python tools/gen_html_oracle.py --config tools/html_oracle_cases.toml --out tests/fixtures/html_oracle --profile core -j 4.
- [ ] Generate local_needs with uv run --locked --project tools/oracle_profiles/local_needs python tools/gen_html_oracle.py --config tools/html_oracle_cases.toml --out tests/fixtures/html_oracle --profile local_needs --needs-root C:\Users\johnm\Documents\repos\sphinx-needs -j 4.
- [ ] Implement --verify and run uv run --locked --project tools/oracle_profiles/core python tools/gen_html_oracle.py --config tools/html_oracle_cases.toml --out tests/fixtures/html_oracle --verify.
- [ ] Verify every file record, size, hash, blob name, logical path, lock digest, source-set count, source/ledger key equality, hash formula, sorted serialization, symlink rule, path rule, root-leak rule, and final fixture size. Print total size and warn if it exceeds 150 MB.
- [ ] Run cargo test --test html_differential html_oracle_exhaustive -- --ignored --nocapture. Expected result: known-red nonzero with complete report.md and report.json.
- [ ] Run cargo fmt --all -- --check; cargo clippy --all-targets --all-features -- -D warnings; cargo test; uv run --locked --project tools/oracle_profiles/core python -m pytest tools/test_gen_html_oracle.py -q; git diff --check.
- [ ] Review that only the planned generator, profiles, fixtures, Rust support, tests, and NOTICE.md changed. Commit feat: commit complete HTML oracle corpus.

## Final review checklist

- [ ] src/main.rs still supports positional source/output, -b html, and -d cache paths exactly as used.
- [ ] tools/gen_sphinx_fixture.py still has extensions=[], master_doc='index', exclude_patterns=['_build'], smartquotes=False, and keep_warnings=True.
- [ ] tools/gen_inventory_fixture.py still has four SPHINX_PROJECTS entries.
- [ ] tests/fixtures still has exactly the seven named HTML projects.
- [ ] The sibling checkout still has packages/sphinx-needs/src/sphinx_needs and packages/sphinx-needs/tests/doc_test with exactly 142 conf.py directories.
- [ ] The local-needs commit and subtree tree match 58bcb59d861da95f2aca79f343e8bae6ec5c1250 and 958172a89defcec69704f6b9d61e482e7c4e8409.
- [ ] index.json contains only the five statuses above.
- [ ] The comparison code implements only the fixed table.
- [ ] Static blobs are deduplicated while all logical paths are present.
- [ ] The ignored exhaustive test uses CARGO_BIN_EXE_sphinx-ultra, external cache paths, all built and build-error cases, bounded diagnostics, deterministic ordering, and complete reports.
- [ ] No implementation code is written outside the planned paths.

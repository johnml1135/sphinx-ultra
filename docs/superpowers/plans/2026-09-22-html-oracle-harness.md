# HTML Oracle Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task.

**Goal:** Build a reproducible, offline, provenance-preserving HTML differential harness that captures real Sphinx HTML output, compares it with the actual CARGO_BIN_EXE_sphinx-ultra CLI, and covers every discovered local upstream case without pretending unsupported sphinx-needs directives are native Ultra inputs.

**Architecture:** A locked Python generator reads tools/html_oracle_cases.toml, discovers all declared source sets, runs pinned Sphinx HTML builds, and atomically writes a complete fixture tree plus a one-record-per-case ledger. Rust integration support loads and validates that ledger, executes the real binary in isolated roots with cache directories outside output, applies only declared path and CRLF normalization, and compares structured or opaque output according to explicit file policies. The default suite validates schema, provenance, hashes, discovery, normalization, comparators, and a smoke case; one ignored exhaustive test runs every runnable case and aggregates failures.

**Tech Stack:** Python 3.12 tomllib, pytest, uv locked offline environments, Sphinx 9.1.0, Docutils 0.22.4 for core, sphinx-needs 8.5.0 with Sphinx 9.1.0 and Docutils 0.21.2 for the local-needs profile, Rust integration tests, serde, serde_json, toml, blake3, sha2, tempfile, walkdir, and wait-timeout.

---

## Design contract and reviewable commit strategy

This is one comprehensive first PR because the value of an HTML oracle comes
from the generator, pinned reference environments, fixture ledger, comparator,
CLI runner, and exhaustive test agreeing on one contract. Splitting those
pieces across unrelated PRs would permit a green harness with no complete
corpus or a corpus with no trustworthy runner. The work is still divided into
small reviewable commits. Each commit below leaves its bounded layer tested;
the final commit wires the exhaustive opt-in behavior and runs the full
verification matrix.

The required implementation paths are:

- Create: tools/gen_html_oracle.py
- Create: tools/html_oracle_cases.toml
- Create: tests/html_differential.rs
- Create: tests/support/html_oracle.rs
- Create: tests/support/diagnostics.rs
- Create: tests/fixtures/html_oracle/index.json
- Create: tests/fixtures/html_oracle/inputs/
- Create: tests/fixtures/html_oracle/refs/
- Create: tests/fixtures/html_oracle/blobs/
- Create: tests/fixtures/html_oracle/NOTICE.md

The plan also creates the locked Python profile files and generator unit tests
needed to make regeneration executable without network access.

### Task 1: Add the locked profile and ledger schema contract

**Files:**

- Create: tools/html_oracle_cases.toml
- Create: tools/oracle_profiles/core/pyproject.toml
- Create: tools/oracle_profiles/core/uv.lock
- Create: tools/oracle_profiles/local_needs/pyproject.toml
- Create: tools/oracle_profiles/local_needs/uv.lock
- Create: tools/test_gen_html_oracle.py
- Modify: Cargo.toml
- Modify: Cargo.lock

- [ ] **Step 1: Write schema tests first.**

Add these tests to tools/test_gen_html_oracle.py:

~~~python
from pathlib import Path

import pytest

from gen_html_oracle import load_spec, validate_spec


ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "tools" / "html_oracle_cases.toml"


def test_spec_declares_all_statuses_and_expectations():
    spec = load_spec(SPEC)
    validate_spec(spec)
    assert spec["statuses"] == [
        "active",
        "alias",
        "excluded-network",
        "excluded-plantuml",
        "excluded-external-test-fixture",
        "unsupported-builder",
        "reference-crash",
    ]
    assert spec["expectations"] == ["match", "expected-failure", "reference-only"]


def test_profile_pins_are_exact():
    spec = load_spec(SPEC)
    assert spec["profiles"]["core"] == {
        "sphinx": "9.1.0",
        "docutils": "0.22.4",
        "lock": "tools/oracle_profiles/core/uv.lock",
    }
    assert spec["profiles"]["local_needs"] == {
        "sphinx": "9.1.0",
        "docutils": "0.21.2",
        "sphinx_needs": "8.5.0",
        "commit": "58bcb59d861da95f2aca79f343e8bae6ec5c1250",
        "subtree_tree": "958172a89defcec69704f6b9d61e482e7c4e8409",
        "lock": "tools/oracle_profiles/local_needs/uv.lock",
    }


def test_invalid_status_is_rejected(tmp_path):
    bad = tmp_path / "bad.toml"
    bad.write_text(
        "schema_version = 1\nstatuses = ['active', 'not-valid']\nexpectations = ['match']\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="status"):
        validate_spec(load_spec(bad))
~~~

- [ ] **Step 2: Run the schema tests and verify the red result.**

Run:

~~~powershell
python -m pytest tools/test_gen_html_oracle.py -q
~~~

Expected result before implementation: FAIL because
tools/gen_html_oracle.py and tools/html_oracle_cases.toml do not exist.

- [ ] **Step 3: Add the exact declarative schema.**

Create tools/html_oracle_cases.toml with this contract:

~~~toml
schema_version = 1
statuses = [
  "active",
  "alias",
  "excluded-network",
  "excluded-plantuml",
  "excluded-external-test-fixture",
  "unsupported-builder",
  "reference-crash",
]
expectations = ["match", "expected-failure", "reference-only"]

[corpus_floors]
docutils_snippets = 735
sphinx_read_snippets = 489
environment_projects = 29
environment_documents = 84
html_projects = 7
inventory_projects = 4
pattern_cases = 881
sphinx_needs_doc_tests = 142

[profiles.core]
sphinx = "9.1.0"
docutils = "0.22.4"
lock = "tools/oracle_profiles/core/uv.lock"

[profiles.local_needs]
sphinx = "9.1.0"
docutils = "0.21.2"
sphinx_needs = "8.5.0"
commit = "58bcb59d861da95f2aca79f343e8bae6ec5c1250"
subtree_tree = "958172a89defcec69704f6b9d61e482e7c4e8409"
lock = "tools/oracle_profiles/local_needs/uv.lock"

[discovery]
needs_root_env = "SPHINX_NEEDS_ROOT"
html_fixture_root = "tests/fixtures"
inventory_manifest = "tests/fixtures/inventories/manifest.json"
docutils_fixture = "tests/fixtures/doctree_differential.json"
sphinx_fixture = "tests/fixtures/sphinx_doctree_differential.json"
environment_fixture = "tests/fixtures/env_differential.json"
pattern_fixture = "tests/fixtures/pattern_differential.json"

[[source_sets]]
name = "docutils_snippets"
kind = "fixture_cases"
path = "tests/fixtures/doctree_differential.json"
profile = "core"
id_field = "name"
floor = 735

[[source_sets]]
name = "sphinx_read_snippets"
kind = "fixture_cases"
path = "tests/fixtures/sphinx_doctree_differential.json"
profile = "core"
id_field = "name"
floor = 489

[[source_sets]]
name = "environment_projects"
kind = "environment_projects"
path = "tests/fixtures/env_differential.json"
profile = "core"
id_field = "name"
floor = 29
document_floor = 84

[[source_sets]]
name = "html_projects"
kind = "checked_in_html_projects"
path = "tests/fixtures"
profile = "core"
names = ["basic", "basic_missing_ref", "deps_image", "intersphinx", "literalinclude", "toctree_forms", "toctree_glob"]
floor = 7

[[source_sets]]
name = "inventory_projects"
kind = "inventory_projects"
path = "tests/fixtures/inventories/manifest.json"
profile = "core"
floor = 4

[[source_sets]]
name = "pattern_cases"
kind = "fixture_cases"
path = "tests/fixtures/pattern_differential.json"
profile = "core"
id_field = "pattern_id"
floor = 881

[[source_sets]]
name = "sphinx_needs_doc_tests"
kind = "sphinx_needs_doc_tests"
profile = "local_needs"
floor = 142

[[policy]]
path = "**/*.json"
kind = "json"

[[policy]]
path = "**/searchindex.js"
kind = "searchindex-js"

[[policy]]
path = "**/objects.inv"
kind = "objects-inv"

[[policy]]
path = "**/*"
kind = "opaque"
~~~

The generator must reject unknown keys in the schema, duplicate source-set
names, duplicate status values, non-relative fixture paths, and floors below
the evidence table. pattern_id is assigned by the generator from the sorted
(pattern, path) pair because the existing pattern fixture has no explicit
case-name field.

- [ ] **Step 4: Add the profile projects and generate their locks.**

Create tools/oracle_profiles/core/pyproject.toml:

~~~toml
[project]
name = "sphinx-ultra-html-oracle-core"
version = "0.0.0"
requires-python = ">=3.12,<3.13"
dependencies = ["sphinx==9.1.0", "docutils==0.22.4"]
~~~

Create tools/oracle_profiles/local_needs/pyproject.toml:

~~~toml
[project]
name = "sphinx-ultra-html-oracle-local-needs"
version = "0.0.0"
requires-python = ">=3.12,<3.13"
dependencies = ["sphinx==9.1.0", "docutils==0.21.2", "sphinx-needs==8.5.0", "pytest==8.3.5"]
~~~

Generate both committed locks with the network disabled:

~~~powershell
uv lock --offline --project tools/oracle_profiles/core
uv lock --offline --project tools/oracle_profiles/local_needs
~~~

Expected result: both commands succeed using the local package cache and
write uv.lock. If either command needs a download, stop with a clear error;
do not relax the offline requirement or substitute a floating dependency.

- [ ] **Step 5: Implement only schema loading and validation.**

Create tools/gen_html_oracle.py with these exact interfaces before adding
discovery logic:

~~~python
from pathlib import Path
import tomllib


def load_spec(path: Path) -> dict:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def validate_spec(spec: dict) -> None:
    required_statuses = {
        "active",
        "alias",
        "excluded-network",
        "excluded-plantuml",
        "excluded-external-test-fixture",
        "unsupported-builder",
        "reference-crash",
    }
    required_expectations = {"match", "expected-failure", "reference-only"}
    statuses = spec.get("statuses")
    expectations = spec.get("expectations")
    if spec.get("schema_version") != 1:
        raise ValueError("schema_version must equal 1")
    if set(statuses or ()) != required_statuses or len(statuses) != len(required_statuses):
        raise ValueError("status vocabulary does not match the HTML oracle contract")
    if set(expectations or ()) != required_expectations or len(expectations) != len(required_expectations):
        raise ValueError("expectation vocabulary does not match the HTML oracle contract")
    if set(spec.get("profiles", ())) != {"core", "local_needs"}:
        raise ValueError("core and local_needs profiles are required")
    if len({entry["name"] for entry in spec.get("source_sets", ())}) != len(spec["source_sets"]):
        raise ValueError("source-set names must be unique")
~~~

- [ ] **Step 6: Add the Rust parsing dependencies and test the contract.**

Add these development dependencies to Cargo.toml:

~~~toml
toml = "0.9"
sha2 = "0.10"
wait-timeout = "0.2"
walkdir = "2.5"
~~~

Run:

~~~powershell
cargo check --tests --locked
python -m pytest tools/test_gen_html_oracle.py -q
~~~

Expected result: PASS for the Python schema tests and PASS for Cargo
dependency resolution. The lockfile must change only for these direct test
dependencies and their exact transitive entries.

- [ ] **Step 7: Commit the bounded schema task.**

~~~powershell
git add Cargo.toml Cargo.lock tools/html_oracle_cases.toml tools/oracle_profiles tools/gen_html_oracle.py tools/test_gen_html_oracle.py
git commit -m "test: define html oracle profiles and schema"
~~~

### Task 2: Discover every source set and emit the provenance ledger

**Files:**

- Modify: tools/gen_html_oracle.py
- Modify: tools/test_gen_html_oracle.py
- Create: tests/fixtures/html_oracle/NOTICE.md

- [ ] **Step 1: Write discovery tests before implementation.**

Add tests that call discover_source_sets(ROOT, spec) and assert the exact
floors and identities:

~~~python
from gen_html_oracle import discover_source_sets


def test_existing_source_sets_meet_the_evidence_floors():
    discovered = discover_source_sets(ROOT, load_spec(SPEC))
    assert len(discovered["docutils_snippets"]) >= 735
    assert len(discovered["sphinx_read_snippets"]) >= 489
    assert len(discovered["environment_projects"]) >= 29
    assert sum(len(case.documents) for case in discovered["environment_projects"]) >= 84
    assert len(discovered["html_projects"]) == 7
    assert len(discovered["inventory_projects"]) >= 4
    assert len(discovered["pattern_cases"]) >= 881


def test_html_projects_are_the_seven_checked_in_projects():
    discovered = discover_source_sets(ROOT, load_spec(SPEC))
    assert [case.case_id for case in discovered["html_projects"]] == [
        "basic",
        "basic_missing_ref",
        "deps_image",
        "intersphinx",
        "literalinclude",
        "toctree_forms",
        "toctree_glob",
    ]


def test_duplicate_source_identity_fails():
    records = [{"source_set": "x", "case_id": "same"}, {"source_set": "x", "case_id": "same"}]
    with pytest.raises(ValueError, match="exactly once"):
        validate_unique_source_ids(records)
~~~

- [ ] **Step 2: Run the discovery tests and verify the red result.**

~~~powershell
python -m pytest tools/test_gen_html_oracle.py -q -k "source_sets or duplicate_source"
~~~

Expected result before implementation: FAIL because the discovery functions
are not defined.

- [ ] **Step 3: Implement typed source records and exact discovery.**

Define these records in tools/gen_html_oracle.py:

~~~python
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceCase:
    source_set: str
    case_id: str
    profile: str
    origin_path: str
    origin_nodeid: str | None
    source_files: Sequence[str]
    status: str
    expectation: str
    builder: str
    license_spdx: str


@dataclass(frozen=True)
class ProjectCase(SourceCase):
    documents: Sequence[str]


def validate_unique_source_ids(records: list[dict]) -> None:
    seen: set[tuple[str, str]] = set()
    for record in records:
        key = (record["source_set"], record["case_id"])
        if key in seen:
            raise ValueError(f"source case {key!r} must appear exactly once")
        seen.add(key)
~~~

Implement one deterministic adapter per kind in the TOML schema. The adapters
must:

- read each JSON fixture's case names without changing its input bytes;
- read environment project names and count document keys;
- list exactly the seven named HTML projects and reject an extra or missing
  checked-in project;
- derive the four Sphinx-built inventory project identities from the real
  SPHINX_PROJECTS entries represented by raw_objects, while preserving the
  twelve committed .inv files as separate artifact records;
- assign sorted (pattern, path) IDs to all 881 pattern records;
- walk the configured SPHINX_NEEDS_ROOT doc_test tree, collect 142 project
  cases with their pytest node IDs, and fail if the root is absent, the
  pinned commit is wrong, or the subtree tree is wrong;
- record license_spdx, source revision, profile lock digest, builder, and
  origin path on every record.

The local-needs adapter must read the originating pytest expectation rather
than inventing one. It must record the test node ID, expected status, expected
file-regression or assertion artifact path, and a digest of the expectation.
If the test does not expose a machine-readable artifact, emit a
reference-only record with the original node ID and reason; never synthesize
an Ultra expectation.

- [ ] **Step 4: Add provenance and licensing records.**

Write tests/fixtures/html_oracle/NOTICE.md with one table row for each source
set:

~~~markdown
# HTML Oracle Fixture Notices

The generated fixture contains source-derived inputs and reference outputs.
The repository's MIT license applies to harness code. Imported source and
expected artifacts retain the license of their originating project.

| Source set | Revision or lock | License record | Scope |
|---|---|---|---|
| docutils_snippets | Docutils 0.22.4 | BSD-2-Clause | Derived test inputs and outputs |
| sphinx_read_snippets | Sphinx 9.1.0, Docutils 0.22.4 | BSD-2-Clause | Derived test inputs and outputs |
| environment_projects | Sphinx 9.1.0, Docutils 0.22.4 | BSD-2-Clause | Derived project inputs and outputs |
| html_projects | Repository commit containing this fixture | MIT | Checked-in project inputs and outputs |
| inventory_projects | Sphinx 9.1.0, Docutils 0.22.4 | BSD-2-Clause | Derived inventory records and bytes |
| pattern_cases | Sphinx 9.1.0 | BSD-2-Clause | Derived pattern inputs and results |
| sphinx_needs_doc_tests | sphinx-needs 8.5.0 at commit 58bcb59d861da95f2aca79f343e8bae6ec5c1250, subtree 958172a89defcec69704f6b9d61e482e7c4e8409 | MIT | Derived project inputs and originating pytest expectations |
~~~

The generator must append the exact upstream license-file paths and SHA-256
digests discovered in each pinned environment. It must not copy package code,
fonts, JavaScript, or images unless the license table says the artifact may be
redistributed. Every copied external file gets an origin and license in the
ledger.

- [ ] **Step 5: Implement an atomic deterministic ledger writer.**

Use this interface and ordering:

~~~python
import json
import os
import tempfile


def write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def ordered_cases(cases: list[SourceCase]) -> list[SourceCase]:
    return sorted(cases, key=lambda case: (case.source_set, case.case_id))
~~~

The generated index.json must contain schema_version, generator, exact profile
records, source-set counts and digests, and one cases entry per source
identity. It must reject duplicate IDs before writing. The generator must write
to a temporary fixture root and atomically replace only the final files after
all builds and hash checks succeed.

- [ ] **Step 6: Run the discovery tests and commit.**

~~~powershell
python -m pytest tools/test_gen_html_oracle.py -q
git add tools/gen_html_oracle.py tools/test_gen_html_oracle.py tests/fixtures/html_oracle/NOTICE.md
git commit -m "test: discover html oracle source sets"
~~~

Expected result: all discovery, pin, provenance, and duplicate-identity tests
pass, with no generated output yet.

### Task 3: Build pinned Sphinx references with no network

**Files:**

- Modify: tools/gen_html_oracle.py
- Modify: tools/test_gen_html_oracle.py
- Modify: tools/html_oracle_cases.toml

- [ ] **Step 1: Write reference-build tests first.**

Add tests for command construction and network denial:

~~~python
from gen_html_oracle import build_reference_command, network_denied


def test_reference_command_uses_html_builder_and_external_doctree_dir():
    command = build_reference_command(
        profile="core",
        source=Path("source"),
        output=Path("output"),
        doctree=Path("cache"),
    )
    assert command[-7:] == [
        "-b", "html",
        "-d", "cache",
        "-q",
        "source",
        "output",
    ]


def test_network_guard_rejects_socket_connect():
    with network_denied():
        with pytest.raises(RuntimeError, match="network disabled"):
            socket.create_connection(("127.0.0.1", 9), timeout=0.01)
~~~

- [ ] **Step 2: Run the tests and verify the red result.**

~~~powershell
python -m pytest tools/test_gen_html_oracle.py -q -k "reference_command or network_guard"
~~~

Expected result before implementation: FAIL because the reference command
and network guard do not exist.

- [ ] **Step 3: Add the pinned reference command and network guard.**

Implement build_reference_command so it invokes the profile's Python with
python -m sphinx, -b html, -d pointing to a cache directory outside the
output, and -q. The command must run with PYTHONNOUSERSITE=1, NO_PROXY=*,
no_proxy=*, and an empty Sphinx intersphinx mapping unless the case is marked
excluded-network. network_denied() must patch socket.socket.connect,
socket.create_connection, urllib.request.urlopen, and subprocess.Popen when
its executable is a network client. The guard is active during reference
generation and restored in a finally block.

Use this build record:

~~~python
@dataclass(frozen=True)
class ReferenceBuild:
    case_id: str
    status: str
    exit_code: int | None
    timed_out: bool
    stdout: bytes
    stderr: bytes
    output_root: Path
    cache_root: Path
~~~

For every active or expected-failure case, materialize the input under a
temporary absolute root, run a real HTML build, capture all output files, and
retain warnings and exit status. A snippet case becomes a one-document
project with conf.py and index.rst; a project case is copied as a complete
source tree. Reference builds never call a private read-phase parser as their
sole oracle.

- [ ] **Step 4: Add explicit handling for reference-only and excluded cases.**

Before spawning Sphinx, validate the ledger record:

~~~python
def should_build_reference(case: dict) -> bool:
    return case["status"] in {"active", "alias", "reference-crash"} and case["expectation"] != "reference-only"
~~~

excluded-network, excluded-plantuml, excluded-external-test-fixture, and
unsupported-builder records receive no subprocess. Their ledger entry must
contain excluded_reason and reference_only = true. A local-needs case with an
unsupported extension directive receives its originating pytest expectation
and reference-only; it is not rewritten into a plain RST case.

- [ ] **Step 5: Run a real core reference build and verify green behavior.**

~~~powershell
$env:PYTHONNOUSERSITE = "1"
uv run --offline --locked --project tools/oracle_profiles/core python tools/gen_html_oracle.py --config tools/html_oracle_cases.toml --profile core --source-set html_projects --no-network --out tests/fixtures/html_oracle
~~~

Expected result: every one of the seven checked-in projects is built through
the HTML builder, caches are outside each output directory, and the generator
reports deterministic case IDs and zero network attempts.

- [ ] **Step 6: Commit the reference-build task.**

~~~powershell
git add tools/gen_html_oracle.py tools/test_gen_html_oracle.py tools/html_oracle_cases.toml
git commit -m "test: build pinned html oracle references"
~~~

### Task 4: Capture complete logical trees and deduplicate static assets

**Files:**

- Modify: tools/gen_html_oracle.py
- Modify: tools/test_gen_html_oracle.py
- Create: tests/fixtures/html_oracle/inputs/
- Create: tests/fixtures/html_oracle/refs/
- Create: tests/fixtures/html_oracle/blobs/
- Modify: tests/fixtures/html_oracle/index.json

- [ ] **Step 1: Write tree and hash tests first.**

Add tests that prove identical assets share one blob while both logical trees
retain their own paths:

~~~python
from gen_html_oracle import capture_tree, sha256_bytes


def test_static_asset_dedup_keeps_both_logical_paths(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    (first / "_static").mkdir()
    (second / "_static").mkdir()
    (first / "_static" / "theme.css").write_bytes(b"body{}\r\n")
    (second / "_static" / "theme.css").write_bytes(b"body{}\r\n")
    store = tmp_path / "store"
    one = capture_tree(first, store, "case-one")
    two = capture_tree(second, store, "case-two")
    assert one["files"]["_static/theme.css"]["blob"] == two["files"]["_static/theme.css"]["blob"]
    assert len(list((store / "blobs").iterdir())) == 1
    assert set(one["files"]) == {"_static/theme.css"}
    assert set(two["files"]) == {"_static/theme.css"}


def test_hash_is_sha256_of_exact_bytes():
    assert sha256_bytes(b"body{}\r\n") == "c7dcb398bf735520e2241af1a61a2b5ed12d9b54551f0d60b9658c134370a31d"
~~~

- [ ] **Step 2: Run the tree tests and verify the red result.**

~~~powershell
python -m pytest tools/test_gen_html_oracle.py -q -k "dedup or hash"
~~~

Expected result before implementation: FAIL because tree capture and hash
storage are not defined.

- [ ] **Step 3: Implement exact-byte capture and content-addressed blobs.**

Use SHA-256 for the on-disk content address and preserve every relative output
path in the logical tree:

~~~python
import hashlib


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def capture_tree(output_root: Path, fixture_root: Path, case_key: str) -> dict:
    files = {}
    for path in sorted(p for p in output_root.rglob("*") if p.is_file()):
        logical = path.relative_to(output_root).as_posix()
        data = path.read_bytes()
        digest = sha256_bytes(data)
        if is_static_asset(logical):
            blob = fixture_root / "blobs" / digest
            blob.parent.mkdir(parents=True, exist_ok=True)
            if blob.exists() and blob.read_bytes() != data:
                raise ValueError(f"blob collision for {digest}")
            if not blob.exists():
                blob.write_bytes(data)
            files[logical] = {"kind": "blob", "blob": digest, "bytes": len(data)}
        else:
            ref = fixture_root / "refs" / case_key / logical
            ref.parent.mkdir(parents=True, exist_ok=True)
            ref.write_bytes(data)
            files[logical] = {
                "kind": "ref",
                "path": ref.relative_to(fixture_root).as_posix(),
                "sha256": digest,
                "bytes": len(data),
            }
    return {"case_key": case_key, "files": files}
~~~

is_static_asset must return true for .css, .js, .png, .jpg, .jpeg, .gif,
.svg, .woff, .woff2, .ttf, .ico, .webp, and .map; it must return false for
HTML, JSON, searchindex.js, objects.inv, warning logs, and the structured
reference records. This keeps semantic files inspectable while deduplicating
static bytes. A missing output file is a tree error, not an implicit empty
file.

- [ ] **Step 4: Preserve inputs and complete tree metadata.**

Copy each input source tree into tests/fixtures/html_oracle/inputs/<case_key>
with sorted paths and exact bytes. Add input_sha256, tree_sha256,
reference_status, and files to the case record. The files map must list every
logical output path, including files that point to the same blob. The generator
must reject symlinks, absolute logical paths, parent path components, duplicate
case keys, and output files not represented in the map.

- [ ] **Step 5: Run deterministic generation twice and compare manifests.**

~~~powershell
$env:PYTHONNOUSERSITE = "1"
uv run --offline --locked --project tools/oracle_profiles/core python tools/gen_html_oracle.py --config tools/html_oracle_cases.toml --profile core --no-network --out tests/fixtures/html_oracle
Copy-Item tests/fixtures/html_oracle/index.json $env:TEMP/html-oracle-index-one.json
uv run --offline --locked --project tools/oracle_profiles/core python tools/gen_html_oracle.py --config tools/html_oracle_cases.toml --profile core --no-network --out tests/fixtures/html_oracle
Compare-Object (Get-Content $env:TEMP/html-oracle-index-one.json) (Get-Content tests/fixtures/html_oracle/index.json)
~~~

Expected result: Compare-Object emits no differences; all inputs, refs, and
blobs have stable names and bytes. Repeat the command for the local-needs
profile with SPHINX_NEEDS_ROOT set to the pinned sibling checkout.

- [ ] **Step 6: Commit the fixture storage task.**

~~~powershell
git add tools/gen_html_oracle.py tools/test_gen_html_oracle.py tests/fixtures/html_oracle
git commit -m "test: store complete deduplicated html trees"
~~~

### Task 5: Add bounded process diagnostics and deterministic execution

**Files:**

- Create: tests/support/diagnostics.rs
- Create: tests/html_differential.rs
- Modify: Cargo.toml
- Modify: Cargo.lock

- [ ] **Step 1: Write diagnostic runner tests first.**

Add unit tests in tests/support/diagnostics.rs for success, spawn failure,
timeout, and bounded output:

~~~rust
#[test]
fn captures_a_successful_process() {
    let outcome = run_checked(CommandSpec::new("rustc").args(["--version"]), Duration::from_secs(5));
    assert_eq!(outcome.status, ExitStatusKind::Success);
    assert!(!outcome.stdout.is_empty());
}

#[test]
fn records_spawn_failure_without_panicking() {
    let outcome = run_checked(CommandSpec::new("definitely-not-a-program"), Duration::from_secs(1));
    assert!(matches!(outcome.status, ExitStatusKind::SpawnFailure(_)));
}

#[test]
fn kills_a_timeout_and_caps_output() {
    let outcome = run_checked(CommandSpec::new("rustc").args(["--version"]), Duration::from_nanos(1));
    assert!(matches!(outcome.status, ExitStatusKind::TimedOut));
    assert!(outcome.stdout.len() <= MAX_CAPTURE_BYTES);
    assert!(outcome.stderr.len() <= MAX_CAPTURE_BYTES);
}
~~~

- [ ] **Step 2: Run the diagnostic tests and verify the red result.**

~~~powershell
cargo test --test html_differential diagnostics -- --nocapture
~~~

Expected result before implementation: FAIL because the support module and
integration test target do not exist.

- [ ] **Step 3: Implement the bounded runner.**

Create these exact types and constants:

~~~rust
pub const MAX_CAPTURE_BYTES: usize = 256 * 1024;
pub const MAX_DIFF_BYTES: usize = 64 * 1024;
pub const CASE_TIMEOUT: Duration = Duration::from_secs(120);

pub struct CommandSpec {
    pub program: PathBuf,
    pub args: Vec<OsString>,
    pub env: BTreeMap<OsString, OsString>,
    pub cwd: Option<PathBuf>,
}

pub enum ExitStatusKind {
    Success,
    Exit(i32),
    Signaled,
    TimedOut,
    SpawnFailure(String),
}

pub struct ProcessOutcome {
    pub status: ExitStatusKind,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
    pub truncated_stdout: bool,
    pub truncated_stderr: bool,
}

pub fn run_checked(spec: CommandSpec, timeout: Duration) -> ProcessOutcome;
~~~

Spawn with piped stdout and stderr, kill_on_drop(true), the supplied
environment, and a clean RUST_LOG. Read both pipes on dedicated threads into
bounded buffers, use wait-timeout for the deadline, kill the child on timeout,
and join both readers before returning. Never panic on a spawn, wait, read, or
kill operation. Encode failures in ExitStatusKind.

- [ ] **Step 4: Add deterministic command ordering and network denial.**

Implement sorted_case_ids with BTreeMap and BTreeSet, and add these environment
overrides to every Ultra invocation:

~~~rust
cmd.env_remove("HTTP_PROXY")
    .env_remove("HTTPS_PROXY")
    .env_remove("ALL_PROXY")
    .env("NO_PROXY", "*")
    .env("no_proxy", "*")
    .env("RUST_LOG", "off");
~~~

The runner must refuse any case whose status is excluded-network unless the
test is explicitly exercising ledger exclusion. The output and diagnostic
lists are sorted by source set, case ID, and logical path before formatting.

- [ ] **Step 5: Run and commit diagnostics.**

~~~powershell
cargo fmt --all
cargo test --test html_differential diagnostics -- --nocapture
git add tests/support/diagnostics.rs tests/html_differential.rs Cargo.toml Cargo.lock
git commit -m "test: bound oracle process diagnostics"
~~~

Expected result: process tests pass, timeout output is bounded, and failure
records are deterministic.

### Task 6: Implement ledger validation, exact normalizers, and comparators

**Files:**

- Create: tests/support/html_oracle.rs
- Modify: tests/html_differential.rs
- Modify: Cargo.toml
- Modify: Cargo.lock

- [ ] **Step 1: Write normalizer and comparator tests first.**

Add these tests to tests/support/html_oracle.rs:

~~~rust
#[test]
fn normalizes_crlf_and_only_declared_path_fields() {
    let policy = NormalizationPolicy {
        path_fields: vec![FieldBoundary::JsonPointer("/metadata/source".into())],
    };
    let input = br#"{"metadata":{"source":"ROOT_A/index.rst"},"body":"ROOT_A"}\r
"#;
    let actual = normalize_json(input, &policy, "ROOT_A", "ROOT_B").unwrap();
    assert_eq!(actual, br#"{"body":"ROOT_A","metadata":{"source":"ROOT_B/index.rst"}}
"#);
}

#[test]
fn opaque_bytes_are_compared_exactly() {
    assert!(compare_opaque(b"a\r\n", b"a\n").is_err());
}

#[test]
fn searchindex_comparison_preserves_array_order() {
    let left = b"Search.setIndex({\"docnames\":[\"a\",\"b\"]})";
    let right = b"Search.setIndex({\"docnames\":[\"b\",\"a\"]})";
    assert!(compare_searchindex(left, right).is_err());
}

#[test]
fn objects_inventory_compares_decoded_records() {
    let left = include_bytes!("../../tests/fixtures/inventories/std_objects_and_docs.inv");
    assert!(compare_objects_inv(left, left).is_ok());
}
~~~

- [ ] **Step 2: Run the comparator tests and verify the red result.**

~~~powershell
cargo test --test html_differential normalizer -- --nocapture
~~~

Expected result before implementation: FAIL because
tests/support/html_oracle.rs is absent.

- [ ] **Step 3: Add the ledger and fixture model.**

Define the serde model in tests/support/html_oracle.rs:

~~~rust
#[derive(Deserialize)]
pub struct Ledger {
    pub schema_version: u32,
    pub profiles: BTreeMap<String, ProfileRecord>,
    pub source_sets: BTreeMap<String, SourceSetRecord>,
    pub cases: Vec<CaseRecord>,
}

#[derive(Deserialize)]
pub struct CaseRecord {
    pub source_set: String,
    pub case_id: String,
    pub status: String,
    pub expectation: String,
    pub profile: String,
    pub builder: String,
    pub input_root: String,
    pub reference_root: String,
    pub origin: OriginRecord,
    pub files: BTreeMap<String, FileRecord>,
}

#[derive(Deserialize)]
#[serde(tag = "kind")]
pub enum FileRecord {
    Ref { path: String, sha256: String, bytes: usize },
    Blob { blob: String, bytes: usize },
}
~~~

Load index.json with serde_json, reject a schema version other than 1,
validate all status and expectation values, require sorted unique case keys,
require every input and reference path to remain below the fixture root, and
verify every SHA-256 before comparison. Use the toml crate to parse
tools/html_oracle_cases.toml in a schema test and assert that the generated
ledger's source-set names and floor metadata match it.

- [ ] **Step 4: Implement exact-boundary normalization.**

Define:

~~~rust
pub enum FieldBoundary {
    JsonPointer(String),
    SearchIndexJsonPointer(String),
    InventoryField { record: usize, field: String },
    HtmlAttribute { tag: String, attribute: String },
}

pub struct NormalizationPolicy {
    pub path_fields: Vec<FieldBoundary>,
}

pub fn normalize_crlf(bytes: &[u8]) -> Vec<u8>;
pub fn normalize_json(bytes: &[u8], policy: &NormalizationPolicy, from: &str, to: &str) -> Result<Vec<u8>, String>;
pub fn normalize_searchindex(bytes: &[u8], policy: &NormalizationPolicy, from: &str, to: &str) -> Result<Value, String>;
~~~

normalize_crlf changes only CRLF to LF; it does not trim, collapse
whitespace, decode opaque bytes, or change lone CR. Path replacement operates
only on the declared field boundary and requires a complete from-root token
followed by slash or the end of the field. A path token found anywhere else is
a hard failure. There is no global string replacement.

- [ ] **Step 5: Implement the structured policies.**

Implement these comparators:

~~~rust
pub fn compare_json(left: &[u8], right: &[u8], policy: &NormalizationPolicy) -> Result<(), String>;
pub fn compare_searchindex(left: &[u8], right: &[u8]) -> Result<(), String>;
pub fn compare_objects_inv(left: &[u8], right: &[u8]) -> Result<(), String>;
pub fn compare_opaque(left: &[u8], right: &[u8]) -> Result<(), String>;
~~~

For JSON, parse both values, recursively sort object keys for comparison, and
preserve array order and scalar types. For searchindex.js, require the exact
Search.setIndex( prefix and closing parenthesis, parse the enclosed JSON, sort
object keys only, and preserve every array order and value. For objects.inv,
parse the four header lines, zlib-decompress the payload, parse each semantic
record with the existing inventory semantics, sort records by the Sphinx
writer key (domain, name, display_name, objtype, docname, anchor, priority),
and compare the resulting headers and records. Do not compare zlib-compressed
bytes as a semantic inventory contract. For all other files, compare exact
bytes after the file's declared CRLF policy; images, fonts, maps, and unknown
binaries never receive text normalization.

- [ ] **Step 6: Add comparator and schema tests to the default integration test.**

The tests must prove:

- every ledger case appears once;
- each source set meets its floor and equals its discovered identity set;
- every input, ref, and blob is referenced exactly as declared;
- every blob digest matches its bytes;
- every reference file has a policy;
- no absolute path or parent segment appears in logical paths;
- normalization changes only CRLF and declared fields;
- JSON, searchindex, inventory, and opaque comparisons follow their policies.

- [ ] **Step 7: Run and commit the comparison layer.**

~~~powershell
cargo fmt --all
cargo test --test html_differential normalizer -- --nocapture
cargo test --test html_differential schema -- --nocapture
git add Cargo.toml Cargo.lock tests/support/html_oracle.rs tests/html_differential.rs
git commit -m "test: add html oracle comparators"
~~~

### Task 7: Execute the real Ultra CLI and test two absolute roots

**Files:**

- Modify: tests/html_differential.rs
- Modify: tests/support/html_oracle.rs
- Modify: tests/support/diagnostics.rs

- [ ] **Step 1: Write the CLI smoke and repeat-build tests first.**

Add these tests:

~~~rust
#[test]
fn html_oracle_smoke_runs_the_actual_binary() {
    let case = load_ledger().case("html_projects", "basic");
    let run = build_with_ultra(case, CASE_TIMEOUT);
    assert!(matches!(run.status, ExitStatusKind::Success), "{run:?}");
    assert!(run.output_root.join("index.html").is_file());
}

#[test]
fn same_case_matches_under_two_absolute_roots() {
    let case = load_ledger().case("html_projects", "basic");
    let first = build_with_ultra_at(case, absolute_root("oracle-root-a"), CASE_TIMEOUT);
    let second = build_with_ultra_at(case, absolute_root("oracle-root-b"), CASE_TIMEOUT);
    assert_ne!(first.source_root, second.source_root);
    assert_ne!(first.output_root, second.output_root);
    assert!(!first.cache_root.starts_with(&first.output_root));
    compare_logical_trees(case, &first, &second).unwrap();
    compare_logical_trees(case, &first, &load_reference_tree(case)).unwrap();
}
~~~

- [ ] **Step 2: Run the tests and verify the red result.**

~~~powershell
cargo test --test html_differential html_oracle_smoke -- --nocapture
cargo test --test html_differential same_case_matches -- --nocapture
~~~

Expected result before the harness is wired: FAIL because the new fixture and
CLI helpers are absent. The default smoke test asserts that the actual binary
runs and emits a complete logical tree for the checked-in basic project. Full
reference parity remains exclusively in the ignored exhaustive test so the
default suite stays green while native HTML parity is incomplete.

- [ ] **Step 3: Implement the exact CLI invocation.**

build_with_ultra_at must use the binary exported by Cargo, never a PATH lookup
or a library call:

~~~rust
let mut command = Command::new(env!("CARGO_BIN_EXE_sphinx-ultra"));
command
    .arg(&source_root)
    .arg(&output_root)
    .args(["-b", "html", "-d"])
    .arg(&cache_root)
    .arg("-q")
    .env_remove("HTTP_PROXY")
    .env_remove("HTTPS_PROXY")
    .env_remove("ALL_PROXY")
    .env("NO_PROXY", "*")
    .env("no_proxy", "*")
    .env("RUST_LOG", "off");
~~~

Create source, output, and cache directories under a fresh TempDir, but use
two different absolute parent directory names for repeat builds. Copy inputs
with exact bytes. Assert cache_root is outside output_root before spawning the
process. The helper records exit status, bounded stdout and stderr, full
logical output paths, and the cache location.

- [ ] **Step 4: Compare complete logical output trees.**

For every path listed by the ledger, load the matching Ultra file. Report a
missing file, unexpected file, byte length mismatch, digest mismatch, warning
stream mismatch, or comparator mismatch as a structured Mismatch record:

~~~rust
pub struct Mismatch {
    pub case_id: String,
    pub logical_path: String,
    pub category: String,
    pub summary: String,
    pub diff: String,
}
~~~

Sort mismatches by case ID and logical path, and cap each diff at
MAX_DIFF_BYTES with a deterministic "[diff truncated at 65536 bytes]" suffix.
Do not stop after the first mismatch.

- [ ] **Step 5: Run default smoke, formatting, and tests.**

~~~powershell
cargo fmt --all -- --check
cargo test --test html_differential html_oracle_smoke -- --nocapture
cargo test --test html_differential same_case_matches -- --nocapture
~~~

Expected result: the smoke and repeat-root tests pass for the active basic
case, all files are represented, cache directories are outside output, and
the same normalized logical tree is produced under both absolute roots.

- [ ] **Step 6: Commit the CLI task.**

~~~powershell
git add tests/html_differential.rs tests/support/html_oracle.rs tests/support/diagnostics.rs
git commit -m "test: run html oracle through ultra cli"
~~~

### Task 8: Add complete source-set equality and expected-failure handling

**Files:**

- Modify: tools/gen_html_oracle.py
- Modify: tools/test_gen_html_oracle.py
- Modify: tests/support/html_oracle.rs
- Modify: tests/html_differential.rs

- [ ] **Step 1: Write source equality tests first.**

~~~rust
#[test]
fn discovered_source_ids_equal_ledger_ids_exactly() {
    let ledger = load_ledger();
    for source_set in ledger.source_sets() {
        let discovered = discover_ids_from_checked_in_source(source_set).unwrap();
        let ledger_ids = ledger.ids_for(source_set);
        assert_eq!(discovered, ledger_ids, "source-set drift in {source_set}");
    }
}

#[test]
fn unsupported_needs_directives_are_not_sent_to_ultra() {
    for case in load_ledger().cases_for("sphinx_needs_doc_tests") {
        if case.expectation == "reference-only" {
            assert_ne!(case.status, "active");
            assert!(case.origin.pytest_nodeid.is_some());
        }
    }
}
~~~

- [ ] **Step 2: Run the tests and verify the red result.**

~~~powershell
cargo test --test html_differential source_ids -- --nocapture
cargo test --test html_differential unsupported_needs -- --nocapture
~~~

Expected result before the checks are implemented: FAIL because discovery and
ledger-set helpers are not defined.

- [ ] **Step 3: Implement exact source-set equality and floors.**

load_ledger must derive expected source identity sets from the checked-in
fixtures and the pinned local-needs checkout, then compare them as
BTreeSet<(source_set, case_id)>. A missing case, extra ledger case, renamed
case, duplicate case, or stale alias is an error. Floors are checked in
addition to exact equality so an accidental replacement by a smaller source
fixture cannot pass after both sides drift together.

- [ ] **Step 4: Implement alias resolution and expectation modes.**

Aliases must contain exactly one canonical_case_id in the same source set.
Resolve an alias to the canonical input and reference but retain the alias in
the ledger and in the completeness count. expected-failure cases run Ultra
only when their status is active; the test requires the recorded exit class
and compares bounded diagnostics to the originating expectation. A
reference-only case is never sent to Ultra. A reference-crash case stores the
reference process failure and is never treated as a passing output comparison.

- [ ] **Step 5: Verify local-needs provenance.**

The generator must fail if a local-needs record has no pytest node ID,
originating test path, assertion or regression expectation digest, profile
pin, or directive capability classification. The Rust exhaustive runner must
print a deterministic SKIP reference-only line containing the node ID and
reason, not a false PASS line.

- [ ] **Step 6: Run and commit corpus completeness.**

~~~powershell
python -m pytest tools/test_gen_html_oracle.py -q
cargo test --test html_differential source_ids -- --nocapture
cargo test --test html_differential unsupported_needs -- --nocapture
git add tools/gen_html_oracle.py tools/test_gen_html_oracle.py tests/support/html_oracle.rs tests/html_differential.rs
git commit -m "test: enforce complete oracle corpus"
~~~

### Task 9: Add the ignored exhaustive differential test

**Files:**

- Modify: tests/html_differential.rs
- Modify: tests/support/html_oracle.rs
- Modify: tests/support/diagnostics.rs

- [ ] **Step 1: Write the aggregate-failure test contract first.**

Add this ignored test with a real failure assertion:

~~~rust
#[test]
#[ignore = "runs every active HTML oracle case and is expected to be red until parity is complete"]
fn exhaustive_html_differential_known_red() {
    let mut mismatches = Vec::new();
    for case in load_ledger().runnable_cases_in_order() {
        match run_and_compare(case) {
            Ok(()) => {}
            Err(mut case_mismatches) => mismatches.append(&mut case_mismatches),
        }
    }
    mismatches.sort_by(|left, right| {
        (&left.case_id, &left.logical_path, &left.category)
            .cmp(&(&right.case_id, &right.logical_path, &right.category))
    });
    assert!(
        mismatches.is_empty(),
        "{} HTML oracle mismatch(es):\n{}",
        mismatches.len(),
        render_capped(&mismatches)
    );
}
~~~

- [ ] **Step 2: Run the ignored test and verify the red result.**

~~~powershell
cargo test --test html_differential exhaustive_html_differential -- --ignored --nocapture
~~~

Expected result while HTML parity is incomplete: the test runs every runnable
case, reports all mismatches in sorted order, caps each diff, and exits
nonzero. It must not stop at the first mismatch or report unsupported
sphinx-needs directives as native Ultra failures.

- [ ] **Step 3: Implement runnable-case selection.**

runnable_cases_in_order includes active cases and active aliases. It includes
an active expected-failure case only to verify its expected failure. It excludes
all four excluded statuses, unsupported-builder, reference-crash, and every
reference-only case. It emits a separate deterministic summary for every
excluded or reference-only record, proving the record was considered.

- [ ] **Step 4: Implement aggregate execution with bounded diagnostics.**

Run each case in a fresh source, output, and cache root. Use the 120-second
per-case deadline, 256 KiB stdout and stderr caps, and 64 KiB per mismatch
diff. Compare the full logical tree, warning stream, exit status, and
structured reference records. Catch spawn failures and timeouts as mismatches
with categories spawn-failure and timeout; never panic or abort the rest of
the corpus.

- [ ] **Step 5: Add the known-red invocation to the documented command set.**

The test remains ignored by default. Its only opt-in command is:

~~~powershell
cargo test --test html_differential exhaustive_html_differential -- --ignored --nocapture
~~~

The command must exit zero only when all runnable cases pass and all expected
failures match their recorded expectation. While native Ultra is incomplete,
the nonzero result is the intended signal.

- [ ] **Step 6: Commit the exhaustive task.**

~~~powershell
git add tests/html_differential.rs tests/support/html_oracle.rs tests/support/diagnostics.rs
git commit -m "test: add exhaustive html differential run"
~~~

### Task 10: Verify regeneration, artifact integrity, and the complete first PR

**Files:**

- Modify: tools/gen_html_oracle.py
- Modify: tools/test_gen_html_oracle.py
- Modify: tests/html_differential.rs
- Modify: tests/support/html_oracle.rs
- Modify: tests/support/diagnostics.rs
- Modify: tests/fixtures/html_oracle/index.json
- Modify: tests/fixtures/html_oracle/inputs/
- Modify: tests/fixtures/html_oracle/refs/
- Modify: tests/fixtures/html_oracle/blobs/
- Modify: tests/fixtures/html_oracle/NOTICE.md

- [ ] **Step 1: Run generator unit and schema tests.**

~~~powershell
python -m pytest tools/test_gen_html_oracle.py -q
~~~

Expected result: all generator, profile, discovery, source-set equality,
network-denial, provenance, deterministic-order, and dedup tests pass.

- [ ] **Step 2: Regenerate the core corpus deterministically.**

~~~powershell
$env:PYTHONNOUSERSITE = "1"
uv run --offline --locked --project tools/oracle_profiles/core python tools/gen_html_oracle.py --config tools/html_oracle_cases.toml --profile core --no-network --out tests/fixtures/html_oracle
git diff --exit-code -- tests/fixtures/html_oracle
~~~

Expected result: the command succeeds and git diff --exit-code is clean after
a committed regeneration. The generator must use atomic writes and must not
change file ordering, line endings, case keys, or hashes between runs.

- [ ] **Step 3: Regenerate the pinned local-needs corpus.**

~~~powershell
$env:PYTHONNOUSERSITE = "1"
$needsRoot = (Resolve-Path $env:SPHINX_NEEDS_ROOT).Path
uv run --offline --locked --project tools/oracle_profiles/local_needs python tools/gen_html_oracle.py --config tools/html_oracle_cases.toml --profile local_needs --needs-root $needsRoot --no-network --out tests/fixtures/html_oracle
git diff --exit-code -- tests/fixtures/html_oracle
~~~

Expected result: the generator verifies sphinx-needs 8.5.0, commit
58bcb59d861da95f2aca79f343e8bae6ec5c1250, subtree tree
958172a89defcec69704f6b9d61e482e7c4e8409, Sphinx 9.1.0, and Docutils
0.21.2 before it writes. The source set contains at least 142 local-needs
cases, and every unsupported directive is ledgered with its originating pytest
expectation rather than sent to Ultra.

- [ ] **Step 4: Run Rust formatting, lint, default tests, and artifact integrity.**

~~~powershell
cargo fmt --all -- --check
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all
cargo test --test html_differential -- --nocapture
~~~

Expected result: formatting, clippy, all default tests, schema checks, path
checks, hash checks, normalizer tests, comparator tests, and smoke tests pass.
The exhaustive test is still listed as ignored by the default run.

- [ ] **Step 5: Run the opt-in exhaustive known-red test.**

~~~powershell
cargo test --test html_differential exhaustive_html_differential -- --ignored --nocapture
~~~

Expected result during this first implementation PR: nonzero with a sorted,
capped aggregate of every remaining mismatch. The output must also list
excluded-network, excluded-plantuml, excluded-external-test-fixture,
unsupported-builder, reference-crash, alias, expected-failure, and
reference-only counts so review can distinguish missing coverage from known
parity work.

- [ ] **Step 6: Verify only the intended implementation files are present.**

~~~powershell
git status --short
git diff --check
git diff --stat HEAD~9..HEAD
~~~

Expected result: every changed path is one of the planned harness, profile,
fixture, or dependency files; git diff --check is clean; no production
builder code, parser code, or unrelated test code changed.

- [ ] **Step 7: Commit the complete first PR state.**

~~~powershell
git add Cargo.toml Cargo.lock tools tests/html_differential.rs tests/support tests/fixtures/html_oracle
git commit -m "test: complete html oracle harness"
~~~

The commit sequence is intentionally reviewable, but the deliverable remains
one coherent first PR because its correctness depends on all layers sharing
the same pinned ledger, provenance, normalization, and execution contract.

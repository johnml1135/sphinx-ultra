---
name: html-oracle
description: Sphinx HTML/needs.json oracle for sphinx-ultra. Use when running the HTML differential, fixing an Ultra mismatch against Sphinx output, regenerating tests/fixtures/html_oracle, or adding oracle cases.
---

# HTML oracle

`tests/fixtures/html_oracle/<profile>/` holds real Sphinx 9.1.0 output
(`core`), and real Sphinx + sphinx-needs 8.5.0 output including `needs.json`
(`local_needs`), for every document-shaped case in the repo's existing
fixtures. Ultra is **1-for-1** with it: a case passes only when Ultra's tree,
warnings and exit class match the reference under the fixed per-path policy in
`tests/support/html_oracle.rs`. The references are the spec; edit Ultra, never
a reference file.

## Fix loop

1. Run the exhaustive diff, optionally narrowed:
   `HTML_ORACLE_FILTER=<substring> cargo test --test html_differential html_oracle_exhaustive -- --ignored --nocapture`.
   It is **red** by design until parity; nonzero exit is the normal signal.
2. Open `target/html-oracle/report.md`. Pick from the *most common
   first-divergence* table: one fix there clears many cases.
3. For a failing case, work in
   `target/html-oracle/runs/<profile>/<source_set>/<case_id>/`: diff
   `expected/` against `actual/`, and `warnings.txt` against
   `actual-warnings.txt`. The report prints the exact filter to rerun only
   that case.
4. Fix Ultra under `src/` with a focused unit or e2e test, then rerun the
   filter until the case is green. Done when that case passes and default
   `cargo test` is still green.

Read categories as a locator: `html-body` means content rendering,
`html-chrome` means the theme/template around the body, `need-field`,
`missing-need` and `extra-need` point at needs extraction and linking, and
`status` or `warning` point at diagnostics.

## Regenerating references

Regenerate only when a pin or the case corpus changes, never to make Ultra
pass. The exact commands are in the module docstring of
`tools/gen_html_oracle.py`. These gotchas aren't obvious from the code:

- Set `PYTHONNOUSERSITE=1`. A user-site Pygments silently changes
  highlighted output.
- `local_needs` requires `--needs-root` pointing to a sphinx-needs checkout
  at the commit and `packages/sphinx-needs` tree pinned in
  `tools/html_oracle_cases.toml`, with that subtree clean. The generator
  refuses anything else.
- sphinx-needs emits random UUIDs and timestamps; the runner's determinism
  shims (recorded as `determinism_shims` in each `index.json`) make output
  reproducible. Two regenerations must be byte-identical; check with
  `git diff --exit-code tests/fixtures/html_oracle`.
- Each profile regenerates independently and replaces only its own subtree.
- In a sandboxed agent, point `UV_CACHE_DIR` and `UV_PYTHON_INSTALL_DIR` at
  git-excluded directories inside the worktree.

## Adding cases

Add the input to its source fixture (e.g. a snippet in
`tests/fixtures/sphinx_doctree_differential.json` via its generator), then
regenerate the profile. The exact source-set counts in
`tools/html_oracle_cases.toml` are deliberate tripwires; update them in the
same commit.

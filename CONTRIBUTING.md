# Contributing to Moirai

## Toolchain

The build toolchain is pinned in `rust-toolchain.toml` to **1.97.0** (with
`clippy` and `rustfmt`). `rustup` selects it automatically inside the repository.

The support floor is separate: the workspace `rust-version` is **1.95**, and the
Python-bindings workflow builds against 1.95.0 so the MSRV claim stays verified.
Do not raise either without a matching change to the other consumers in the
workspace.

## Layout

The repository is a Cargo workspace. Domain crates live at the top level
(`moirai-core`, `moirai-executor`, `moirai-scheduler`, …), the published facade
is `moirai/` (package `moirai-runtime`, library name `moirai`), and
`benchmarks/` and `tests/` are unpublished harnesses.

- `examples/` holds the example sources; they are registered as `[[example]]`
  targets on the `moirai` facade package, so run them with
  `cargo run -p moirai-runtime --example <name>`.
- `docs/book/` is an mdBook deployed to GitHub Pages.
- `docs/adr/` carries one architecture decision per file, indexed by the
  generated `docs/adr/README.md`; `docs/*-checklist.md` carries their execution
  state; `docs/backlog.md` is the work board.

## Branching and pull requests

Work happens on a short-lived branch off `main`, named `<type>/<slug>` matching
the commit type and scope — for example `fix/readme-packaging`,
`feat/route-topology`. Open a pull request into `main`; the Rust workspace gate
runs on every pull request that touches `**/*.rs`, `**/Cargo.toml`,
`**/Cargo.lock`, `**/build.rs`, `.cargo/**`, or `rust-toolchain.toml`.

Commit subjects follow conventional commits: `type(scope): Imperative summary`,
where `scope` is the crate or bounded context (`fix(moirai-core): …`). Types map
to change classes: `feat` is a minor bump, `fix`/`perf`/`test`/`docs`/`refactor`/
`chore`/`build`/`ci` are patch, and a breaking change appends `!` and carries a
`BREAKING CHANGE:` footer.

## The gate

CI (`.github/workflows/rust-ci.yml`) runs exactly this sequence. Run it locally
before pushing:

```bash
cargo fmt --all -- --check
cargo clippy --locked --workspace --all-features --all-targets -- -D warnings
cargo nextest run --locked --workspace --all-features
cargo test --locked --workspace --all-features --doc
RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --all-features --no-deps
```

Notes:

- `cargo nextest` is the sanctioned test runner. `.config/nextest.toml` reports a
  test as slow at 30 seconds and terminates it at 60. Crossing the slow bound is
  a performance defect in the code under test — profile and fix it rather than
  raising the bound or shrinking the workload. A termination is a hang to
  root-cause.
- nextest does not run doctests; `cargo test --doc` is a separate required step.
- Rustdoc is built with `-D warnings`, so a broken intra-doc link fails CI. Most
  published crates carry `#![deny(missing_docs)]`; `moirai-core` and `moirai-gpu`
  are still at `#![warn(missing_docs)]` and should be raised, not lowered.

Benchmark targets must at least compile:

```bash
cargo bench -p moirai-benchmarks --no-run
```

Lock-free code carries `loom` interleaving models. `loom` is a
`[target.'cfg(loom)'.dev-dependencies]` entry, so the models are not compiled by
the default run:

```bash
RUSTFLAGS="--cfg loom" cargo test -p moirai-scheduler --test loom_chase_lev
RUSTFLAGS="--cfg loom" cargo test -p moirai-executor --test loom_wake_handshake
RUSTFLAGS="--cfg loom" cargo test -p moirai-async --test loom_result_slot
```

Other models: `moirai-executor`'s `loom_join_quiescence`, `loom_lifo_slot`, and
`loom_scope_completion`.

## Python bindings

`moirai-python` is not published to crates.io. Its gate
(`.github/workflows/python-ci.yml`) formats and lints the binding crate, runs
`cargo nextest` and doctests against it, checks `moirai-core` under
`--no-default-features --features std`, then builds and smoke-tests a wheel on
Linux, Windows, and macOS.

```bash
py -3.13 -m pip install -e moirai-python
py -3.13 -m unittest discover moirai-python/tests
```

## Tests

- Unit tests live in co-located `#[cfg(test)]` modules; cross-crate integration
  tests live in the `tests/` harness crate; benchmarks live in `benchmarks/`.
- Assertions are value-semantic. `assert!(result.is_ok())` on its own is not a
  test — assert the computed value, or a bound derived from the specification.
- nextest runs tests in parallel processes: no shared mutable global state, no
  fixed ports, no wall-clock sleeps for synchronization. Use channels, barriers,
  or event gates.
- Numeric tolerances are derived, not tuned until green.

## Documentation

Documentation ships in the same change as the code it describes.

- Public API contracts belong in Rustdoc, with `# Errors`, `# Panics`, and
  `# Safety` sections where they apply and a runnable `# Examples` doctest.
- Architecture decisions belong in `docs/adr/` (one record per file) and the
  per-ADR checklists.
- Externally observable changes go in `CHANGELOG.md` under `## [Unreleased]`,
  in Keep a Changelog format. Do not bump versions in a feature PR; the version
  is assigned at release.
- The README is the crates.io landing page for the facade and every crate that
  inherits `readme.workspace`. Keep it factual: a claim in it must resolve to
  code in this repository.

## Unsafe code

The facade crate contains no `unsafe`. Where lower-level crates need it, isolate
it behind a safe API and precede every `unsafe` block with a `// SAFETY:`
comment naming the invariant relied on. Reachable unsafe paths are expected to
run clean under `cargo +nightly miri test` for the owning crate; where miri
cannot execute the code (platform I/O, SIMD, FFI, GPU), say so and substitute
targeted tests.

## Releases

Releases are tag-driven and publish through OIDC trusted publishing — there is
no stored registry token.

- Rust crates: a GitHub Release tagged `crate-<package>-v<version>` packages,
  verifies, and publishes that package. `workflow_dispatch` validates a package
  without publishing.
- Python wheels: a GitHub Release tagged `moirai-python-v<version>` attaches
  CPython 3.10–3.13 wheels for Linux, Windows, and macOS and publishes them to
  PyPI.

## License

Contributions are dual licensed under [Apache-2.0](LICENSE-APACHE) and
[MIT](LICENSE-MIT). By submitting a contribution you agree it may be
distributed under those terms.

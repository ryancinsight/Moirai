# Milestone ladder ([ADR 0021](../docs/adr/0021-release-scope-1-0.md))
<a id="M-core-execution"></a> **M-core-execution** (0.12.0) needs: (none) — unified scheduler and parallel execution.
- exit: run basic usage example (`cargo run -p moirai --example basic_usage --locked --all-features` exits successfully).
- exit: run book task spawning example (`cargo run -p moirai --example book_task_spawning --locked --all-features` exits successfully).
- exit: run book parallel reduce example (`cargo run -p moirai --example book_parallel_reduce --locked --all-features` exits successfully).
- exit: pass integration tests (`cargo test -p moirai-tests --locked --all-features` passes).
- standing: no open correctness item.
<a id="M-async-io-transport"></a> **M-async-io-transport** (0.13.0) needs: M-core-execution — async I/O and transport.
- exit: pass async I/O compatibility tests (`cargo test -p moirai-async --locked --all-features` passes).
- exit: pass iter and parallel core tests (`cargo test -p moirai-iter --locked --all-features` and `cargo test -p moirai-parallel --locked --all-features` pass).
- exit: pass transport and QUIC tests (`cargo test -p moirai-transport --locked --all-features` and `cargo test -p moirai-quic --locked --all-features` pass).
- standing: no open correctness item.
<a id="M-1.0"></a> **M-1.0** (1.0.0) needs: M-async-io-transport — stable unified execution release (authority-gated).
- standing: `cargo fmt --all -- --check`, `cargo clippy --workspace --locked --all-targets --all-features -- -D warnings`, `cargo test --workspace --locked --all-features`, `cargo test --workspace --locked --doc --all-features`, and `cargo doc --workspace --locked --no-deps --all-features` pass; SemVer and publish dry-run gates required for declared releases pass; README and rustdoc are current; no open correctness item.

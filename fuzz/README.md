# Moirai fuzz targets

libFuzzer targets for parsers crossing trust boundaries
(MOI-AUDIT-SEC-001). This tree is excluded from the workspace: normal
and CI builds never compile it.

## Run

```
rustup toolchain install nightly-2026-08-01
cargo install cargo-fuzz --version 0.13.2 --locked
cargo +nightly-2026-08-01 fuzz run http_response seeds/http_response
cargo +nightly-2026-08-01 fuzz run ipc_layout seeds/ipc_layout
```

Targets:

- `http_response` — drives `moirai_http::codec::read_response` with
  arbitrary bytes under a 1 MiB slowloris budget. Any panic, hang, or
  allocation blowup on hostile input is a defect; rejection paths must
  be typed `io::Error`s.
- `ipc_layout` - throws arbitrary element-size and capacity pairs at the pure
  shared-queue layout check (`layout_total`) behind a cfg(fuzzing) accessor.
  Overflowing or degenerate sizes must remain typed `IpcError`s, never panics.

Pull-request verification executes every committed seed once for both
targets. The weekly and manually dispatched jobs run both mutation campaigns
concurrently for 180 seconds per target; generated corpus entries remain in
the runner's temporary output directory.

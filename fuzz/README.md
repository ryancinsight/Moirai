# Moirai fuzz targets

libFuzzer targets for parsers crossing trust boundaries
(MOI-AUDIT-SEC-001). This tree is excluded from the workspace: normal
and CI builds never compile it.

## Run

```
rustup toolchain install nightly
cargo install cargo-fuzz
cargo +nightly fuzz run http_response -- fuzz/seeds
```

Targets:

- `http_response` — drives `moirai_http::codec::read_response` with
  arbitrary bytes under a 1 MiB slowloris budget. Any panic, hang, or
  allocation blowup on hostile input is a defect; rejection paths must
  be typed `io::Error`s.
- `ipc_header` - throws peer-controlled header bytes and size pairs at
  the pure shared-queue checks (`parse_header_capacity`, `layout_total`)
  behind a cfg(fuzzing) accessor. Short or malformed headers must remain
  typed `IpcError`s, never panics.


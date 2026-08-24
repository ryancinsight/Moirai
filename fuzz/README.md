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

Coverage note: the IPC shared-memory validator (`layout_for`,
`SharedQueue::open`) is not yet fuzzable because its validation logic is
private and inseparable from OS shared-memory resources; that target is
blocked on a public pure-validation seam and tracked on the SEC-001
board item.

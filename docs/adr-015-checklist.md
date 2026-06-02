# ADR-015 Implementation Checklist: Native HTTP/S3 Transport Stack

Concrete contracts and tasks to replace consus's Tokio/rusoto/reqwest S3 transport with a
pure-Moirai stack built over the existing reactor-bound async sockets (ADR-014/006/013).
Status: **Proposed** — P1 opens only after ADR-015 sign-off. Each phase leaves the tree green.

## 0. Foundation audit (no new code) — `[arch]`

- [ ] Confirm `moirai-async::net::AsyncTcpStream` connect/read/write works over the active
  `IoReactor` under `Moirai::block_on` on **Linux** (epoll) and **Windows** (IOCP).
- [ ] Confirm `moirai-async::io::{AsyncRead, AsyncWrite}` trait surface is stable and
  cancellation-safe (ADR-006-async contracts hold).
- [ ] Spike: loopback echo + a `rustls` client/server handshake over a `moirai-async`
  TCP pair. Exit gate: green TLS-roundtrip integration test on Linux + Windows.

## 1. `moirai-tls` — `[minor]`

Contract: drive `rustls` (sans-I/O) over a `moirai-async` `AsyncTcpStream`; Moirai owns only
the byte pump and the handshake-completion future. No hand-rolled cryptography.

- [ ] `TlsConnector` (sealed `Transport` strategy) wrapping `rustls::ClientConfig`; roots via
  `rustls-platform-verifier` or `webpki-roots`.
- [ ] `TlsStream<S: AsyncRead + AsyncWrite>` implementing `AsyncRead + AsyncWrite`; pump
  `rustls` `read_tls`/`write_tls` ↔ socket, `process_new_packets` ↔ plaintext buffers.
- [ ] Handshake future cancellation-safe (drop mid-handshake leaks no buffer; ADR-006).
- [ ] **Tests**: loopback handshake vs a rustls server; plaintext-roundtrip **differential vs
  `tokio-rustls`** (dev-dep); adversarial cert validation — expired / wrong-host /
  untrusted-root must **fail closed** (value-semantic, not `is_err()`-only).
- [ ] No `dyn` on the hot path; `#![forbid(unsafe_code)]` or `// SAFETY:` + miri on any unsafe.

## 2. `moirai-http` — `[minor]`

Contract: HTTP/1.1 client over `moirai-tls`/`moirai-net`, reusing `http` + `httparse`
(sans-I/O). HTTP/2 out of scope.

- [ ] Request serialization from `http::Request`; response head parse via `httparse`.
- [ ] Body framing: Content-Length **and** chunked transfer decoding (incl. trailers).
- [ ] Bounded-capacity keep-alive connection pool (no unbounded queues; idle-eviction timer).
- [ ] Redirect handling (3xx, capped hops) and per-request deadline via Moirai timer.
- [ ] **Tests**: local HTTP/1.1 test server; property tests on chunked vs Content-Length
  framing and pool connection reuse; **differential — identical GET via `reqwest` vs
  `moirai-http` → byte-identical status/headers/body**; slow-loris timeout; server close
  mid-pool; 100-continue.

## 3. consus S3 client on `moirai-http` — `[minor]` (in **consus**, NOT moirai)

Contract: rebuild consus's existing S3 backend (`consus-io` `io/async_io/s3.rs`,
`consus-zarr` `S3Store`) on `moirai-http`. Moirai stays AWS-agnostic; SigV4/GetObject
live here. `aws-sigv4` + `quick-xml` are **consus** deps, never moirai's.

- [ ] SigV4 signing via `aws-sigv4` (or direct HMAC-SHA256 canonical-request impl).
- [ ] Credential resolution: env vars + `~/.aws/credentials`. Region/endpoint/bucket/key as
  validating newtypes (no primitive obsession).
- [ ] `GetObject` with `Range` header + `HeadObject` over `moirai-http`; S3 error-XML decode
  via `quick-xml` (preserve distinct error modes — `NoSuchKey` etc. — no stringly catch-all).
- [ ] `S3Reader: AsyncReadAt` adapter (drop-in for the rusoto one).
- [ ] consus-io/zarr feature axis: `s3-moirai` (dep: `moirai-http`) vs `s3-tokio` (legacy
  rusoto/reqwest), both implementing `AsyncReadAt`; no public storage-API change.
- [ ] consus-hdf5 async tests drop `#[tokio::test]` → `Moirai::block_on` (format layer is
  already runtime-agnostic; removes the tokio dev-dep there).
- [ ] **Tests**: **SigV4 known-answer tests** (AWS published canonical-request → string-to-sign
  → signature vectors); **differential vs `rusoto_s3` against local MinIO/`s3mock` →
  byte-identical `GetObject(Range)` / `HeadObject`**; existing consus S3 property/integration
  tests green on **both** backends.

## 4. Comparative benchmarks — `[minor]` (consus)

- [ ] criterion harness: ranged-GET throughput, p50/p99 latency, CPU-time/req, allocations —
  `s3-moirai` (consus S3 on `moirai-http`) vs `s3-tokio` (`rusoto_s3`), against (a) localhost
  MinIO (RTT≈0) and (b) `toxiproxy` latency-injected (realistic RTT).
- [ ] Record results; regression → profile/optimize (pool warm-up, buffer reuse, vectored
  writes), **not** revert (optimization farsight). Expect parity (network is latency-bound).

## 5. Default flip + Tokio removal — `[major]` (consus)

- [ ] On parity: flip consus default to `s3-moirai`; demote rusoto/reqwest to legacy/optional;
  remove Tokio from consus production tree (Tokio = benchmark/reference only, per ADR-001/013).

## Gates (every phase)

- [ ] `cargo fmt --check` → `cargo clippy --all-targets --all-features -D warnings` →
  `cargo nextest run` → `cargo doc --no-deps`.
- [ ] `cargo deny check` on new deps — moirai: rustls, http, httparse; consus: aws-sigv4,
  quick-xml (AWS/XML deps stay out of moirai's tree); MSRV pinned.
- [ ] `cargo miri test` for any crate with unsafe; `cargo-semver-checks` before each `[minor]`.
- [ ] CHANGELOG + version bump committed atomically with each phase.

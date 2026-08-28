# ADR-015 Implementation Checklist: Native HTTP/S3 Transport Stack

Concrete contracts and tasks to replace consus's Tokio/rusoto/reqwest S3 transport with a
pure-Moirai stack built over the existing reactor-bound async sockets (ADR-014/006/013).
**Status (2026-06-02): P0–P4 done; P5 partial.**
- P0 (moirai bcf3ed1): net spike incl. Windows/IOCP. P1 `moirai-tls` (rustls handshake +
  fail-closed). P2 `moirai-http` (framing/keep-alive/timeout). P3 `consus-io` `s3-moirai`
  (SigV4 KAT vs AWS vector + mock round-trip; production tree tokio-free).
- **P4 correctness: DONE** — rusoto↔moirai **byte-identical differential** over GetObject(Range)
  + HeadObject via an in-process mock (no Docker), passing locally (consus 84139fb). A MinIO
  CI job (`s3-minio` in `.github/workflows/ci.yml`) exercises moirai-s3 (path-style+SigV4)
  against a live server via the env-gated `moirai_s3_real_endpoint` test.
- **P4 comparative performance bench (criterion vs rusoto, localhost + toxiproxy RTT):
  REMAINING** — best run on CI with MinIO; not yet wired.
- **P5 mostly done:** consus-hdf5 production async path is tokio-free (`async-io` →
  `consus-io/async-traits`). consus-io gained a full `S3Client` (GET/PUT/DELETE/HEAD/
  ListObjectsV2 with SigV4 + quick-xml; verified put→get→head→range→list→delete round-trip).
  consus-zarr `S3MoiraiStore` implements the sync `Store` trait over it via `block_on` —
  a **complete tokio-free object-store path** (consus-zarr → consus-io S3Client → moirai-http/
  tls/sockets), verified end-to-end vs an in-process mock (consus b662382). Behind `s3-moirai`,
  alongside the legacy rusoto `s3`. **REMAINING:** comparative perf bench (criterion vs rusoto;
  CI/MinIO) and the eventual default flip / rusoto+tokio removal once s3-moirai is battle-tested.
Each phase leaves the tree green.

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
- [x] Bounded-capacity keep-alive connection pool with access-triggered idle
  eviction. Eviction runs while the pool is accessed rather than through a
  background timer task, preserving the bounded-state contract without adding
  worker lifecycle state.
- [x] Redirect handling for 301/302/303/307/308 with capped hops, RFC 3986
  relative-reference resolution, destination-aware header filtering, and one
  Moirai timer deadline across the complete logical request.
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

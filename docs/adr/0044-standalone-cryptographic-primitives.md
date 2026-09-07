# ADR 0044: Standalone cryptographic primitives

Status: Accepted

Date: 2026-09-07

Driver: [MOI-CRYPTO-2026-09-07](../backlog.md#MOI-CRYPTO-2026-09-07),
[Metis crypto](../../metis/backlog.md#METIS-CRYPTO-001).

## Context

Metis needs SHA-256, HMAC-SHA256 and fixed-width authentication comparison for
capability and audit contracts. Its local implementation duplicated the
RustCrypto algorithms already used by Moirai's TLS provider. Depending on the
full provider for protocol hashing would also compile TLS code that a consumer
does not need.

## Decision

`moirai-crypto` publishes `Sha256`, `sha256`, `hmac_sha256` and
`constant_time_eq_32` at its crate root. The APIs are always available. The
`provider` feature owns the existing rustls implementation and is enabled by
default; disabling it leaves only the standalone RustCrypto hash and HMAC
dependencies. This keeps the implementation in one Atlas provider while
allowing protocol-only consumers to avoid a TLS dependency.

The primitives use the existing `sha2` and `hmac` dependencies. HMAC accepts
keys of every length, so construction's unreachable error is guarded by an
invariant assertion. The fixed-width comparison accumulates XOR differences
over all bytes and documents that source-level constant work is not a machine
code timing proof. CRC-32 remains in the protocol owner because it detects
framing corruption and is not an authentication primitive.

Metis removes its SHA-256, HMAC and comparison implementation and imports the
Moirai surface. The upstream addition is an additive minor API; the Metis
removal is a breaking migration documented in its own decision record.

## Alternatives

- Keep the Metis copy: rejected because two implementations can diverge in
  vectors, audit behavior and constant-time review.
- Add direct `sha2` and `hmac` dependencies to every consumer: rejected because
  it forks Atlas's provider ownership and repeats dependency policy.
- Create a second crypto crate: rejected because a feature boundary on the
  existing provider supplies the required no-TLS build without another
  publishable package or dependency edge.

## Verification

The standalone module carries FIPS 180-4 SHA-256 vectors, RFC 4231 HMAC vector,
streaming padding-boundary cases and equal/different comparison cases. The
provider and standalone feature builds run separate warning-denied Clippy and
native test checks; the standalone no-default build also compiles for
`wasm32-unknown-unknown` without the TLS provider. The provider's existing
browser build remains outside this contract because its current `getrandom`
configuration intentionally rejects an unconfigured WebAssembly host.
Release assembly inspection checks that the fixed-width comparison has no
input-dependent loop exit; it does not establish a hardware timing proof.

## Limits

This decision does not claim formal side-channel resistance for every compiler,
CPU or runtime. Consumers must keep keys out of logs and bind verification to
their own session and origin policy. A future machine-code or platform review
can replace the comparison implementation behind the same contract if its
evidence falsifies the current code-generation check.

# ADR 0082: Recovery state observation

Date: 2026-10-14
Status: Accepted
Change class: [patch], diagnostic investigation within unreleased workspace 0.10.0.

## Specification and rejection criteria

Continue MOI-QUIC-RECOVERY-001 with the same 108 cases per stock policy,
30 competing experiments, seeded network conditions, exact message bytes,
mutual TLS, acknowledged FIN, five-second operations and 60-second pairs.
The independent uninstrumented original CUBIC job remains authoritative.
Diagnostic observations cannot certify the pristine engine's performance.

Observe actual per-space keys, generated/consumed CRYPTO offsets, queued and
sent CRYPTO ranges, received ACK ranges, outstanding packet bytes, connection
and stream credit, and timer deadlines/selected PTO space. For this non-migrating
workload, the sum of all outstanding congestion-accounted packet sizes equals
path flight bytes. CRYPTO ranges are metadata; never retain plaintext, keys,
tokens or certificate secrets. Queued/sent frame lengths include retransmitted
copies and do not equal unique delivered bytes.

A trace transition is an observation at a driver boundary, not a continuous
state-duration proof. Retain every handshake state and Initial/Handshake ACK
and CRYPTO event. Established snapshots retain exact first/last and extrema
within the existing one-second observation intervals; preserve PTO snapshots.
No time fraction is inferred from event counts or interval extrema. Require
actual child capture in both preflight peers and reject absent/malformed state.

## Architecture and alternatives

Pinned quinn-proto 0.11.19 already supports aggregate tracing. Its qlog feature
reports flight metrics but omits frame contents and would add dependencies and
a public configuration surface. Existing traces omit early ACK/CRYPTO frame
metadata and complete flight/credit state. Read-only instrumentation at driver
poll, ACK processing and CRYPTO reception supplies these missing observations.

A reproducible script verifies the cached crate against its registry checksum,
copies it and the committed workspace into task-owned build directories, and
applies a checked-in instrumentation patch. Only added read-only observer calls,
one private leaf module and a read-only stream accessor are permitted. No
algorithm statement is removed or replaced. The canonical workspace and locks
remain unchanged. The diagnostic copy's lock removes only the patched engine's
registry source/checksum; all package versions and other identities must match.
Record the patch, pristine/observed source hashes, build lock and binary hashes.
Never edit the registry cache or ship the instrumentation as a production fork.

The example's existing opt-in tracing layer owns aggregation. Namespace-local
forwarding and real authenticated preflights remain mandatory. Keep numerical
state parsing and audit separate from transport behavior and preserve preceding
archives. A candidate recovery correction requires causal evidence, a supported
production path, and a fresh full workload/fairness/latency/queue comparison.

## Verification plan

Test deterministic patch provenance and reject modified source/removal patches.
Use real authenticated peers to verify cross-space flight sums, monotonic
CRYPTO offsets/packet counters, ACK ranges and complete child observations.
Exercise disabled capture and reject corrupt state. Then run all six matched
Linux shards and the unchanged independent original gate. Document unsuccessful
experiments without assembling acceptance across sources or changing deadlines.

## Causal handshake experiment and candidate boundary

All six early server failures in the completed Linux competing shards retain
634 queued client Handshake CRYPTO bytes, zero Handshake flight and Data PTOs.
The server consumes 2246 of 2880 generated client Handshake bytes and remains
in the TLS handshake. Pending CRYPTO leaves flight when declared lost. Normal
retransmission remains subject to the shared congestion window; a Data-space
probe does not grant a Handshake-space probe exception. The observed combination
of queued CRYPTO, flight above the window and Data-only PTO selection produces
a circular wait: the server cannot process Data until the missing handshake
bytes arrive, while those bytes cannot pass normal congestion admission.

A real UDP relay forwards each datagram unchanged except one of the first three
client datagrams containing Handshake packets. It parses only unprotected QUIC
v1 packet layout, retains full 16-stream/128-message/65560-byte exchanges and
original deadlines, and performs no shaping. Three repetitions per stock policy
retain 27 cases per engine. The observed baseline completes 17/27; a focused
experimental engine patch completes 27/27. Every baseline failure retains the
queued-CRYPTO/Data-PTO mechanism. Packet batching and TLS entropy vary; this is
a matched fault rule rather than an identical encrypted packet tape or a rate
comparison. Both source/patch identities and actual frame metadata are retained.
The preceding nine-case experiments and an unsuccessful result-file recorder
remain separate from these complete repetitions.

The candidate includes queued CRYPTO in PTO eligibility and requires client
handshake confirmation before Application Data PTO selection, consistent with
[RFC 9002 section 6.2.1](https://www.rfc-editor.org/rfc/rfc9002.html#section-6.2.1).
[RFC 9001 section 4.1.2](https://www.rfc-editor.org/rfc/rfc9001.html#section-4.1.2)
distinguishes local TLS completion from confirmation of the peer handshake.
Both changes operate inside recovery; neither admits unauthenticated server
connections or changes an application deadline. All 23 existing native QUIC
contract tests pass against the isolated candidate.

This candidate is an experimental patch artifact, not a production dependency
or accepted provider correction. Once handshake keys and queued CRYPTO are
absent, both modified guards reduce to the original guards; the established
loss throughput failure remains a separate problem. Full candidate-domain Linux
policy/fairness/latency/queue measurement, native platform checks and a supported
reviewed production dependency remain required before adoption. The canonical
provider remains on the pristine engine and the original promotion gate remains
authoritative. Do not splice candidate cases into a baseline inventory.

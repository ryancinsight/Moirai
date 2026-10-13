# ADR 0075: QUIC multiplexed native sessions

Date: 2026-10-13
Status: Accepted

Decision scope: the user-requested native transport increment.
Change class: [minor]. Target: workspace 0.9.0.

## Intent and boundaries

Add a reliable, authenticated, multiplexed UDP transport to Moirai, consumed by
Metis for native remote sessions. HTTP/2 multiplexes over TCP; HTTP/3 maps HTTP
semantics onto QUIC. Metis's binary session protocol needs neither HTTP mapping
nor an unreliable datagram protocol. One bidirectional QUIC stream carries one
ordered session; concurrent sessions share a connection without sharing replay
or capability state. Process pipes continue to provide local process isolation.

## Selected architecture

`moirai-pal` owns cancellable, shared socket readiness. `moirai-quic`
owns Quinn integration, Moirai task spawning/timers, TLS configuration, stream
admission, and bounded message framing. Consumers import only the provider's
types. The async crate already depends on moirai-transport, so placing QUIC in that
crate introduces a dependency cycle. A sibling provider above moirai-async
preserves the same topology as moirai-tls and moirai-http. Quinn uses Tokio
synchronization primitives, with runtime/socket features disabled. Moirai
drives every task, timer and socket. Quinn owns RFC 9000
loss recovery, congestion control, stream multiplexing and packet protection.

Quinn 0.11 requires its `rustls-ring` feature for the upstream TLS/Retry bridge.
The optional QUIC feature therefore adds statically linked ring native code;
the existing TCP TLS provider remains RustCrypto. Reimplementing packet crypto
or forking Quinn to preserve a dependency slogan introduces a second protocol
owner and a larger security audit. Native builds require ring's C/assembly build
tools; operating-system socket libraries remain platform dependencies. No C
crypto dependency is introduced into builds without the QUIC feature.

## Contracts and proof obligations

- Only TLS 1.3 connections with the caller's ALPN and validated certificate
  chain/name are admitted. Production trust roots and keys are caller-owned;
  no certificate-verification bypass or automatic trust installation exists.
- Early data and unreliable datagrams are disabled. An application operation
  executes only after the authenticated handshake. Transport loss recovery
  never repeats application dispatch.
- Positive finite limits bound handshake, stream admission and frame I/O.
  For maximum frame size B and concurrent streams S, stream receive windows
  sum to at most S(B+4); the connection receive window enforces the same bound.
  Connection count and pending handshakes are separately bounded.
- Each message is `u32` big-endian length followed by exactly that many bytes.
  A declared length greater than B is rejected before allocating its payload.
  EOF is clean only before a new prefix; partial prefixes/payloads are errors.
- Dropping or timing out a partial frame operation resets/stops that stream.
  An ambiguous write is never retried on the same framing state. Other streams
  retain their own framing and application state.
- `finish` queues FIN; `finish_and_wait` awaits transport acknowledgement before
  connection teardown. Metis consumes correlated replies/events and peer FIN
  before closing the connection; transport ACK alone does not confirm dispatch.
- All writable waiters share one reactor registration per socket/interest;
  readiness wakes every admitted waiter. Cancellation removes registrations
  before the last descriptor owner closes; one waiter cannot displace another.

These are native transport contracts, not browser HTTP/3/WebTransport support.
UDP-blocked networks require an explicit existing TCP endpoint selection;
there is no silent downgrade or automatic application replay.

## Alternatives and verification

Raw UDP multiplexing requires a second reliable, authenticated congestion and
retransmission protocol. HTTP/2 retains TCP's cross-stream loss coupling.
HTTP/3/WebTransport requires browser negotiation, HTTP authority, QPACK and a
separate application mapping; it is outside this native binary-session slice.

Acceptance uses real UDP sockets, upstream Quinn peers, certificate rejection,
multiple streams and connections, exact-size payloads, malformed prefixes,
timeout/drop reset and independent-stream progress. Metis runs its real trusted
backend handshake, computation, capability rejection and replay checks over the
same provider. Property checks compare frame contents against input bytes;
warning-denying checks and semver checks validate the additive surfaces.

References: [RFC 9000](https://www.rfc-editor.org/rfc/rfc9000),
[RFC 9001](https://www.rfc-editor.org/rfc/rfc9001),
[RFC 9114](https://www.rfc-editor.org/rfc/rfc9114),
[Quinn runtime interface](https://docs.rs/quinn/0.11.12/quinn/trait.Runtime.html).

## Typed session closure

The shared native session seam identifies a validated WebSocket close through
`WebSocketPeerClosed`, preserving its status/reason and keeping truncated input
as an error. Close validation returns borrowed fields; only a received peer
close allocates its owned diagnostic cause. No diagnostic string establishes
protocol state.

# ADR 0046: Bounded WebSocket service substrate

Status: Accepted

Date: 2026-09-07

Driver: [MOI-HTTP-WS-2026-09-07](../backlog.md#MOI-HTTP-WS-2026-09-07),
[Metis async](../../metis/backlog.md#METIS-ASYNC-001).

## Context

Metis has a bounded browser WebSocket client and a framed asynchronous IPC
contract, but Moirai has no native HTTP upgrade or WebSocket service. Adding a
third-party WebSocket runtime would duplicate the existing Moirai readiness,
timers and resource ownership and would put an unbounded protocol queue next to
the bounded PAL. The browser client also currently returns while its JavaScript
WebSocket is still connecting, so the first frame can race the OPEN event.

The service is a trust boundary. HTTP headers and WebSocket frames are hostile
input, and a browser `Origin` header identifies the page origin without proving
the operating-system window or session. The consumer must perform its own
origin, session and capability checks after the protocol layer accepts the
upgrade.

## Decision

Moirai owns a small, pure-Rust service substrate in `moirai-http` over the
existing `moirai_async::net::TcpListener`/`TcpStream` facades. The substrate is
message-oriented because consumers such as Metis carry one complete binary IPC
frame per WebSocket message. Its public API consists of a bounded
`WebSocketConfig`, validated `HttpRequestHead`, `accept_websocket`,
`accept_websocket_with_validator`, and a `WebSocketStream<S>` with
`recv_message`, `send_binary` and `close` operations. The validator runs after
protocol parsing and before the `101` response, so a consumer can reject a
browser origin without acknowledging an unauthorized socket.
The stream is generic over any Moirai `AsyncRead + AsyncWrite` connection, so a
future `moirai_tls::TlsAcceptor` can be added without changing the Metis seam.

The HTTP parser admits only `GET`/`HTTP/1.1` upgrade requests, caps the request
head and header count, rejects request bodies and ambiguous framing, requires a
non-empty `Host`, and requires exactly one valid `Upgrade: websocket`,
`Connection: Upgrade`, `Sec-WebSocket-Version: 13` and `Sec-WebSocket-Key`.
The optional `Origin` value is retained for the consumer's policy check; the
protocol layer does not invent window or session identity. A valid request
receives the exact `101 Switching Protocols` response and no body. Consumers
that use `accept_websocket_with_validator` perform their origin check before
that response is written.

The WebSocket codec accepts masked client binary frames, rejects unmasked,
fragmented, reserved-bit and invalid control frames, rejects non-minimal
126/127 length encodings, and bounds every advertised length before
allocation. It answers ping with pong and treats close as a terminal state.
Text and continuation data are rejected because the Metis consumer's contract
is binary message transport. Every read, write and handshake is wrapped by a
caller-supplied finite deadline. A frame timeout or partial I/O error also
terminalizes the stream so a caller cannot resume at an ambiguous byte
position. The codec never spawns a task or retains a queue; cancellation drops
the stream and its socket.

The RFC 6455 accept value uses SHA-1 followed by Base64 solely for protocol
interoperability. It is explicitly not an authentication primitive. The
implementation is kept in Moirai's crypto provider and covered by the RFC
vector; Metis continues to use HMAC-SHA256 for authority.

The WASM PAL gains a cancellation-safe OPEN readiness future. A browser
transport must await this future before sending, and dropping it removes its
OPEN waiter. The existing bounded message queue and callback guards remain the
single owner of browser WebSocket resources.

Revision 2026-09-07: cancellation and close callbacks extract their waiters
while holding the state mutex, then wake them after the guard is dropped. This
keeps a waker from re-entering the WebSocket state machine through a lock held
by the callback or destructor.

## Alternatives

Using a third-party WebSocket crate would add a second runtime/resource model
and violate the Atlas first-party ownership decision. Exposing raw byte reads
would lose message boundaries and make every consumer reimplement frame
validation. Treating the browser `Origin` as an OS identity would turn an
untrusted header into authority. Sending before OPEN would race the browser
state machine and produce a transport failure that cannot be retried safely.

## Threat model and limits

The request head and frames are attacker-controlled and can attempt memory
exhaustion, parser desynchronization, replay, cross-origin use, masking abuse,
or indefinite waits. Header/message/connection bounds, strict grammar,
finite deadlines, and terminal protocol errors address those classes. Metis
must still validate `Origin`, bind a trusted window/session context, reject
replayed IPC sequences, and verify its HMAC capability after upgrade. Plain
`ws://` loopback is suitable for local conformance only; production browser
confidentiality requires a TLS acceptor or a private OS transport. The current
increment does not provide HTTP routing, static asset serving, TLS server
authentication, or inbound WASM networking.

## Verification

Provider tests cover RFC handshake vectors, partial reads/writes, malformed
headers, masking, length boundaries, control frames, ping/pong, close,
deadline and owned-stream cancellation. The Metis consumer must add an
asynchronous server/session adapter with replay, authority and loopback
connection-accounting tests. Native tests and warning-denied Clippy pass on the
pinned provider; `moirai-pal` compiles and lints for `wasm32-unknown-unknown`.
The `moirai-http` WASM check remains blocked by its existing native client
dependency graph (`getrandom`/`socket2`), and the manual records a real
loopback browser trace only after the complete service path is connected.

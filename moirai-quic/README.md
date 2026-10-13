# moirai-quic

Native authenticated QUIC sessions over Moirai's task spawner, timer and PAL
socket readiness. Quinn supplies RFC 9000 loss recovery, congestion control and
stream multiplexing. No Tokio or secondary runtime is constructed.
Quinn uses Tokio synchronization primitives internally; its Tokio runtime and
socket features remain disabled. Moirai owns all task, timer and socket driving.

`QuicEndpoint` binds before advertising readiness. Clients supply trusted roots
and the certificate's server name; servers supply their certificate chain/key
and optional client trust roots for required mutual TLS. Both negotiate one
explicit ALPN, TLS 1.3 and no early data. Trust/authorization remains caller policy.

`QuicConfig` validates positive message, stream, connection and handshake budgets
and finite operation/idle deadlines. The per-connection receive/send window is
`max_streams * (max_frame_bytes + 4)`. A `QuicStream` carries four-byte big-endian
length-prefixed messages, checks size before payload allocation, distinguishes
empty messages from clean EOF and aborts framing on timeout, error or future drop.
Commands use reliable streams; unreliable datagrams are disabled.

Keep the supplied Moirai executor running until endpoints close and drain.
The facade implements the fallible `TaskSpawner` trait, so an `Arc<Moirai>` and
an `Arc<HybridExecutor>` both drive the provider without another pool.
Concurrent bidirectional streams remain independent, but messages in one stream
retain ordering. A stream dropped without FIN resets its incomplete output.
`finish` queues FIN; `finish_and_wait` awaits acknowledgement of buffered output
and FIN before connection teardown. An acknowledgement confirms transport
delivery; decoded application replies establish the operation's result.
Transport selection is explicit; no command replay or implicit TCP downgrade
occurs. Browser HTTP/3/WebTransport is a separate application contract.

The Unix/Windows facade always includes QUIC, even with default features
disabled. Import `moirai::quic` or depend on `moirai-quic` directly; no QUIC
feature flag exists. Native builds require ring's statically linked C/assembly
code and its build tools. The TCP TLS provider uses RustCrypto. System socket
libraries remain platform dependencies; browser builds exclude native QUIC.

[ADR 0075](../docs/adr/0075-quic-multiplexed-sessions.md) records the contract.

The `quic_load` example and `scripts/quic_profiles.py` implement ADR 0075's
controlled performance contract: mutual TLS, complete payload validation,
acknowledged shutdown, integer RTT/rate evidence and task-owned Linux netem
namespaces. Public fixture credentials are for these controlled probes only.
Reports retain every sample, qdisc counters and the tested source/runner.

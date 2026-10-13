# ADR 0078: Applied UDP socket configuration

Date: 2026-10-13
Status: Accepted
Change class: [minor], with a pre-1.0 breaking configuration migration.
Target: workspace 0.9.0; moirai-quic 0.1.0.

## Intent and selected architecture

The async UDP facade exposes `buffer_size` and `multicast`, but neither affects
the socket. Replace those fields with one PAL-owned configuration contract used
by async UDP and authenticated QUIC. PAL owns operating-system options; async
UDP owns operation counters; QUIC owns TLS, framing and protocol resource limits.
The dependency graph remains directed from transport providers to PAL.

`UdpBufferConfig` expresses optional independent receive/send requests.
`UdpConfig` adds broadcast and an optional `MulticastConfig`, whose variants
carry the matching IP group and interface. Bind applies options before returning
the owned socket. Invalid input and failed socket operations return errors and
drop the unpublished socket. No option is accepted without applying its native
operation. Default buffer requests are absent, preserving kernel defaults.

QUIC consumes only `UdpBufferConfig`: broadcast and multicast do not belong to
its authenticated point-to-point contract. Its endpoint reports effective
buffer sizes captured after native configuration. Metis uses this same builder
surface and retains its required QUIC provider dependency.

## Invariants and rejection criteria

For each requested buffer size b, the type proves b > 0 and validation proves
b <= i32::MAX, the signed native socket-option representation. socket2 casts
usize to the native signed integer, so validation precedes every setter.
`None` performs no buffer setter. The bind result is published only after all
requested setters, group admission and interface selection succeed.

Effective sizes are the values returned by SO_RCVBUF and SO_SNDBUF, not a claim
of payload capacity. Kernels may round, cap or account for metadata. In
particular, Linux doubles the set value for bookkeeping. Readback exposes that
distinction; no retry, privileged override or invented normalization exists.
Socket queue sizes are independent of QUIC's per-connection S(B+4) protocol
window and do not alter framing, stream admission or operation deadlines.

A multicast group must satisfy IP multicast classification and match the bound
socket family. IPv4 carries an interface address; IPv6 carries an interface
index. Joining the group and selecting that outgoing interface use the same
typed variant. No boolean substitutes for group or interface identity. Socket
ownership bounds membership lifetime. Multicast supplies neither peer identity
nor reliable delivery; Metis command sessions continue to require QUIC/TLS.

## Alternatives and verification

Keeping ignored fields preserves a false contract. Configuring async UDP and
QUIC independently duplicates platform policy. Forcing larger default queues
without measured local loss changes memory and latency without evidence: the
existing controlled loss failures have zero kernel UDP errors and remain open.

Tests compare effective buffers against a separately configured real native
socket, exercise receive/send requests independently, reject native integer
overflow, unicast groups and family mismatch, and verify configured IPv4/IPv6
unicast and IPv4 multicast bytes and source addresses. Configured QUIC completes
TLS, exact-message exchange and acknowledged FIN. Native platform gates include
the async facade as well as PAL and QUIC; existing profile workloads, counts and
deadlines remain unchanged.

Migration: replace `UdpConfig.buffer_size` with `buffers.receive` and/or
`buffers.send` using NonZeroUsize. Replace `multicast: true` with the appropriate
group/interface variant. `UdpConfig` moves out of the async `types` module to
the PAL-owned vocabulary exported by `moirai_async::net`.

References: [socket2 source](https://github.com/rust-lang/socket2/blob/v0.6.0/src/socket.rs),
[Linux socket options](https://man7.org/linux/man-pages/man7/socket.7.html),
[Windows multicast options](https://learn.microsoft.com/en-us/windows/win32/winsock/ipproto-ip-socket-options),
[RFC 1112](https://www.rfc-editor.org/rfc/rfc1112).

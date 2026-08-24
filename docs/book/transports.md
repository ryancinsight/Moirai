# Transports and their capability contract

Part IV moves from *within one process* (Parts I–III) to *across
boundaries*. This chapter defines the transport abstraction: what every
communication backend must provide, how addresses select a backend, and
which failure semantics a caller may rely on.

## The `Transport` contract

Everything that can carry bytes between endpoints implements one
three-method trait:

```rust,ignore
pub trait Transport: Send + Sync {
    fn send(&self, target: &Address, data: Vec<u8>) -> TransportResult<()>;
    fn recv(&self, source: &Address) -> TransportResult<Vec<u8>>;
    fn supports(&self, address: &Address) -> bool;
}
```

Three properties of this contract drive the rest of the design:

1. **Bytes, not types.** A transport moves `Vec<u8>`. Typing is layered on
   top by the archive channels (Chapter [Channels](channels.md)),
   because — as the removal note in `lib.rs` records — a channel generic
   over arbitrary `Send` T cannot serialize without a serialization bound.
   The split keeps backends free of serialization concerns and
   serialization free of backend concerns.
2. **`supports` is the capability predicate.** A transport answers for the
   address shapes it can serve; nothing else decides reachability. The
   in-memory transport supports exactly `Address::Local`, the network
   transport exactly `Address::Remote`.
3. **Two failure modes only.** `TransportError::Closed` means "this path
   cannot carry the message" (unsupported address, connect/bind failure,
   I/O error, stall timeout); `Full` means "the message exceeds a declared
   capacity". There is no third silent outcome: every failure surfaces as
   one of these typed values, never as a hang.

## Addresses

```rust,ignore
pub enum Address {
    Local(String),                    // rendered local://{id}
    Remote(RemoteAddress),            // {host, port, service}
}
```

An address is data, not a connection. `send`/`recv` take addresses by
reference and resolve them per call; there is no persistent session object
to leak or forget. The `service` field on a remote address names the
logical endpoint behind host:port so routing layers (*The router* (Chapter 5))
can make policy decisions without parsing sockets.

## In-memory: shared-nothing threads, shared channels

`InMemoryTransport` backs each `Local(id)` with an MPMC channel pair
(default capacity 1024). Two details matter more than they look:

- **Steady state never takes a write lock.** Channel lookup runs under a
  concurrent read lock and clones an already-cloned handle — the MPMC
  channel itself is lock-free — so per-message coordination is the
  channel's atomics, not a global mutex. The write lock guards creation
  only, and re-checks after acquiring it so two racers create one channel.
- **Self-addressing is the model.** `transport.send(Local("a"))` followed
  by `transport.recv(Local("a"))` is not a degenerate test case; it *is*
  the local communication pattern. Distances between "threads in one
  process" and "processes on one machine" are absorbed by which transport
  instance holds the channel map, not by different APIs.

## Network: bounded blocking frames

The network backend speaks a deliberately small protocol over TCP:
little-endian `u64` length prefix, then payload.

Every bound in `network.rs` exists because a hostile or broken peer can
violate it:

| Bound | Value | Defended against |
| --- | --- | --- |
| Frame size | 16 MiB (`MAX_NETWORK_MESSAGE_BYTES`) | oversized allocation from a lying length prefix → `Full` |
| Connect attempts | 64 × 1 ms | transient listener startup races |
| Write timeout | 30 s | peer that accepts then never drains |
| Read timeout | 30 s | peer that connects then stalls mid-frame |

Both directions are bounded: the sender cannot be pinned by a full receive
buffer, and the receiver's `read_exact` cannot hang past the timeout. A
timeout surfaces as `Closed` — the same type an unsupported address
produces — because from the caller's perspective both mean "this path is
not currently usable"; retry policy (*The router* (Chapter 5)) is
layered above and may treat them differently.

One structural consequence worth internalizing: `recv` binds a fresh
listener per call (`read_network_frame`). Each receive is a complete
server interaction — accept once, read one frame. That makes the backend
stateless and trivially testable over loopback (the unit tests bind port
0 and race sender against receiver), at the cost of one handshake per
message. Persistent-connection pooling is a router-layer concern, not a
transport invariant.

## Composition: manager and connection tracking

`TransportManager` routes by first-`supports`-wins across its registered
backends — the default order (in-memory, then network) means local traffic
never touches the network stack, and unknown address shapes fail with
`Closed` rather than guessing. `ConnectionManager` is deliberately dumber:
it tracks `Connected`/`Disconnected` labels per address for observation
and policy, and asserts nothing about liveness. State that would require
probing belongs to health checks above this layer.

## Process boundaries

Crossing into another OS process reuses the same byte-frame discipline;
the lifecycle primitives in `process.rs` (`ManagedProcessId`,
`ProcessSpec`, drop policies `TerminateOnDrop`/`DetachOnDrop`, bounded
`ProcessWaitPolicy`) exist so spawn/wait/terminate compose with the same
bounded-wait philosophy as the frame timeouts: no unbounded waits, no
orphaned children by accident.

## Worked example: loopback round trip

From the transport test suite (`transport.rs` tests) — the shape every
backend must support:

```rust,ignore
let manager = TransportManager::new();
let addr = Address::Remote(loopback_remote_address());

let receiver = std::thread::spawn({
    let addr = addr.clone();
    move || TransportManager::new().recv(&addr)
});
manager.send(&addr, b"payload".to_vec())?;

assert_eq!(receiver.join().unwrap(), b"payload");
```

The same three lines with `Address::Local("id")` run entirely in-memory.
That symmetry — identical call sites, different physical paths selected by
`supports` — is the capability contract the rest of Part IV builds on.

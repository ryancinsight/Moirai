# Remote tasks: capabilities and server lifecycle

The final Part IV chapter assembles everything before it into the
module that motivated it: running a unit of work inside another process
or on another machine, and getting a validated answer back.
`remote_task.rs` plus its `capability.rs` and `server.rs` submodules
show every Part IV idea earning its keep — regions, archive views,
bounded transports, and typed endpoints.

## Capabilities make wrong requests unrepresentable

A remote operation is a fixed-format wire program: today, `EchoBytes`
and `SumU64`. Nothing about those enums stops a caller from pairing a
byte payload with the sum op — unless the pairing lives in the type:

```rust,ignore
pub trait RemoteCapability: Sealed + Copy + Default + Send + Sync + 'static {
    const OPERATION_KIND: RemoteTaskOperationKind;
}

pub trait IntoRemoteOperation<C: RemoteCapability> {
    fn into_remote_operation(self, token: RemoteCapabilityToken<C>) -> RemoteTaskOperation;
}
```

`IntoRemoteOperation` is implemented only for the coherent pairs:
`Vec<u8>` with `EchoBytesCapability`, `Vec<u64>` with
`SumU64Capability`. `build_remote_operation(payload, token)` therefore
compiles only when payload shape matches capability — a `Vec<u64>`
offered to the echo capability is a type error, not a runtime surprise.
The sealed supertrait keeps the admitted set upstream-owned, exactly as
with payload regions in Chapter 2. The token itself is zero-sized; its
entire job is to exist so overload resolution can select the right
conversion.

## Zero-copy execution on the server

The server never materializes a request to run it. The archived
envelope validates into `RemoteTaskEnvelopeView<'a>`, whose `SumU64`
arm holds a `RemoteU64List<'a>` — a length plus a borrow of the raw
archive bytes. Execution is then:

```rust,ignore
self.bytes
    .chunks_exact(core::mem::size_of::<u64>())
    .fold(0u64, |sum, chunk| {
        let bytes: [u8; 8] = chunk.try_into().expect("chunk size is fixed");
        sum.wrapping_add(u64::from_le_bytes(bytes))
    })
```

No owned `Vec<u64>` is ever built; the sum walks the wire bytes in place,
and the *wrapping* arithmetic is part of the operation's contract rather
than an overflow accident. Echo copies once — because the reply needs an
owned buffer anyway — which is the correct amount of copying for the
job.

## Correlation: the id comes home

Every envelope carries `task_id` and `reply_to`; the server's reply
echoes the id unchanged. The client's `execute` refuses to trust that:
the returned result's id must equal the requested one or the answer is
rejected (`Closed`). On a transport where each call binds fresh sockets
(Chapter 1), misdirected replies are possible in principle; correlation
turns "some result arrived" into "*this* result arrived."

## Two servers, two lifecycles

`RemoteTaskServer::serve_one` is the minimal loop made explicit:
receive → tag with `ServerPayloadRegion` → validate via
`ArchivedMessage::get` → execute → archive reply → send to `reply_to`.
One request, start to finish, no hidden state.

`BoundedRemoteTaskServer` scales the same loop under three constraints,
each expressed as a validating newtype that normalizes zero into one
(`RemoteTaskQueueCapacity`, `RemoteTaskWorkerCount`) or bounds a run
(`RemoteTaskRequestLimit`):

```rust,ignore
let listener = TcpListener::bind(...)?;
let (sender, receiver) = mpsc::sync_channel(self.queue_capacity.get());
// N workers share the receiver; the accept loop feeds the queue.
```

- **Backpressure by construction**: the bounded channel applies Chapter
  1's rule at the server itself — when the queue is full, the accept
  loop blocks, TCP backs up to the peer, and nothing grows unboundedly.
- **Starvation defense**: the accept loop sets the shared 30 s read
  timeout before reading each frame, so one stalled peer cannot wedge
  the single accepter while workers sit idle.
- **Graceful drain**: serving exactly `limit` requests ends with
  `drop(sender)`; workers see a closed channel, finish in-flight work,
  and exit — `RemoteTaskServerStats { accepted, completed }` reports the
  accounting, and the difference between those numbers is visible
  instead of lost.

## The full arc, end to end

From Chapter 4's routed clients down to silicon:

1. Scheduler selects `SchedulerRoute::Server(id)`.
2. Address book resolves it against the endpoint catalog (or the
   documented local fallback).
3. Capability + payload compile into an operation; the envelope archives
   with a `ServerPayloadRegion`.
4. Transport carries the frame under its size and timeout bounds.
5. Server validates the archive view, executes zero-copy where the
   operation allows, and replies to `reply_to`.
6. Client correlates ids, converts the borrowed output view to owned,
   and hands a `RemoteTaskResult` back to the scheduler layer.

Six steps, six earlier sections' invariants composed. That composition —
each layer adding one guarantee and delegating the rest — is the routing
and transport design this part of the book set out to teach.

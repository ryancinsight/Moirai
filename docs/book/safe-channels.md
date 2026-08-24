# Safe channels: typed endpoints over raw links

Chapter 1 ended with transports moving untyped bytes; Chapter 2 gave
those bytes meaning through archive traits. This chapter completes the
stack: `safe_channel.rs` puts the two together into *typed endpoints* —
a sender that accepts `&T`, a receiver that yields validated views of
`T` — over any transport, addressed by name.

## Why `T: Send` was not enough

The module's history is part of its design lesson. An earlier
`UniversalChannel<T: Send>` accepted any sendable value and did nothing
with it: `send` ignored its argument and returned `Closed`. It looked
generic and was a mock — removed rather than fixed, per the note in
`lib.rs`. The insight survives in the replacement's shape: a channel
generic over an arbitrary `Send` type **cannot** serialize, because
`Send` says nothing about representation. Serialization is a separate
capability, and so the endpoint traits split:

```rust,ignore
pub trait ArchiveSerialize: Send + 'static { /* encode_archive ... */ }
pub trait ArchiveView: Send + 'static {
    type Archived<'a> where Self: 'a;
    fn view_archive(bytes: &[u8]) -> TransportResult<Self::Archived<'_>>;
}
```

The sender side needs *emit*; the receiver side needs *validate-and-view*.
Nothing forces one type to implement both, and the receiver's GAT lets
each type choose what its borrowed form is — `i32` views as `i32`,
`String` views as `&str`.

## The endpoints

```rust,ignore
pub struct ArchivedUniversalSender<T: ArchiveSerialize + ?Sized> {
    transport: Arc<TransportManager>,
    target: Address,
    _phantom: PhantomData<T>,
}

pub struct ArchivedUniversalReceiver<T: ArchiveView> {
    transport: Arc<TransportManager>,
    source: Address,
    _phantom: PhantomData<T>,
}
```

Three deliberate details:

- **The type travels in `PhantomData`, not in storage.** Endpoints hold a
  shared transport manager and an address; the marker makes wrong-type
  sends unrepresentable at zero runtime cost. `T: ?Sized` on the sender
  means `ArchivedUniversalSender<str>` works — you can send borrowed text
  without constructing an owned `String`.
- **Endpoints are names, not connections.** Each binds a `(manager,
  address)` pair; nothing is dialed until `send`/`recv` runs. Two
  endpoints cooperate only because their addresses route to the same
  channel or socket, which keeps the Chapter 1 capability story intact.
- **The receiver owns, then lends.** `recv()` returns
  `ArchivedMessage<T>` — the byte buffer moved out of the transport — and
  `.get()` validates on demand, lending a `T::Archived<'_>` tied to the
  message's lifetime. Validation is deferred but never skipped: you cannot
  see bytes-as-`T` without passing through `view_archive`.

## The reference implementations

Two built-in impl pairs define the house style for archive formats.

Fixed-width (`i32`): four little-endian bytes, hint = size, view = try a
4-byte array conversion. Trivial, and the baseline every format starts
from.

Length-prefixed (`str`/`String`) is worth reading line by line because it
is a complete hostile-input validator in fifteen lines:

```rust,ignore
fn view_archive(bytes: &[u8]) -> TransportResult<&str> {
    if bytes.len() < 4 { return Err(TransportError::Closed); }
    let len_bytes: [u8; 4] = bytes[0..4].try_into()...;
    let len = u32::from_le_bytes(len_bytes) as usize;
    let end = len.checked_add(4).ok_or(TransportError::Closed)?;
    let payload = bytes.get(4..end).ok_or(TransportError::Closed)?;
    if bytes.len() != end { return Err(TransportError::Closed); }
    str::from_utf8(payload).map_err(|_| TransportError::Closed)
}
```

Five checks, each answering a specific attack or bug class: short header,
overflowing length arithmetic (`checked_add` before slicing), length past
the buffer (`get` instead of index), **trailing garbage rejected by exact
equality** — a stricter choice than "at least this many bytes", which
makes frame boundaries self-describing — and finally UTF-8 validity.
Every failure collapses to the single transport error type; richer error
taxonomies would buy callers nothing here, since all outcomes mean "the
bytes are not a `String`".

The matching encoder writes `u32::LE` length then payload, with
`try_from` rejecting lengths beyond `u32` — symmetric with the reader, so
the two can never drift apart silently.

## Composition example

Sender and receiver meet only through the transport manager's address
routing:

```rust,ignore
let manager = Arc::new(TransportManager::new());
let addr = Address::Local("demo".into());

let sender = ArchivedUniversalSender::<str>::new(manager.clone(), addr.clone());
let receiver = ArchivedUniversalReceiver::<String>::new(manager, addr);

sender.send("typed over bytes")?;
let message = receiver.recv()?;
assert_eq!(message.get()?, "typed over bytes");
```

`&str` in, `&str`-as-view out, zero copies beyond the single buffer the
transport already produced — and the same code runs in-memory or over TCP
purely by address shape, inheriting every bound and failure semantic from
Chapter 1.

## Where this leaves Part IV

With payloads (Chapter 2) providing region-tagged bytes and safe channels
providing typed endpoints, the remaining chapters move up one level:
routes select among paths (Chapter 4), the router adds retry and
backpressure policy (Chapter 5), and remote tasks bind capabilities to
executors across those paths (Chapter 6).

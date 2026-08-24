# Payload framing and ownership regions

Chapter 1 established that transports move `Vec<u8>`. This chapter covers
the two questions that immediately follow: *who owns those bytes as they
cross each boundary*, and *what gives the bytes meaning*. Both answers in
`moirai-transport` are type-level, and both are cheaper than they look.

## Ownership regions

Raw byte buffers are dangerous to pass loosely: a pointer into a shared
mapping is meaningful inside one process and nonsense inside another.
`payload.rs` encodes *where bytes are allowed to be* as a sealed family of
zero-sized markers:

```rust,ignore
pub trait PayloadRegion: Sealed + Copy + Default + Send + Sync + 'static {
    const BOUNDARY: PayloadBoundary;
    const POINTER_TRANSFER_ALLOWED: bool;
}
```

Four implementors exist — `ThreadPayloadRegion`, `ProcessPayloadRegion`,
`ServerPayloadRegion`, `DevicePayloadRegion` — and because the supertrait
is private, no downstream crate can add a fifth without upstream review.
That is deliberate: the region set mirrors the routing boundaries of
Part IV, and an open set would let a new boundary skip the analysis of
whether raw identity survives it. Only the thread region sets
`POINTER_TRANSFER_ALLOWED = true`; within one process, a pointer keeps its
meaning across a handoff. Everywhere else it does not, and the constant
says so at compile time — pinned by a `const _: () = { assert!(...) }`
block in the module's tests so a future edit cannot silently flip it.

## The payload wrapper

```rust,ignore
#[repr(transparent)]
pub struct TransportPayload<R: PayloadRegion> {
    bytes: Vec<u8>,
    _region: PhantomData<R>,
}
```

The marker costs nothing (`repr(transparent)`, zero-sized `PhantomData`);
the type is the bytes plus a compile-time fact about where they may go.
Moving between regions is `handoff::<Target>()`, which transfers the same
allocation and re-tags it — the unit test asserts the returned buffer
keeps its exact address through Thread → Process → Device handoff. Zero
copies, zero runtime checks: the region system constrains *reasoning*,
not execution, and disappears from the machine code entirely.

## The archive contract

Bytes acquire meaning through two traits in `safe_channel.rs`:

```rust,ignore
pub trait ArchiveSerialize: Send + 'static {
    fn archive_size_hint(&self) -> usize { 0 }
    fn encode_archive(&self, output: &mut Vec<u8>) -> TransportResult<()>;
    fn archive_bytes(&self) -> TransportResult<Vec<u8>> { /* hint + encode */ }
}

pub trait ArchiveView: Send + 'static {
    type Archived<'a> where Self: 'a;
    fn view_archive(bytes: &[u8]) -> TransportResult<Self::Archived<'_>>;
}
```

The split follows the same producer/consumer shape as everything else in
Moirai. A sender needs only to *emit* bytes (`encode_archive` appends into
a caller-owned buffer, pre-sized by the hint); a receiver needs to
*validate and borrow*: `view_archive` checks the byte stream and returns a
`Self::Archived<'_>` that points into the original buffer. Nothing is
deserialized into a fresh value unless the receiver asks for that — this
is the rkyv-style zero-copy discipline, and it is why the framing layer
has no decode step to get wrong.

`archive_transport_payload` composes the halves: archive a value, tag the
bytes with a region, hand the typed bundle onward.

## Validation is the trust boundary

Because receivers borrow rather than decode, `view_archive` is the single
point where hostile or corrupted bytes meet the type system. Every safety
argument downstream rests on it having run: an `ArchivedMessage<T>::get()`
call validates before producing a view, so a truncated or malformed frame
surfaces as a typed error instead of a bogus `&T`. This is the same
philosophy as the HTTP codec's restriction floor (SEC-001): parse nothing
unchecked at a trust boundary. When fuzz targets extend beyond the HTTP
codec, the archive validators are the next candidates.

## What is not here yet: versioning

The wire format has no magic number or version field today. That is safe
only while every writer and reader build from the same tree — true inside
one repository's integration runs, false the moment long-lived processes
or independent release cadences appear. The evolution recipe when needed
is fixed by policy elsewhere in this stack: add an explicit header,
migrate readers old→new with committed fixtures from every supported
prior version, and contract tests on the round trip. The framing types
above need no redesign to carry such a header; treat its absence as
recorded debt, not a property of the design.

## Worked example

From the module's own tests — archive, tag, hand off, and observe that
neither the bytes nor their address changed:

```rust,ignore
let payload = archive_transport_payload::<ServerPayloadRegion, _>(
    &"route payload".to_string(),
)?;

// rkyv framing: u32 little-endian length, then bytes.
assert_eq!(payload.as_bytes()[..4], 13u32.to_le_bytes());

let device_payload = payload.handoff::<DevicePayloadRegion>();
assert_eq!(device_payload.len(), 17);
assert!(!TransportPayload::<DevicePayloadRegion>::pointer_transfer_allowed());
```

The length prefix visible in the assertion is part of the string archive
format; receivers never parse it manually — `view_archive` does — which
is exactly the division of labor this chapter argues for.

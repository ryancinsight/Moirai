# ADR 0024: SpscChannel is reached only through its halves

Status: Accepted

- Date: 2026-07-27
- Change class: [arch] [major]
- Refs: PR #106 audit round

**Intent.** Decide how `moirai-core`'s single-producer/single-consumer channel
is exposed, given that its `Sync` impl is sound for the sender/receiver pair but
not for the bare channel the crate also exported.

**Context.** `moirai-core/src/channel/spsc.rs` is a bounded SPSC ring buffer.
The protocol is correct and is not at issue: the producer loads `head` relaxed
and `tail` acquire, writes the slot, then release-stores `head`; monotonic
counters make the full check exact without a wasted slot; `Drop` drops exactly
`[tail, head)`.

The halves are correct too, and deliberately so. `SpscSender` and
`SpscReceiver` are neither `Clone` — so at most one of each exists — nor `Sync`,
because each carries a `PhantomData<Cell<()>>` that removes `Sync` while leaving
`Send`. A half can therefore be moved to the thread that owns its role but never
shared, which is what makes one producer and one consumer a type-level fact
rather than a documented convention.

The bare channel escaped alongside them. `SpscChannel` was exported twice, from
`channel/mod.rs` and again from the crate root, with a `pub fn new`; it
implements the exported `Channel<T>` trait, whose `send`/`recv` take `&self`;
and it carries `unsafe impl<T: Send> Sync`. Those three facts compose into safe
code that bypasses the discipline the halves enforce:

```rust
let ch = Arc::new(SpscChannel::<u64>::new(8));
let (a, b) = (Arc::clone(&ch), Arc::clone(&ch));
std::thread::spawn(move || { let _ = a.send(1); });
std::thread::spawn(move || { let _ = b.send(2); });
```

Both producers load the same `head` and write `buffer[head & mask]`: a data race
on one slot, with one `T` overwritten without being dropped. Two concurrent
`recv` calls are worse — both `assume_init_read` the same slot, moving out of it
twice and then dropping it twice. This was verified by compiling the program
above as an integration test, which sees the crate as an external consumer; the
runtime race was not demonstrated, and does not need to be, since a safe API
that *can* produce it is already a defect.

`SpscChannel` is the only channel in that module carrying an `unsafe impl Sync`.
Mpmc and Hybrid need none. It exists to satisfy `Channel<T>: Send + Sync`, a
supertrait written for genuinely shareable channels, and the SPSC type was
fitted to a bound its access pattern cannot honour.

**Constraints.**

- The ring protocol and the halves' design are correct and stay unchanged.
- `Sync` on the channel cannot simply be removed: both halves hold
  `Arc<SpscChannel<T>>`, and `Arc<T>: Send` requires `T: Send + Sync`, so
  dropping it would make the halves unsendable and break the intended use.
- The fix must make misuse unrepresentable, not merely documented. A safe API
  that can reach undefined behaviour is a defect regardless of what its docs say.

**Options.**

1. *Seal the channel.* Make `SpscChannel` `pub(crate)` and stop re-exporting it,
   leaving `channel::spsc(capacity)` — already public, already the idiom used by
   the crate's own tests — as the only way in. Subtractive; no behaviour change;
   breaking only for code naming the type directly.
2. *Drop the `Channel<T>` impl from `SpscChannel`.* Removes the `&self`
   entry points but leaves a public type whose `Sync` impl is still unsound for
   any inherent `&self` method added later. Treats a symptom.
3. *Runtime producer claim.* An atomic flag turning concurrent misuse into a
   panic. Non-breaking, but it adds a check to a low-latency hot path and
   converts undefined behaviour into a panic rather than making it impossible.
4. *Make it MPMC with compare-exchange on both indices.* Sound, and discards the
   reason the type exists.

**Decision.** Option 1. The halves already encode the invariant; the fix is to
stop offering a way around them. `channel::spsc` is unchanged, so the supported
path is untouched, and the only in-workspace user of the type — `select.rs`,
through that same factory — needs no edit.

**Consequences.**

- *Breaking for direct users.* Code naming `moirai_core::channel::SpscChannel`
  or `moirai_core::SpscChannel` no longer compiles. Migration is one line:
  `SpscChannel::channel(n)` becomes `channel::spsc(n)`, returning the same pair.
- *The `Channel` trait no longer has an SPSC implementor in its public surface.*
  Generic code written against `Channel<T>` can still use `MpmcChannel` and
  `HybridChannel`, which are genuinely shareable; an SPSC channel was never
  substitutable there, and the trait bound had been asserting otherwise.
- *The halves' `PhantomData<Cell<()>>` becomes load-bearing in a second way.* It
  was always what kept them unshareable; now it is also the only barrier, since
  the bare channel is gone. It reads like an unused field, so it is documented at
  each site and pinned by an assertion.

**Verification plan.**

1. The exploit above must stop compiling — the same integration test that
   demonstrated the hole, re-run against the sealed crate.
2. `assert_not_impl_any!` on both halves' `Sync`, because a positive assertion
   cannot express it: deleting the marker only widens the impls, so every
   `assert_impl_all` keeps passing while the race returns.
3. Existing SPSC channel tests stay green unchanged, proving the supported path
   is unaffected.
4. `cargo check --workspace --all-targets`, since removing a public export is
   breaking and any in-workspace user must surface.
5. `fmt --check`, `clippy --all-targets -D warnings`, `nextest`, and a
   warning-denied `cargo doc`, with at least one gate run `--locked` because CI
   passes it and a manifest change otherwise fails every job.

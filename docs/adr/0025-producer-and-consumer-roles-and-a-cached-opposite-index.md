# ADR 0025: Producer and Consumer roles, and a cached opposite index

Status: Accepted

- Date: 2026-07-27
- Change class: [arch] [minor]
- Refs: ADR-024, PR #108

**Intent.** Restore, soundly, the capability ADR-024 removed — writing generic
code against an SPSC half — and cut the steady-state cost of the ring while
doing it.

**Context.** ADR-024 sealed `SpscChannel` because `Channel<T>` requires
`Send + Sync` and an SPSC half cannot honestly be `Sync`: `send` takes `&self`,
so two threads sharing a sender would claim the same slot. Sealing closed the
soundness hole but left a functional one — a caller who wanted "something I can
send into" had `Channel<T>` as the only vocabulary, and the SPSC halves were
excluded from it by the very bound that made them safe.

The cost side is separate and pre-existing. Every `try_send` loaded the
consumer's `tail` with `Acquire`, and every `try_recv` loaded the producer's
`head`. The two counters sit on their own cache lines (`CacheAligned`), so each
operation pulled the opposite thread's line across the interconnect even when
the queue was nowhere near full or empty — a coherence miss on an operation
that is otherwise two loads and a store.

**Constraints.** No compatibility shim for the sealed type. Nothing may weaken
the `!Sync` property the halves depend on. The full check must stay exact — no
sacrificial slot bought back for a cheaper test.

**Options.**

1. *Relax `Channel`'s supertrait to `Send`.* One edit, and it re-admits the
   halves. Rejected: it weakens the contract every genuinely shareable channel
   currently states, to describe a type that is not one of them.
2. *Blanket-impl the roles over `Channel<T>`.* Attractive — existing channels
   would satisfy the roles for free. Rejected on coherence: a downstream crate
   may implement `Channel<TheirType>` for `SpscSender<TheirType>`, so the
   compiler must treat the blanket impl as overlapping every per-type impl, and
   the per-type impls are exactly the ones this ADR exists to add. Verified,
   not predicted — the impls were written and the compiler rejected them
   (E0119).
3. *Split the roles and implement them per type.* Chosen.

For the cache, the alternative to per-half caching was to drop `CacheAligned`
so both counters share a line. That trades a guaranteed false-sharing bounce
for a smaller struct, which is the wrong direction for a queue whose whole
purpose is two threads writing two counters.

**Decision.** Two traits, `Producer<T>` and `Consumer<T>`, neither requiring
`Sync`, each carrying only its own half of the contract plus the introspection
that half can answer. They are implemented per type, beside the type: on both
SPSC halves, both MPMC halves, and on `MpmcChannel` itself, so a shareable
channel is usable whole or split.

Each half also gains a `Cell<usize>` holding its last known value of the
opposite counter. An operation consults the cache first and touches the other
thread's line only when the cache says the queue is full (producer) or empty
(consumer), at which point it refreshes and retries once. The soundness of this
is one-sided and worth stating precisely: only the consumer advances `tail`, and
only forward, so a stale `cached_tail` places the consumer *behind* where it
really is and therefore makes the queue look **fuller** than it is. The cache
can cost an unnecessary slow path; it can never report space that does not
exist. The consumer's `cached_head` mirrors this — it can understate what is
queued, never invent an element.

The same `Cell` supplies the `!Sync` property, which previously came from a
`PhantomData<Cell<()>>` marker. This is deliberate and reduces a hazard rather
than adding one: the old marker was load-bearing but inert, so a future reader
could delete it as dead weight. A field the send path reads on every call
cannot be removed without the code failing to compile, and
`assert_not_impl_any!(SpscSender<u64>: Sync)` still fails if its type is
changed to something `Sync`.

**Consequences.** The sealed capability is back, expressed by traits that state
what each half actually promises instead of the union of what all channels do.
The roles are additive — no existing signature changed — but they are a public
surface, so the blanket impl is now permanently foreclosed; a future shareable
channel writes two short impls beside its `Channel` impl. The steady-state send
and receive no longer read the opposite cache line.

**Verification plan.**

1. `spsc_halves_satisfy_the_roles` calls a `fn drain_into<P: Producer<u64>>`
   with an SPSC sender — the generic use ADR-024 made impossible, which is the
   test that would have failed before this change.
2. `shareable_channels_and_their_halves_satisfy_the_roles` covers `MpmcChannel`
   whole and split, so the per-type impls are not silently SPSC-only.
3. `auto_traits`' `assert_not_impl_any!(SpscSender<u64>: Sync)` guards the
   property the `Cell` now carries; positive assertions cannot express it,
   because removing an impl only widens what compiles.
4. The existing SPSC suite — blocking, close-with-value-in-flight, drop of
   either half — covers the cached fast and slow paths, since a full queue is
   what forces the refresh.
5. A `spsc_throughput` criterion bench, one producer thread and one consumer
   thread at capacities 64 and 8192: the deep queue is where the cache should
   pay, the shallow one where it must at least not harm. Recorded as an
   instrument here; the measurement is not claimed until it runs on a quiet
   host, since a memory-bound benchmark taken beside a dozen concurrent
   compiles measures the host, not the queue.

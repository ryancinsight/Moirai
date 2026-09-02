# ADR 0026: A borrowed SPSC half pair beside the `Arc`-backed one

Status: Accepted

- Date: 2026-07-27
- Change class: [arch] [minor]
- Refs: ADR-024, ADR-025, PR #109

**Intent.** Offer an SPSC pair with no reference counting for the common case
where both halves live inside a known scope, without giving up the `'static`
pair that independent threads need.

**Context.** `spsc(capacity)` gives each half an `Arc<SpscChannel<T>>`. That is
the right shape when producer and consumer threads outlive each other
arbitrarily. It is pure overhead when both halves are created and dropped
inside one `thread::scope`, a frame loop, or a pipeline stage: the scope already
proves the ring outlives the halves, so the refcount is enforcing a lifetime
relationship the compiler could check statically. The cost is one heap
allocation for the ring plus an atomic decrement and a conditional deallocation
per half.

ADR-025's cached-index work made this cheap to add without noticing: the ring
primitives were changed to take `&self` plus an externally supplied `Cell`, so
they are already agnostic about whether the caller reached the ring through an
`Arc` or a borrow. No primitive changed for this ADR.

**Constraints.** One producer and one consumer, statically. No duplication of
the ring protocol — a second copy of the send/receive logic would be the cloned
variant the standards forbid. Both flavours must satisfy the same roles, so
generic code does not fork.

**Options.**

1. *A `'brand`-style generative lifetime* (`GhostCell`-flavoured), giving each
   split a unique invariant lifetime. Rejected: it delivers exactly the
   guarantee `&mut self` already delivers here, at the cost of a closure-scoped
   API (`ring.split(|tx, rx| ...)`) and an invariant-lifetime parameter in every
   consumer signature. Branding earns its complexity when *several* independent
   objects must be provably paired; a single ring handing out its own halves is
   not that case.
2. *A runtime `split_once` flag* on the ring. Rejected: a runtime check for
   something the borrow checker already refuses.
3. *`split(&mut self)` returning halves that borrow the ring.* Chosen. The
   exclusive borrow is consumed by the returned halves, so a second split, a
   move, or a drop of the ring is a compile error while either half is alive.
   This is the same guarantee the `Arc`-backed halves get from being non-`Clone`,
   obtained from the borrow checker instead of from a refcount.

**Decision.** `SpscRing<T>` owns the ring by value and `split(&mut self)` yields
`SpscProducer<'_, T>` / `SpscConsumer<'_, T>`, which hold `&SpscChannel<T>` plus
their own cached index. Both implement the ADR-025 roles, so a function generic
over `Producer<T>` accepts a borrowed half and an `Arc`-backed one alike, with
no `'static` bound. `spsc(capacity)` is unchanged.

The module becomes a directory — `spsc/{ring, shared, borrowed}` — because one
file now holds the ring protocol and two independent wrappers over it, and was
already at the 500-line target.

**A ring may be re-split once its halves are dropped**, which is what makes it
reusable across phases without reallocating. Two things make that sound, and
both are why `split` seeds the caches from the live counters rather than from
zero:

- `closed` is cleared. A half sets it on drop so its peer stops blocking;
  leaving it set would fail the next round's first operation.
- The consumer's cache must satisfy `tail <= cached_head <= head`. A zeroed
  `cached_head` against a non-zero `tail` breaks the left inequality, and
  `has_value` then reports a value present in an empty ring and
  `assume_init_read`s a slot that was never written — undefined behaviour, not
  merely a wrong value. Seeding from `head` restores the invariant; seeding the
  producer's cache from `tail` is exact for the same reason.

Reading both counters `Relaxed` in `split` is correct only because `&mut self`
proves no half exists, so nothing else can be advancing them. That justification
lives on `SpscChannel::indices`, which is the single place the ring's counters
are read without synchronization.

**Consequences.** Scope-local pipelines lose the refcount traffic and one
allocation; dropping a half becomes a single store. The `Arc` pair remains for
threads with independent lifetimes, so this is additive. The cost is a second
public pair of half types — accepted because they wrap, rather than duplicate,
the one ring protocol, and because the roles keep consumers from having to
choose between them.

**Verification plan.**

1. `resplitting_a_drained_ring_reports_empty` is the falsification test for the
   seeding rule. Verified to fail on the naive implementation: seeding
   `cached_head` from zero makes it panic with its own message and then hang the
   run to nextest's 60s termination — the signature of reading uninitialized
   memory rather than of a wrong comparison.
2. `resplitting_preserves_queued_values` covers the other half of re-splitting:
   values queued in an earlier round survive.
3. `dropping_a_loaded_ring_drops_each_value_once` counts `Drop` on queued values
   when the ring dies undrained.
4. `capacity_is_exact_with_no_sacrificed_slot` asserts all `N` slots are usable,
   so a future cheaper full-check cannot quietly buy back a slot.
5. `dropping_the_producer_closes_the_consumer` covers the close handshake a
   scoped thread depends on to join.
6. `scoped_threads_move_the_halves_across_the_boundary` runs the real
   two-thread transfer under `thread::scope` and checks the summed payload.
7. `auto_traits` asserts `Send` on both halves and, in `halves_are_not_sync`,
   the absence of `Sync` — the property that keeps the pair SPSC, which no
   positive assertion can express.

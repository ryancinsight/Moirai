//! Chase-Lev work-stealing deque.
//!
//! Single owner (`push`/`pop` at the bottom), many thieves (`steal` at the top),
//! implemented per the weak-memory-correct formulation of Lê, Pop, Cohen &
//! Nardelli (PPoPP 2013). `storage.rs` backs the slots;
//! `tests/loom_chase_lev.rs` and `tests/loom_chase_lev_resize_gate.rs`
//! model-check the transfer and resize-exclusion protocols under `--cfg loom`.
//!
//! # Memory ordering
//!
//! The protocol synchronizes two indices — `bottom` (owner-advanced) and `top`
//! (thief-advanced) — so that a slot is transferred to exactly one consumer. The
//! happens-before edges each atomic access establishes:
//!
//! - **`push`**: the slot write is published to thieves by the `Release` store to
//!   `bottom`; a thief's `Acquire` load of `bottom` that observes the new value
//!   therefore sees the initialized slot. `top` is read `Acquire` to observe
//!   completed steals before deciding whether to grow.
//! - **`pop`**: `bottom` is decremented (`Relaxed`, owner-private) to claim the
//!   slot, then a `SeqCst` fence orders that store before the `top` load so the
//!   owner and a racing thief cannot both take the last element — the fence pairs
//!   with the thief's `SeqCst` fence, and the last-element tie is resolved by a
//!   `SeqCst` CAS on `top`. On x86/x86_64 (TSO) the fence is skipped when
//!   `bottom - top >= MAX_BATCH_STEAL`, as a heuristic that a steal is unlikely
//!   to reach the popped slot. That is not what makes it sound: TSO leaves the
//!   delay before the `bottom` store reaches a thief unbounded, and a thief,
//!   unlike the thieves of Morrison–Afek, steals from any size. Soundness comes
//!   from the slot state and the thief's re-read of `bottom` after its claim
//!   (see `steal`), modelled exhaustively in `tests/loom_chase_lev_slot_claim.rs`,
//!   where the fast path without that re-read takes an item twice.
//! - **`steal`**: `top` is read `Acquire`, then a `SeqCst` fence orders it before
//!   the `Acquire` load of `bottom` (pairing with `pop`'s fence); the thief first
//!   claims the slot's generation state, then uses the successful `SeqCst` CAS
//!   to claim the index before reading it. A slot's state equals its index
//!   whether it holds an item or is free, so a claim can succeed on a slot the
//!   fence-free `pop` emptied and republished; the thief therefore re-reads
//!   `bottom` (`Acquire`, which the claim synchronizes with the owner's
//!   publish for) after the claim and returns the slot when `bottom` no longer
//!   covers the index. The generation state prevents the owner from reusing a
//!   wrapped slot until the read completes, so a losing thief never creates a
//!   speculative second value. The array pointer is loaded
//!   `Acquire` to pair with `resize`'s `Release` store, so a thief never
//!   dereferences a stale buffer.
//!
//! The steal gate packs resize ownership and active thief count into one atomic:
//! bit zero is the exclusive owner claim and each thief contributes two. Every
//! admission and claim is a read-modify-write on that single word, so its
//! modification order alone decides whether an admission precedes a claim; one
//! word also avoids the ABA window that separate flag and counter atomics would
//! open. The data edges come from the orderings on each access, documented per
//! site in `gate.rs`: an admission observes the Release that cleared the claim
//! bit, and the drain observes each departing thief's Release exit. Admissions,
//! the backoff, and the drain are SeqCst because the loom model of the gate does
//! not close under weaker orderings on that retry path; the claim is Relaxed.
//! The gate is entered once per *access*, not once per
//! element: a batch steal holds it across all of its items, so `resize` waits
//! behind a whole batch rather than a single steal, and a batch never stalls
//! mid-flight on a resize that opens between two of its items.
//!
//! A resize closes the steal gate, waits for active thieves to leave, copies the
//! live generation state, and then publishes the new buffer. Old buffers
//! displaced by `resize` are retired to a guarded list and reclaimed only once
//! no accessor
//! is in-flight (epoch reclamation via the `ReclaimPolicy`), closing the
//! use-after-free window a thief's `Acquire` array load would open.
//!
//! `ChaseLevDeque::shrink_to` is the reverse resize: the owner claims the same
//! gate, and because the drain leaves no thief holding any buffer it frees the
//! displaced and retired buffers immediately instead of retiring them.
//!
//! Storage-generation claims and resize-owner draining use bounded cooperative
//! waits: 64 processor spin hints are followed by a thread yield, then the
//! sequence repeats. A thief that observes an active resize yields immediately
//! before retrying admission. These private transitions allocate nothing and
//! never sleep.
//!
//! # Layout
//!
//! - `capacity`: validated allocation capacity.
//! - `steal_outcome`: the steal result and the batch container.
//! - `inner`: the shared state, its lifecycle, and the advisory observers.
//! - `owner`: `push` and `pop`.
//! - `thief`: single and batch `steal`.
//! - `resize`: growth, shrink, and retired-buffer reclamation.
//! - `endpoints`: the public owner and stealer handles.
//! - `gate`, `contention`, `storage`: the steal gate, bounded waits, and
//!   generation-tagged slot storage.

mod capacity;
mod contention;
mod endpoints;
mod gate;
mod inner;
mod owner;
mod resize;
mod steal_outcome;
mod storage;
mod thief;

pub use capacity::{DequeCapacity, DequeCapacityError};
pub use endpoints::{ChaseLevDeque, ChaseLevStealer};
pub use steal_outcome::{StealResult, StolenBatch};

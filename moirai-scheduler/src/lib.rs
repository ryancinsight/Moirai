//! # Work-Stealing Primitives
//!
//! Lock-free work-stealing deques and NUMA-topology primitives. These are the
//! reusable building blocks consumed by the canonical runtime scheduler
//! (`moirai_executor`'s `ThreadScheduler`); this crate intentionally provides
//! no scheduler of its own.
//!
//! ## Deques
//!
//! Three lock-free work-stealing deques, all single-owner / multi-thief:
//! - [`ChaseLevDeque`] — the canonical Chase-Lev deque: O(1) wait-free local
//!   push/pop for the owner, lock-free steal for thieves, dynamic resizing,
//!   and `bottom`/`top` isolated to separate cache lines to avoid false sharing.
//! - [`SplitDeque`] — a private owner stack backed by a shared deque, reducing
//!   steal contention when spawn rate greatly exceeds steal rate.
//!
//! Correctness is covered by exactly-once concurrency stress tests and bounded
//! `loom` models of the Chase-Lev transfer and resize-exclusion protocols.
//!
//! ## NUMA primitives
//!
//! [`numa`] exposes [`AdaptiveBackoff`](numa::AdaptiveBackoff) for
//! spin/yield/sleep backoff in stealing loops. Hardware topology is answered
//! by `themis::CpuTopology` directly, not mirrored here (ADR-037).

#![allow(clippy::redundant_closure)]
#![allow(clippy::collapsible_if)]
#![allow(clippy::cast_abs_to_unsigned)]
#![deny(missing_docs)]

pub mod deque;
pub mod numa;

pub use deque::{
    ChaseLevDeque, ChaseLevStealer, DeferredAccessGuard, DeferredReclaim, DeferredState,
    DequeCapacity, DequeCapacityError, DequeReclaimPolicy, DequeReclaimState,
    SharedEpochAccessGuard, SharedEpochReclaim, SharedEpochState, SplitDeque, StealResult,
    StolenBatch,
};

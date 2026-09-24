//! Lock-free queues for high-performance data structures.
//!
//! This module provides an efficient bounded multi-producer multi-consumer
//! queue built on per-slot sequence numbers (the Vyukov algorithm). It is the
//! workspace's only implementation of that algorithm: the scheduler injector,
//! both executors' run queues, and `moirai-core`'s bounded MPMC channel all run
//! on it.

#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

mod ring;
#[cfg(test)]
mod tests;

pub use self::ring::LockFreeQueue;

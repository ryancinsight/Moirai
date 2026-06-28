//! Async iterator implementations for I/O-bound workloads.

mod adapters;
mod consumers;
mod parallel;
mod sources;
mod traits;

pub use adapters::{
    AsyncEnumerate, AsyncFilter, AsyncMap, AsyncParallelAdapter, AsyncSkip, AsyncTake, AsyncZip,
};
pub use consumers::{AsyncCollect, AsyncFold, AsyncForEach, AsyncReduce};
pub use parallel::{ParAsyncFilter, ParAsyncMap};
pub use sources::{AsyncRangeIter, AsyncVecIter};
pub use traits::{AsyncIterator, AsyncParallelIterator, IntoAsyncIterator};

#[cfg(test)]
#[path = "../async_iter_tests.rs"]
mod async_iter_tests;

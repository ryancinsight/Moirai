//! Parallel iterator implementation for Moirai.
//!
//! This module provides a focused Rayon-style non-indexed adapter subset while
//! integrating with Moirai's scheduler and iterator consumers. Indexed
//! data-parallel execution is intentionally outside this adapter layer; callers
//! use `Moirai::for_each_indexed` and `Moirai::map_reduce_indexed` for indexed
//! scheduler work.

#![allow(dead_code)] // Development structures per ADR requirements.

mod adapters;
mod consumers;
mod sorting;
mod sources;
#[cfg(test)]
mod tests;
mod traits;

pub use adapters::{
    Chain, Chunks, Cloned, Copied, Enumerate, Filter, FilterMap, FlatMap, Inspect, Intersperse,
    Map, MapInit, MapWith, PanicFuse, Rev, Skip, Take, Update, WhileSome, Zip,
};
pub use consumers::Reduction;
pub use sorting::ParallelSliceMut;
pub use sources::{
    RangeParIter, RefVecParIter, SequentialAdapter, SequentialIterAdapter, VecNonCloneParIter,
    VecParIter, VecRefParIter,
};
pub use traits::{
    Consumer, IntoParallelIterator, IntoParallelRefIterator, ParallelExtend, ParallelIterator,
};

use consumers::{
    CollectConsumer, FilterConsumer, FindConsumer, InspectConsumer, MapConsumer, NullConsumer,
    ReduceConsumer, ReduceWithConsumer,
};

//! Parallel iterator implementation for Moirai.
//!
//! This module provides a focused Rayon-style adapter subset while integrating
//! with Moirai's scheduler and iterator consumers. The indexed boundary is
//! limited to exact-size source cardinality; full Rayon indexed producers remain
//! outside this adapter layer. Use `Moirai::for_each_indexed` and
//! `Moirai::map_reduce_indexed` for scheduler-owned indexed execution.

#![allow(dead_code)] // Development structures per ADR requirements.

mod adapters;
mod consumers;
mod indexed;
mod sorting;
mod sources;
mod split;
#[cfg(test)]
mod tests;
mod traits;

pub use adapters::{
    Chain, Chunks, Cloned, Copied, Enumerate, Filter, FilterMap, FlatMap, Flatten, Inspect,
    Intersperse, Map, MapInit, MapWith, PanicFuse, Rev, Skip, Take, Update, WhileSome, Zip, ZipEq,
};
pub use consumers::Reduction;
pub use indexed::IndexedParallelIterator;
pub use sorting::ParallelSliceMut;
pub use sources::{
    RangeParIter, RefVecParIter, SequentialAdapter, SequentialIterAdapter, VecParIter,
    VecRefParIter,
};
pub use split::Either;
pub use traits::{
    Consumer, IntoParallelIterator, IntoParallelRefIterator, ParallelExtend, ParallelIterator,
};

use consumers::{
    CollectConsumer, FilterConsumer, FindConsumer, InspectConsumer, MapConsumer, NullConsumer,
    ReduceConsumer, ReduceWithConsumer,
};

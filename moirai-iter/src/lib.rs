//! Moirai Iterator - Unified high-performance iterator system for concurrent, parallel, and async computing.
//!
//! This module provides a comprehensive iterator framework that abstracts over different execution contexts:
//! - **Parallel**: CPU-bound work across multiple threads with work-stealing
//! - **Async**: I/O-bound work with efficient async/await patterns
//! - **Hybrid**: Mixed workloads combining parallel and async execution
//!
//! # Design Principles
//!
//! - **Zero-cost abstractions**: Compile-time optimizations with no runtime overhead
//! - **Memory efficiency**: NUMA-aware allocation and cache-friendly data layouts
//! - **Execution agnostic**: Same API works across all execution contexts
//! - **Type safety**: Comprehensive compile-time guarantees
//! - **Performance**: SIMD vectorization and CPU optimization
//! - **Async compatibility**: Native async/await support throughout

#![deny(missing_docs)]
pub mod advanced_patterns;
pub mod async_iter;
pub mod base;
pub mod cache;
pub mod channel_fusion;
pub mod combinators;
pub mod execution;
pub mod facade;
pub mod iter_ops;
pub mod parallel;
pub mod prefetch;
pub mod simd_iter;
pub mod stream;
#[cfg(test)]
mod test_support;
pub mod windows;

pub use async_iter::{AsyncIterator, AsyncParallelIterator, IntoAsyncIterator};
pub use execution::{
    AsyncContext, ExecutionBase, ExecutionContext, ExecutionStrategy, HybridConfig, HybridContext,
    ParallelContext, PerformanceHistory,
};
pub use facade::{
    async_range, moirai_iter, moirai_iter_async, moirai_iter_hybrid, moirai_iter_parallel,
    par_range, MoiraiIterator,
};
pub use parallel::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelExtend,
    ParallelIterator, ParallelSliceMut, RangeParIter, VecParIter, VecRefParIter,
};

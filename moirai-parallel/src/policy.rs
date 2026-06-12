//! Compile-time execution policies for data-parallel operations.
//!
//! A policy decides *whether* a data-parallel operation runs in parallel. The
//! types are zero-sized **type-level markers** — the decision is an associated
//! function, so generic code over `P: ExecutionPolicy` monomorphizes to one
//! concrete path with no value passed and no dynamic dispatch:
//!
//! - [`Sequential`] / [`Parallel`] return a constant, so the unused branch is
//!   eliminated entirely at compile time.
//! - [`Adaptive`] parallelizes only at or above [`ADAPTIVE_PARALLEL_THRESHOLD`],
//!   a cheap inlined runtime check that routes per workload size (and thus across
//!   the worker threads only when worthwhile).
//!
//! Select a policy by type via the [`ParallelSlice`](crate::ParallelSlice) /
//! [`ParallelSliceMut`](crate::ParallelSliceMut) extension traits
//! (`slice.par_with::<Parallel>()`) or the `*_with::<P>` functions; the `par_*`
//! helpers and `slice.par()` use [`Adaptive`] as the unset default.

/// Element count at or above which [`Adaptive`] chooses parallel execution.
///
/// Mirrors `moirai-iter`'s `parallel_threshold`: below this, dispatch and join
/// overhead typically exceeds the benefit of parallelism.
pub const ADAPTIVE_PARALLEL_THRESHOLD: usize = 1024;

/// Compile-time strategy selector for the data-parallel operations in this crate.
///
/// Implemented by zero-sized marker types; used purely as a type parameter so
/// each operation monomorphizes to a single concrete path.
pub trait ExecutionPolicy: Send + Sync + 'static {
    /// Return `true` if an operation over `len` elements should run in parallel.
    fn parallelize(len: usize) -> bool;

    /// Return `true` if a fixed two-branch operation should run in parallel.
    #[inline(always)]
    fn parallelize_pair() -> bool {
        Self::parallelize(2)
    }
}

/// Always run sequentially (single thread, no scheduling).
#[derive(Debug, Clone, Copy, Default)]
pub struct Sequential;

/// Always run in parallel on the shared work-stealing pool.
#[derive(Debug, Clone, Copy, Default)]
pub struct Parallel;

/// Run in parallel only for inputs at or above [`ADAPTIVE_PARALLEL_THRESHOLD`].
#[derive(Debug, Clone, Copy, Default)]
pub struct Adaptive;

impl ExecutionPolicy for Sequential {
    #[inline(always)]
    fn parallelize(_len: usize) -> bool {
        false
    }
}

impl ExecutionPolicy for Parallel {
    #[inline(always)]
    fn parallelize(_len: usize) -> bool {
        true
    }

    #[inline(always)]
    fn parallelize_pair() -> bool {
        true
    }
}

impl ExecutionPolicy for Adaptive {
    #[inline(always)]
    fn parallelize(len: usize) -> bool {
        len >= ADAPTIVE_PARALLEL_THRESHOLD
    }
}

/// Run in parallel only for inputs at or above the custom threshold `N`.
#[derive(Debug, Clone, Copy, Default)]
pub struct AdaptiveWithThreshold<const N: usize>;

impl<const N: usize> ExecutionPolicy for AdaptiveWithThreshold<N> {
    #[inline(always)]
    fn parallelize(len: usize) -> bool {
        len >= N
    }
}

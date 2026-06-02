//! Compile-time execution policies for data-parallel operations.
//!
//! A policy decides *whether* a data-parallel operation runs in parallel. The
//! types are zero-sized, so generic code over `P: ExecutionPolicy` monomorphizes
//! to the selected strategy with no dynamic dispatch and no runtime cost beyond
//! the (inlined) decision itself:
//!
//! - [`Sequential`] — never parallelize (single-threaded).
//! - [`Parallel`] — always parallelize on the shared work-stealing pool.
//! - [`Adaptive`] — parallelize only above [`ADAPTIVE_PARALLEL_THRESHOLD`]
//!   elements, falling back to sequential for small inputs where scheduling
//!   overhead would dominate. This is the default used by the `par_*` helpers.

/// Element count at or above which [`Adaptive`] chooses parallel execution.
///
/// Mirrors `moirai-iter`'s `parallel_threshold`: below this, the dispatch and
/// join overhead typically exceeds the benefit of parallelism.
pub const ADAPTIVE_PARALLEL_THRESHOLD: usize = 1024;

/// Strategy selector for the data-parallel operations in this crate.
///
/// Implementors are zero-sized and `Copy`; passing one as a generic parameter
/// monomorphizes the operation to a single concrete path.
pub trait ExecutionPolicy: Copy + Send + Sync {
    /// Return `true` if an operation over `len` elements should run in parallel.
    fn parallelize(self, len: usize) -> bool;
}

/// Always run sequentially (no scheduling, single thread).
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
    fn parallelize(self, _len: usize) -> bool {
        false
    }
}

impl ExecutionPolicy for Parallel {
    #[inline(always)]
    fn parallelize(self, _len: usize) -> bool {
        true
    }
}

impl ExecutionPolicy for Adaptive {
    #[inline(always)]
    fn parallelize(self, len: usize) -> bool {
        len >= ADAPTIVE_PARALLEL_THRESHOLD
    }
}

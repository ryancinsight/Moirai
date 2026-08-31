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
/// # This value encodes an assumption about per-element cost
///
/// An element count cannot decide this on its own. Parallel wins once
/// `n * per_element_cost` exceeds the fixed dispatch cost, so the crossover
/// moves with how expensive the body is, and any single count is right for one
/// body weight and wrong for the others.
///
/// Measured on this workstation (best of 30 blocks, `map_reduce`, parallel
/// against the same fold run serially; the dispatch floor is ~11.9 us, which
/// is one task spawned per worker chunk plus the joins):
///
/// ```text
///   body weight                     crossover   parallel/serial at n = 1024
///   one multiply                    ~21K-32K    20.6x worse
///   sqrt + ln_1p                    ~8K          4.1x worse
///   24 chained fused multiply-adds  ~512-1024    0.25x  (parallel wins)
/// ```
///
/// So 1024 is tuned for an expensive body. A caller folding a cheap expression
/// over 1K-16K elements pays 1.3x to 20.6x for the parallel choice, and the
/// earlier claim here — that below this value dispatch overhead "typically
/// exceeds the benefit" — is the opposite of what happens above it for such a
/// body.
///
/// It is left at 1024 deliberately rather than raised: the stack's own heavy
/// consumers (spherical-harmonic mode loops, for one) fold expensive bodies
/// over exactly the 1K-16K range where raising it would serialize them. The
/// two ways out are re-deriving it against a stated body weight, or shrinking
/// the dispatch floor so the choice matters less; both are tracked rather than
/// guessed at here.
///
/// A caller who knows its body is cheap should select [`Sequential`], and one
/// who knows it is expensive should select [`Parallel`]. `Adaptive` is for
/// callers who know neither, and it cannot be right for both.
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

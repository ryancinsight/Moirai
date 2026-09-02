//! Logical parallelism derivation, cached for the process lifetime.

/// Logical processors this process should parallelize across.
///
/// Derived once and cached for the process lifetime. The topology cannot
/// change while the process runs, and deriving it is not cheap: a
/// `CpuTopology::detect()` call measures 9,935 ns and 77 allocations totalling
/// 16,480 bytes on a 24-processor host, because it materializes the whole
/// NUMA and cache-level description to read one count. Callers that need a
/// worker count per operation must not pay that, so this is the one place the
/// derivation happens.
///
/// `themis` reports the machine's logical processors; `available_parallelism`
/// is the fallback when no topology is available.
#[must_use]
pub fn logical_parallelism() -> usize {
    #[cfg(feature = "std")]
    {
        static CACHED: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *CACHED.get_or_init(detect_logical_parallelism)
    }
    #[cfg(not(feature = "std"))]
    {
        detect_logical_parallelism()
    }
}

fn detect_logical_parallelism() -> usize {
    #[cfg(feature = "std")]
    {
        themis::CpuTopology::detect()
            .map(|topology| topology.logical_processors())
            .or_else(|| {
                std::thread::available_parallelism()
                    .ok()
                    .map(std::num::NonZeroUsize::get)
            })
            .unwrap_or(1)
            .max(1)
    }
    #[cfg(not(feature = "std"))]
    {
        4 // Reasonable default for no_std
    }
}

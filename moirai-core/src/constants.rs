//! Global constants for the Moirai concurrency library.
//!
//! This module centralizes all magic numbers and constants following
//! SSOT (Single Source of Truth) and SOC (Separation of Concerns) principles.

/// Percentage conversion factor to maintain precision across metrics
pub const PERCENTAGE_PRECISION_FACTOR: f64 = 100.0;

/// Maximum success rate when no tasks have failed
pub const MAX_SUCCESS_RATE: f64 = 100.0;

/// Default utilization when no workers are available
pub const DEFAULT_UTILIZATION: f64 = 0.0;

/// Bytes to megabytes conversion factor
pub const BYTES_TO_MB_FACTOR: f64 = 1024.0 * 1024.0;

/// Default wait interval for executor polling loop in milliseconds
pub const DEFAULT_POLL_INTERVAL_MS: u64 = 10;

/// Maximum generic spin attempts before falling back to blocking
pub const MAX_SPIN_ATTEMPTS: usize = 64;

/// Maximum backoff iterations for SpinLock (TBB-inspired)
pub const SPINLOCK_MAX_BACKOFF: usize = 64;

/// Maximum spin attempts before yielding to scheduler
pub const SPINLOCK_MAX_SPINS_BEFORE_YIELD: usize = 1000;

/// Cache line size for alignment optimizations
pub const CACHE_LINE_SIZE: usize = 64;

/// Default concurrent map segment count for optimal performance
pub const DEFAULT_CONCURRENT_MAP_SEGMENTS: usize = 16;

/// Default ring buffer capacity (power of 2)
pub const DEFAULT_RING_BUFFER_CAPACITY: usize = 1024;

/// Default MPMC channel capacity
pub const DEFAULT_MPMC_CAPACITY: usize = 1024;

/// Default CPU utilization precision factor (percentage * 100)
pub const CPU_UTILIZATION_PRECISION: u64 = 100;

/// Prime modulo for benchmark variation
pub const BENCHMARK_PRIME_MODULO: usize = 997;

/// Default benchmark operation count
pub const DEFAULT_BENCHMARK_OPS: usize = 1000;

/// Large benchmark data size for performance testing
pub const LARGE_BENCHMARK_SIZE: usize = 10000;

/// SIMD benchmark vector size
pub const SIMD_BENCHMARK_SIZE: usize = 1024;

// Test constants - only included in test builds
/// Test-specific constants for benchmarking and validation.
///
/// These constants are only available during testing to ensure consistent
/// test behavior across different test environments and platforms.
#[cfg(test)]
pub mod test_constants {
    /// Number of test threads for concurrent testing
    pub const TEST_THREAD_COUNT: usize = 10;

    /// Number of operations per test thread
    pub const OPERATIONS_PER_THREAD: usize = 100;

    /// Number of test elements for stress testing
    pub const TEST_ELEMENT_COUNT: usize = 1000;

    /// Test sleep duration multiplier in milliseconds
    pub const TEST_SLEEP_MULTIPLIER_MS: u64 = 10;
}

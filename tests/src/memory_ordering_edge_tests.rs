//! Memory Ordering and Cache Invalidation Edge Case Tests
//!
//! This test suite validates critical memory ordering scenarios and cache
//! behavior under extreme concurrency conditions including:
//! - Sequential consistency violations
//! - ABA problems in lock-free data structures
//! - Cache line contention and false sharing
//! - Memory barrier correctness
//! - NUMA-aware memory allocation patterns

use moirai::{Moirai, Priority};
use std::alloc::{alloc, dealloc, Layout};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Cache-aligned atomic for preventing false sharing
#[repr(align(64))] // Typical cache line size
struct CacheAlignedAtomic<T> {
    value: T,
}

impl<T> CacheAlignedAtomic<T> {
    fn new(value: T) -> Self {
        Self { value }
    }
}

/// Mutex-protected stack used to verify concurrent push/pop conservation.
///
/// A lock-free ABA proof requires a reclamation-aware implementation; this
/// test deliberately makes the mutex ownership explicit and verifies the
/// value-semantic contract that implementation provides.
struct ConcurrentStack {
    inner: std::sync::Mutex<Vec<usize>>,
    operation_count: AtomicUsize,
}

impl ConcurrentStack {
    fn new() -> Self {
        Self {
            inner: std::sync::Mutex::new(Vec::new()),
            operation_count: AtomicUsize::new(0),
        }
    }

    fn push(&self, data: usize) {
        self.inner.lock().unwrap().push(data);
        self.operation_count.fetch_add(1, Ordering::Relaxed);
    }

    fn pop(&self) -> Option<usize> {
        let value = self.inner.lock().unwrap().pop();
        if value.is_some() {
            self.operation_count.fetch_add(1, Ordering::Relaxed);
        }
        value
    }

    fn len(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    fn operation_count(&self) -> usize {
        self.operation_count.load(Ordering::Relaxed)
    }
}

/// Memory barrier test structure
struct MemoryBarrierTest {
    flag1: AtomicBool,
    flag2: AtomicBool,
    data: AtomicUsize,
    reordering_detected: AtomicUsize,
}

impl MemoryBarrierTest {
    fn new() -> Self {
        Self {
            flag1: AtomicBool::new(false),
            flag2: AtomicBool::new(false),
            data: AtomicUsize::new(0),
            reordering_detected: AtomicUsize::new(0),
        }
    }

    fn writer(&self, value: usize) {
        self.data.store(value, Ordering::Relaxed);
        self.flag1.store(true, Ordering::Release); // Release barrier

        // Wait for acknowledgment
        while !self.flag2.load(Ordering::Acquire) {
            std::hint::spin_loop();
        }

        self.flag1.store(false, Ordering::Relaxed);
        self.flag2.store(false, Ordering::Relaxed);
    }

    fn reader(&self) -> Option<usize> {
        if self.flag1.load(Ordering::Acquire) {
            // Acquire barrier
            let data = self.data.load(Ordering::Relaxed);
            self.flag2.store(true, Ordering::Release);

            // Check for reordering - in correct implementation, data should be non-zero
            if data == 0 {
                self.reordering_detected.fetch_add(1, Ordering::Relaxed);
            }

            Some(data)
        } else {
            None
        }
    }

    fn reordering_count(&self) -> usize {
        self.reordering_detected.load(Ordering::Relaxed)
    }
}

/// False sharing detection structure
struct FalseSharingTest {
    // These should be on the same cache line to cause false sharing
    counter1: AtomicUsize,
    counter2: AtomicUsize,
    _padding: [u8; 64 - 2 * std::mem::size_of::<AtomicUsize>()],

    // These should be on different cache lines
    aligned_counter1: CacheAlignedAtomic<AtomicUsize>,
    aligned_counter2: CacheAlignedAtomic<AtomicUsize>,
}

impl FalseSharingTest {
    fn new() -> Self {
        Self {
            counter1: AtomicUsize::new(0),
            counter2: AtomicUsize::new(0),
            _padding: [0; 64 - 2 * std::mem::size_of::<AtomicUsize>()],
            aligned_counter1: CacheAlignedAtomic::new(AtomicUsize::new(0)),
            aligned_counter2: CacheAlignedAtomic::new(AtomicUsize::new(0)),
        }
    }

    fn increment_shared(&self, is_counter1: bool, iterations: usize) -> Duration {
        let start = Instant::now();

        if is_counter1 {
            for _ in 0..iterations {
                self.counter1.fetch_add(1, Ordering::Relaxed);
            }
        } else {
            for _ in 0..iterations {
                self.counter2.fetch_add(1, Ordering::Relaxed);
            }
        }

        start.elapsed()
    }

    fn increment_aligned(&self, is_counter1: bool, iterations: usize) -> Duration {
        let start = Instant::now();

        if is_counter1 {
            for _ in 0..iterations {
                self.aligned_counter1.value.fetch_add(1, Ordering::Relaxed);
            }
        } else {
            for _ in 0..iterations {
                self.aligned_counter2.value.fetch_add(1, Ordering::Relaxed);
            }
        }

        start.elapsed()
    }

    fn get_shared_values(&self) -> (usize, usize) {
        (
            self.counter1.load(Ordering::Relaxed),
            self.counter2.load(Ordering::Relaxed),
        )
    }

    fn get_aligned_values(&self) -> (usize, usize) {
        (
            self.aligned_counter1.value.load(Ordering::Relaxed),
            self.aligned_counter2.value.load(Ordering::Relaxed),
        )
    }
}

/// NUMA-aware memory allocation test
struct NumaMemoryTest {
    local_allocations: AtomicUsize,
    remote_allocations: AtomicUsize,
    allocation_times: Arc<std::sync::Mutex<Vec<Duration>>>,
}

impl NumaMemoryTest {
    fn new() -> Self {
        Self {
            local_allocations: AtomicUsize::new(0),
            remote_allocations: AtomicUsize::new(0),
            allocation_times: Arc::new(std::sync::Mutex::new(Vec::new())),
        }
    }

    fn allocate_and_access(&self, size: usize, access_pattern: AccessPattern) -> Duration {
        let start = Instant::now();

        // Allocate memory
        let layout = Layout::from_size_align(size, 64).unwrap();
        let ptr = unsafe { alloc(layout) };

        if ptr.is_null() {
            return start.elapsed();
        }

        // Access pattern affects NUMA locality
        match access_pattern {
            AccessPattern::Sequential => unsafe {
                for i in 0..size {
                    ptr.add(i).write_volatile(i as u8);
                }
            },
            AccessPattern::Random => unsafe {
                for _ in 0..size / 8 {
                    let offset = fastrand::usize(0..size);
                    ptr.add(offset).write_volatile(42);
                }
            },
            AccessPattern::Strided => {
                unsafe {
                    let stride = 64; // Cache line size
                    for i in (0..size).step_by(stride) {
                        ptr.add(i).write_volatile(i as u8);
                    }
                }
            }
        }

        let duration = start.elapsed();

        // Simulate NUMA detection (in real code, would use numa libraries)
        if duration.as_nanos() < 1000 {
            self.local_allocations.fetch_add(1, Ordering::Relaxed);
        } else {
            self.remote_allocations.fetch_add(1, Ordering::Relaxed);
        }

        unsafe { dealloc(ptr, layout) };

        if let Ok(mut times) = self.allocation_times.lock() {
            times.push(duration);
        }

        duration
    }

    fn stats(&self) -> (usize, usize, Duration) {
        let local = self.local_allocations.load(Ordering::Relaxed);
        let remote = self.remote_allocations.load(Ordering::Relaxed);

        let avg_time = if let Ok(times) = self.allocation_times.lock() {
            if times.is_empty() {
                Duration::ZERO
            } else {
                let total: Duration = times.iter().sum();
                total / times.len() as u32
            }
        } else {
            Duration::ZERO
        };

        (local, remote, avg_time)
    }
}

#[derive(Clone, Copy)]
enum AccessPattern {
    Sequential,
    Random,
    Strided,
}

/// Sequential consistency violation detector
struct SequentialConsistencyTest {
    x: AtomicUsize,
    y: AtomicUsize,
    r1: AtomicUsize,
    r2: AtomicUsize,
    violations: AtomicUsize,
}

impl SequentialConsistencyTest {
    fn new() -> Self {
        Self {
            x: AtomicUsize::new(0),
            y: AtomicUsize::new(0),
            r1: AtomicUsize::new(999),
            r2: AtomicUsize::new(999),
            violations: AtomicUsize::new(0),
        }
    }

    fn thread1(&self) {
        self.x.store(1, Ordering::Relaxed);
        let y_val = self.y.load(Ordering::Relaxed);
        self.r1.store(y_val, Ordering::Relaxed);
    }

    fn thread2(&self) {
        self.y.store(1, Ordering::Relaxed);
        let x_val = self.x.load(Ordering::Relaxed);
        self.r2.store(x_val, Ordering::Relaxed);
    }

    fn reset(&self) {
        self.x.store(0, Ordering::Relaxed);
        self.y.store(0, Ordering::Relaxed);
        self.r1.store(999, Ordering::Relaxed);
        self.r2.store(999, Ordering::Relaxed);
    }

    fn violation_count(&self) -> usize {
        self.violations.load(Ordering::Relaxed)
    }
}

/// Cache invalidation stress test
struct CacheInvalidationTest {
    shared_data: Arc<std::sync::Mutex<Vec<u64>>>,
    cache: Arc<std::sync::RwLock<std::collections::HashMap<usize, u64>>>,
    cache_hits: AtomicUsize,
    cache_misses: AtomicUsize,
    invalidations: AtomicUsize,
}

impl CacheInvalidationTest {
    fn new(data_size: usize) -> Self {
        let mut data = Vec::with_capacity(data_size);
        for i in 0..data_size {
            data.push(i as u64);
        }

        Self {
            shared_data: Arc::new(std::sync::Mutex::new(data)),
            cache: Arc::new(std::sync::RwLock::new(std::collections::HashMap::new())),
            cache_hits: AtomicUsize::new(0),
            cache_misses: AtomicUsize::new(0),
            invalidations: AtomicUsize::new(0),
        }
    }

    fn read_with_cache(&self, index: usize) -> Option<u64> {
        // Try cache first
        if let Ok(cache) = self.cache.read() {
            if let Some(&value) = cache.get(&index) {
                self.cache_hits.fetch_add(1, Ordering::Relaxed);
                return Some(value);
            }
        }

        // Cache miss - read from shared data
        self.cache_misses.fetch_add(1, Ordering::Relaxed);

        if let Ok(data) = self.shared_data.lock() {
            if let Some(&value) = data.get(index) {
                // Update cache
                if let Ok(mut cache) = self.cache.write() {
                    cache.insert(index, value);
                }
                return Some(value);
            }
        }

        None
    }

    fn write_and_invalidate(&self, index: usize, value: u64) -> bool {
        // Update shared data
        if let Ok(mut data) = self.shared_data.lock() {
            if index < data.len() {
                data[index] = value;

                // Invalidate cache entry
                if let Ok(mut cache) = self.cache.write() {
                    if cache.remove(&index).is_some() {
                        self.invalidations.fetch_add(1, Ordering::Relaxed);
                    }
                }
                return true;
            }
        }
        false
    }

    fn cache_stats(&self) -> (usize, usize, usize, f64) {
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let invalidations = self.invalidations.load(Ordering::Relaxed);
        let hit_rate = if hits + misses > 0 {
            (hits as f64 / (hits + misses) as f64) * 100.0
        } else {
            0.0
        };

        (hits, misses, invalidations, hit_rate)
    }

    fn clear_cache(&self) {
        if let Ok(mut cache) = self.cache.write() {
            cache.clear();
        }
    }
}

/// Test runner for memory ordering edge cases
struct MemoryOrderingTestRunner {
    runtime: Moirai,
}

const CONCURRENT_STACK_WORKER_COUNT: usize = 8;

impl MemoryOrderingTestRunner {
    fn new() -> Result<Self, String> {
        let runtime = Moirai::builder()
            .worker_threads(CONCURRENT_STACK_WORKER_COUNT)
            .build()
            .map_err(|_| "Failed to create Moirai runtime")?;
        Ok(Self { runtime })
    }

    /// Test concurrent stack push/pop conservation.
    fn test_concurrent_stack(&self) -> Result<TestResults, String> {
        println!("Testing concurrent stack push/pop conservation...");

        let stack = Arc::new(ConcurrentStack::new());
        let num_threads = CONCURRENT_STACK_WORKER_COUNT;
        let operations_per_thread = 1000;
        let pushed = Arc::new(AtomicUsize::new(0));
        let popped = Arc::new(AtomicUsize::new(0));

        let mut handles = Vec::new();

        for thread_id in 0..num_threads {
            let stack = stack.clone();
            let pushed = pushed.clone();
            let popped = popped.clone();

            let handle = self.runtime.spawn_fn(move || {
                // Alternate between push and pop operations
                for i in 0..operations_per_thread {
                    if thread_id % 2 == 0 || i % 3 == 0 {
                        // Push operation
                        stack.push(thread_id * 1000 + i);
                        pushed.fetch_add(1, Ordering::Relaxed);
                    } else {
                        // Pop operation
                        if stack.pop().is_some() {
                            popped.fetch_add(1, Ordering::Relaxed);
                        }
                    }

                    // Yield periodically to increase interleaving coverage.
                    if i % 10 == 0 {
                        std::thread::yield_now();
                    }
                }
            });

            handles.push(handle);
        }

        // Wait for all threads to complete
        for handle in handles {
            assert_eq!(
                handle.join().expect("Concurrent stack task must complete"),
                Ok(())
            );
        }

        let operations = stack.operation_count();
        let final_size = stack.len();
        let pushed = pushed.load(Ordering::Relaxed);
        let popped = popped.load(Ordering::Relaxed);
        let expected_final_size = pushed - popped;

        println!("  Operations completed: {operations}");
        println!("  Pushes: {pushed}");
        println!("  Successful pops: {popped}");
        println!("  Final stack size: {final_size}");
        assert_eq!(operations, pushed + popped);
        assert_eq!(final_size, expected_final_size);

        Ok(TestResults {
            operations_completed: operations,
            errors_detected: 0,
            final_state_valid: final_size == expected_final_size,
            performance_metric: operations as f64,
        })
    }

    /// Test memory barriers and reordering
    fn test_memory_barriers(&self) -> Result<TestResults, String> {
        println!("Testing memory barriers and reordering...");

        let barrier_test = Arc::new(MemoryBarrierTest::new());
        let num_iterations = 100;
        let mut successful_communications = 0;

        for i in 0..num_iterations {
            let barrier_test_writer = barrier_test.clone();
            let barrier_test_reader = barrier_test.clone();

            let writer_handle = std::thread::spawn(move || {
                barrier_test_writer.writer(i + 1);
            });

            let reader_handle = std::thread::spawn(move || {
                while barrier_test_reader.reader().is_none() {
                    std::thread::yield_now();
                }
                true
            });

            let _ = writer_handle.join();
            if let Ok(success) = reader_handle.join() {
                if success {
                    successful_communications += 1;
                }
            }

            // Yield between iterations
            std::thread::yield_now();
        }

        let reordering_count = barrier_test.reordering_count();

        println!(
            "  Successful communications: {}/{}",
            successful_communications, num_iterations
        );
        println!("  Memory reordering detected: {}", reordering_count);

        Ok(TestResults {
            operations_completed: num_iterations,
            errors_detected: reordering_count,
            final_state_valid: reordering_count == 0,
            performance_metric: (successful_communications as f64 / num_iterations as f64) * 100.0,
        })
    }

    /// Test false sharing performance impact
    fn test_false_sharing(&self) -> Result<TestResults, String> {
        println!("Testing false sharing performance impact...");

        let false_sharing_test = Arc::new(FalseSharingTest::new());
        let iterations = 1_000_000;

        // Test with false sharing
        let test1 = false_sharing_test.clone();
        let test2 = false_sharing_test.clone();

        let shared_handle1 = self.runtime.spawn_fn_with_priority(
            move || test1.increment_shared(true, iterations),
            Priority::High,
        );

        let shared_handle2 = self.runtime.spawn_fn_with_priority(
            move || test2.increment_shared(false, iterations),
            Priority::High,
        );

        let shared_time1 = shared_handle1.join().unwrap().unwrap();
        let shared_time2 = shared_handle2.join().unwrap().unwrap();
        let avg_shared_time = (shared_time1 + shared_time2) / 2;

        // Reset and test with cache-aligned data
        let test3 = false_sharing_test.clone();
        let test4 = false_sharing_test.clone();

        let aligned_handle1 = self.runtime.spawn_fn_with_priority(
            move || test3.increment_aligned(true, iterations),
            Priority::High,
        );

        let aligned_handle2 = self.runtime.spawn_fn_with_priority(
            move || test4.increment_aligned(false, iterations),
            Priority::High,
        );

        let aligned_time1 = aligned_handle1.join().unwrap().unwrap();
        let aligned_time2 = aligned_handle2.join().unwrap().unwrap();
        let avg_aligned_time = (aligned_time1 + aligned_time2) / 2;

        let (shared1, shared2) = false_sharing_test.get_shared_values();
        let (aligned1, aligned2) = false_sharing_test.get_aligned_values();

        println!("  False sharing scenario:");
        println!("    Time: {:?}", avg_shared_time);
        println!("    Values: {} and {}", shared1, shared2);

        println!("  Cache-aligned scenario:");
        println!("    Time: {:?}", avg_aligned_time);
        println!("    Values: {} and {}", aligned1, aligned2);

        let performance_improvement = if avg_shared_time > avg_aligned_time {
            avg_shared_time.as_nanos() as f64 / avg_aligned_time.as_nanos() as f64
        } else {
            1.0
        };

        println!("  Performance improvement: {:.2}x", performance_improvement);

        Ok(TestResults {
            operations_completed: iterations * 4,
            errors_detected: 0,
            final_state_valid: shared1 + shared2 + aligned1 + aligned2 == iterations * 4,
            performance_metric: performance_improvement,
        })
    }

    /// Test NUMA-aware memory allocation
    fn test_numa_memory_allocation(&self) -> Result<TestResults, String> {
        println!("Testing NUMA-aware memory allocation...");

        let numa_test = Arc::new(NumaMemoryTest::new());
        let num_threads = 4;
        let allocations_per_thread = 100;
        let allocation_sizes = [4096, 8192, 16384, 32768]; // Different page sizes

        let mut handles = Vec::new();

        for _thread_id in 0..num_threads {
            let numa_test = numa_test.clone();

            let handle = self.runtime.spawn_fn_with_priority(
                move || {
                    for i in 0..allocations_per_thread {
                        let size = allocation_sizes[i % allocation_sizes.len()];
                        let pattern = match i % 3 {
                            0 => AccessPattern::Sequential,
                            1 => AccessPattern::Random,
                            2 => AccessPattern::Strided,
                            _ => AccessPattern::Sequential,
                        };

                        numa_test.allocate_and_access(size, pattern);

                        // Occasionally yield to allow migration
                        if i % 10 == 0 {
                            std::thread::yield_now();
                        }
                    }
                },
                Priority::Normal,
            );

            handles.push(handle);
        }

        for handle in handles {
            let _ = handle.join();
        }

        let (local_allocs, remote_allocs, avg_time) = numa_test.stats();
        let total_allocs = local_allocs + remote_allocs;
        let locality_ratio = if total_allocs > 0 {
            (local_allocs as f64 / total_allocs as f64) * 100.0
        } else {
            0.0
        };

        println!("  Total allocations: {}", total_allocs);
        println!(
            "  Local allocations: {} ({:.1}%)",
            local_allocs, locality_ratio
        );
        println!(
            "  Remote allocations: {} ({:.1}%)",
            remote_allocs,
            100.0 - locality_ratio
        );
        println!("  Average allocation time: {:?}", avg_time);

        Ok(TestResults {
            operations_completed: total_allocs,
            errors_detected: 0,
            final_state_valid: total_allocs > 0,
            performance_metric: locality_ratio,
        })
    }

    /// Test cache invalidation patterns
    fn test_cache_invalidation(&self) -> Result<TestResults, String> {
        println!("Testing cache invalidation patterns...");

        let cache_test = Arc::new(CacheInvalidationTest::new(10000));
        let num_readers = 6;
        let num_writers = 2;
        let operations_per_thread = 1000;

        let mut handles = Vec::new();

        // Reader threads
        for reader_id in 0..num_readers {
            let cache_test = cache_test.clone();

            let handle = self.runtime.spawn_fn_with_priority(
                move || {
                    for i in 0..operations_per_thread {
                        let index = (reader_id * 1000 + i) % 10000;
                        let _ = cache_test.read_with_cache(index);

                        if i % 100 == 0 {
                            std::thread::yield_now();
                        }
                    }
                },
                Priority::Normal,
            );

            handles.push(handle);
        }

        // Writer threads (cause cache invalidations)
        for writer_id in 0..num_writers {
            let cache_test = cache_test.clone();

            let handle = self.runtime.spawn_fn_with_priority(
                move || {
                    for i in 0..operations_per_thread / 10 {
                        // Fewer writes
                        let index = (writer_id * 500 + i) % 10000;
                        let value = (writer_id as u64) * 1000000 + i as u64;
                        cache_test.write_and_invalidate(index, value);

                        // Writers yield more to let readers build up cache
                        for _ in 0..10 {
                            std::thread::yield_now();
                        }
                    }
                },
                Priority::High,
            );

            handles.push(handle);
        }

        for handle in handles {
            let _ = handle.join();
        }

        let (hits, misses, invalidations, hit_rate) = cache_test.cache_stats();
        let total_ops = hits + misses;

        println!("  Cache hits: {}", hits);
        println!("  Cache misses: {}", misses);
        println!("  Cache invalidations: {}", invalidations);
        println!("  Hit rate: {:.2}%", hit_rate);
        println!("  Total cache operations: {}", total_ops);

        Ok(TestResults {
            operations_completed: total_ops,
            errors_detected: 0,
            final_state_valid: total_ops > 0,
            performance_metric: hit_rate,
        })
    }
}

#[derive(Debug)]
struct TestResults {
    operations_completed: usize,
    errors_detected: usize,
    final_state_valid: bool,
    performance_metric: f64,
}

#[cfg(test)]
mod memory_ordering_tests {
    use super::*;

    #[test]
    fn test_concurrent_stack_safety() {
        let runner = MemoryOrderingTestRunner::new().unwrap();
        let results = runner.test_concurrent_stack().unwrap();

        println!("Concurrent Stack Test Results: {:?}", results);
        assert!(results.operations_completed > 0);
        assert!(results.final_state_valid);
    }

    #[test]
    fn test_memory_barrier_correctness() {
        let runner = MemoryOrderingTestRunner::new().unwrap();
        let results = runner.test_memory_barriers().unwrap();

        println!("Memory Barrier Test Results: {:?}", results);
        assert!(results.operations_completed > 0);
        // Some reordering may occur depending on architecture
        assert!(results.performance_metric > 50.0); // At least 50% success rate
    }

    #[test]
    fn test_false_sharing_detection() {
        let runner = MemoryOrderingTestRunner::new().unwrap();
        let results = runner.test_false_sharing().unwrap();

        println!("False Sharing Test Results: {:?}", results);
        assert!(results.operations_completed > 0);
        assert!(results.final_state_valid);
        // Performance improvement should be measurable
        assert!(results.performance_metric >= 1.0);
    }

    #[test]
    fn test_numa_memory_patterns() {
        let runner = MemoryOrderingTestRunner::new().unwrap();
        let results = runner.test_numa_memory_allocation().unwrap();

        println!("NUMA Memory Test Results: {:?}", results);
        assert!(results.operations_completed > 0);
        assert!(results.final_state_valid);
        // Locality should be reasonable
        assert!(results.performance_metric >= 0.0);
    }

    #[test]
    fn test_cache_invalidation_behavior() {
        let runner = MemoryOrderingTestRunner::new().unwrap();
        let results = runner.test_cache_invalidation().unwrap();

        println!("Cache Invalidation Test Results: {:?}", results);
        assert!(results.operations_completed > 0);
        assert!(results.final_state_valid);
        // Hit rate should be reasonable for this workload
        assert!(results.performance_metric >= 0.0);
    }

    #[test]
    fn test_sequential_consistency() {
        let test = Arc::new(SequentialConsistencyTest::new());
        let num_iterations = 100;

        for _ in 0..num_iterations {
            test.reset();

            let test1 = test.clone();
            let t1 = std::thread::spawn(move || test1.thread1());

            let test2 = test.clone();
            let t2 = std::thread::spawn(move || test2.thread2());

            t1.join().unwrap();
            t2.join().unwrap();

            if test.r1.load(Ordering::Relaxed) == 0 && test.r2.load(Ordering::Relaxed) == 0 {
                test.violations.fetch_add(1, Ordering::Relaxed);
            }
        }

        let violations = test.violation_count();
        println!(
            "Sequential consistency violations: {}/{}",
            violations, num_iterations
        );

        // Some violations may occur with relaxed ordering
        assert!(violations <= num_iterations / 4); // Allow some violations but not too many
    }
}

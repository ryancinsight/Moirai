/// Sharded concurrent hash map with per-segment locking.
pub mod concurrent_hash_map;
/// Futex-backed mutex with OS-assisted parking for fair, contended sections.
pub mod futex_mutex;
/// Sharded pool handing out reusable resources without a global lock.
pub mod resource_pool;
/// Spin lock for critical sections short enough that parking costs more
/// than spinning.
pub mod spin_lock;
/// Barrier that releases once a counted set of tasks has completed.
pub mod wait_group;

#[cfg(test)]
mod tests;

pub use self::concurrent_hash_map::{ConcurrentHashMap, SegmentPoisoned};
pub use self::futex_mutex::{FutexMutex, FutexMutexGuard};
pub use self::resource_pool::{ShardedResourcePool, SizeBounded};
pub use self::spin_lock::{SpinLock, SpinLockGuard};
pub use self::wait_group::WaitGroup;

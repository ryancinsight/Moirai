#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

use super::*;

#[test]
fn test_memory_pool() {
    let pool = MemoryPool::<i32>::new(10);

    // Allocate some items
    let item1 = pool.allocate();
    let item2 = pool.allocate();

    // Pool should be empty initially
    assert_eq!(pool.size(), 0);

    // Return items to pool
    pool.deallocate(item1);
    pool.deallocate(item2);

    // Pool should now have items
    assert_eq!(pool.size(), 2);

    // Allocate again - should reuse from pool
    let _item3 = pool.allocate();
    assert_eq!(pool.size(), 1);
}

#[test]
fn test_memory_pool_real_reuse() {
    // Value-semantic reuse check: the pooled allocation itself is recycled.
    let pool = MemoryPool::<Vec<u8>>::new(4);
    let mut v = vec![0u8; 64];
    let ptr1 = v.as_ptr();
    v.clear();
    pool.deallocate(v);

    let reused = pool.allocate();
    assert_eq!(reused.as_ptr(), ptr1);
    assert_eq!(pool.size(), 0);
}

#[test]
fn test_memory_pool_retention_cap() {
    // deallocate beyond max_size drops the surplus instead of growing.
    let pool = MemoryPool::<i32>::new(2);
    pool.deallocate(1);
    pool.deallocate(2);
    pool.deallocate(3); // beyond cap: dropped
    assert_eq!(pool.size(), 2);

    // The two retained values come back out (LIFO), then Default.
    assert_eq!(pool.allocate(), 2);
    assert_eq!(pool.allocate(), 1);
    assert_eq!(pool.allocate(), 0); // empty pool: i32::default()
}

#[test]
fn cache_aligned_allocation_refuses_a_byte_size_that_wraps() {
    // `8 * (2^61 + 1)` wraps to 8 in a release build, which would allocate one
    // element and report 2^61 + 1 of them.
    assert!(CacheAlignedAllocator::allocate::<u64>((1_usize << 61) + 1).is_none());
    assert!(CacheAlignedAllocator::allocate::<u64>(usize::MAX).is_none());
}

#[test]
fn cache_aligned_allocation_is_line_aligned_and_round_trips() {
    let count = 7;
    let ptr = CacheAlignedAllocator::allocate::<u64>(count).expect("small allocation succeeds");
    assert_eq!(ptr.as_ptr() as usize % CACHE_LINE_SIZE, 0);
    // SAFETY: `ptr` came from `allocate::<u64>(count)` above, is not used after
    // this call, and nothing else references it.
    unsafe { CacheAlignedAllocator::deallocate(ptr, count) };
}

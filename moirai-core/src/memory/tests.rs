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
fn test_unified_ring_buffer() {
    let buffer = UnifiedRingBuffer::<i32>::new(8).unwrap();

    // Test basic operations
    assert!(buffer.is_empty());
    assert_eq!(buffer.len(), 0);

    // Push some items
    assert!(buffer.try_push(1).is_ok());
    assert!(buffer.try_push(2).is_ok());
    assert_eq!(buffer.len(), 2);

    // Pop items
    assert_eq!(buffer.try_pop(), Some(1));
    assert_eq!(buffer.try_pop(), Some(2));
    assert!(buffer.is_empty());
}

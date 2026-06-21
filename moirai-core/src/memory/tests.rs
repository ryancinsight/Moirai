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
fn test_global_memory_manager_real_pooling() {
    let manager = GlobalMemoryManager::instance();

    // Allocate a vector of size 100 (matches pool index 4, size 65..=128)
    let vec1 = manager.allocate(100).unwrap();
    assert_eq!(vec1.len(), 100);
    assert!(vec1.capacity() >= 128);

    // Keep track of the raw pointer of vec1's heap allocation
    let ptr1 = vec1.as_ptr();

    // Return it to the pool
    manager.deallocate(vec1);

    // Allocate again - it should reuse the same backing allocation!
    let vec2 = manager.allocate(100).unwrap();
    assert_eq!(vec2.len(), 100);
    let ptr2 = vec2.as_ptr();

    // Under real pooling, the memory allocation is recycled, so the pointer should be the same
    assert_eq!(ptr1, ptr2);
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

use super::global::GlobalPool;
use super::slab::SlabAllocator;
use super::stack::LockFreeStack;
use super::wrapper::TaskWrapper;
use crate::{Priority, TaskId};

#[test]
fn test_lock_free_stack() {
    let stack = LockFreeStack::new();

    // Test push/pop
    assert_eq!(stack.push(1), Ok(()));
    assert_eq!(stack.push(2), Ok(()));
    assert_eq!(stack.push(3), Ok(()));

    assert_eq!(stack.len(), 3);
    assert_eq!(stack.pop(), Some(3));
    assert_eq!(stack.pop(), Some(2));
    assert_eq!(stack.pop(), Some(1));
    assert_eq!(stack.pop(), None);
    assert_eq!(stack.len(), 0);
}

#[test]
fn test_lock_free_stack_capacity() {
    // Default capacity is the documented constant, not an eager 65536.
    let stack = LockFreeStack::<i32>::new();
    assert_eq!(stack.capacity(), super::stack::DEFAULT_STACK_CAPACITY);

    // Explicit capacity is honored exactly; exhaustion returns the item back.
    let small = LockFreeStack::with_capacity(2);
    assert_eq!(small.capacity(), 2);
    assert_eq!(small.push(10), Ok(()));
    assert_eq!(small.push(20), Ok(()));
    assert_eq!(small.push(30), Err(30)); // full: value handed back, not dropped
    assert_eq!(small.len(), 2);

    // Popping frees a slot; push succeeds again.
    assert_eq!(small.pop(), Some(20));
    assert_eq!(small.push(40), Ok(()));
    assert_eq!(small.pop(), Some(40));
    assert_eq!(small.pop(), Some(10));
    assert_eq!(small.pop(), None);

    // Zero-capacity stack rejects every push.
    let empty = LockFreeStack::with_capacity(0);
    assert_eq!(empty.push(1), Err(1));
    assert_eq!(empty.pop(), None);
}

#[test]
fn test_slab_allocator() {
    let slab = SlabAllocator::new(10);

    // Test insertion
    let idx1 = slab.insert("hello").unwrap();
    let idx2 = slab.insert("world").unwrap();

    assert_eq!(unsafe { slab.get(idx1) }, Some(&"hello"));
    assert_eq!(unsafe { slab.get(idx2) }, Some(&"world"));
    assert_eq!(slab.len(), 2);

    // Test removal
    assert_eq!(slab.remove(idx1), Some("hello"));
    assert_eq!(slab.len(), 1);
    assert_eq!(unsafe { slab.get(idx1) }, None);

    // Test reuse of slot
    let idx3 = slab.insert("reused").unwrap();
    assert_eq!(idx3, idx1); // Should reuse the freed slot
}

#[test]
fn test_task_wrapper() {
    let mut wrapper = TaskWrapper::<String>::new();

    wrapper.init("test".to_string(), TaskId(1), Priority::High);
    assert_eq!(wrapper.task_id(), Some(TaskId(1)));
    assert_eq!(wrapper.priority(), Priority::High);
    assert_eq!(wrapper.take(), Some("test".to_string()));

    wrapper.reset();
    assert_eq!(wrapper.task_id(), None);
    assert_eq!(wrapper.reset_count(), 1);
}

#[test]
fn test_global_pool() {
    let pool = GlobalPool::new(10);

    // Return some objects to the pool
    pool.put(vec![1, 2, 3]);
    pool.put(vec![4, 5, 6]);

    // Get objects (should reuse)
    let obj1 = pool.get();
    assert!(obj1.is_empty() || obj1 == vec![4, 5, 6]);

    let obj2 = pool.get();
    assert!(obj2.is_empty() || obj2 == vec![1, 2, 3]);
}

#[test]
fn test_global_pool_max_size_wired_through() {
    // GlobalPool::new(max_size) sizes the underlying stack to max_size slots;
    // returns beyond the cap are dropped, not retained.
    let pool = GlobalPool::new(1);
    pool.put(vec![1]);
    pool.put(vec![2]); // beyond cap: dropped

    assert_eq!(pool.get(), vec![1]);
    assert_eq!(pool.get(), Vec::<i32>::new()); // pool empty: Default
}

#[test]
fn test_concurrent_stack() {
    use std::sync::Arc;
    use std::thread;

    let stack = Arc::new(LockFreeStack::new());
    let mut handles = vec![];

    // Spawn producers
    for i in 0..4 {
        let stack = stack.clone();
        handles.push(thread::spawn(move || {
            for j in 0..100 {
                stack
                    .push(i * 100 + j)
                    .expect("invariant: 400 pushes fit the default 1024-slot capacity");
            }
        }));
    }

    // Wait for producers
    for handle in handles {
        handle.join().unwrap();
    }

    assert_eq!(stack.len(), 400);

    // Verify all items can be popped
    let mut count = 0;
    while stack.pop().is_some() {
        count += 1;
    }
    assert_eq!(count, 400);
}

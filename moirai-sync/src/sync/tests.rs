use std::hint;
use std::sync::Arc;
use std::thread;

use moirai_core::pool::LockFreeStack;

use super::concurrent_hash_map::ConcurrentHashMap;
use super::futex_mutex::FutexMutex;
use super::spin_lock::SpinLock;
use super::wait_group::WaitGroup;

const TEST_THREAD_COUNT: usize = 10;
const OPERATIONS_PER_THREAD: usize = 100;
const TEST_ELEMENT_COUNT: usize = 1000;
const TEST_SLEEP_MULTIPLIER_MS: u64 = 10;

#[test]
fn test_wait_group() {
    let wg = Arc::new(WaitGroup::new());
    let mut handles = vec![];

    wg.add(3);

    for i in 0..3 {
        let wg = wg.clone();
        handles.push(thread::spawn(move || {
            thread::sleep(std::time::Duration::from_millis(
                i * TEST_SLEEP_MULTIPLIER_MS,
            ));
            wg.done();
        }));
    }

    wg.wait();

    for handle in handles {
        handle.join().unwrap();
    }
}

#[test]
fn test_futex_mutex() {
    let mutex = Arc::new(FutexMutex::new(0));
    let mut handles = vec![];

    for _ in 0..TEST_THREAD_COUNT {
        let mutex = mutex.clone();
        handles.push(thread::spawn(move || {
            for _ in 0..OPERATIONS_PER_THREAD {
                let mut guard = mutex.lock();
                *guard += 1;
            }
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    assert_eq!(*mutex.lock(), 1000);
}

#[test]
fn test_lock_free_stack() {
    let stack = Arc::new(LockFreeStack::new());
    let mut handles = vec![];

    // Push from multiple threads
    for i in 0..10 {
        let stack = stack.clone();
        handles.push(thread::spawn(move || {
            stack.push(i);
        }));
    }

    for handle in handles {
        handle.join().unwrap();
    }

    // Pop all items
    let mut items = vec![];
    while let Some(item) = stack.pop() {
        items.push(item);
    }

    items.sort();
    assert_eq!(items, (0..10).collect::<Vec<_>>());
}

#[test]
fn test_concurrent_hashmap() {
    let map = ConcurrentHashMap::new();

    // Insert some values
    assert!(map.insert("key1", 100).unwrap().is_none());
    assert!(map.insert("key2", 200).unwrap().is_none());

    // Test retrieval
    assert_eq!(map.get(&"key1").unwrap(), Some(100));
    assert_eq!(map.get(&"key2").unwrap(), Some(200));
    assert_eq!(map.get(&"key3").unwrap(), None);

    // Test removal
    assert_eq!(map.remove(&"key1").unwrap(), Some(100));
    assert_eq!(map.get(&"key1").unwrap(), None);
}

#[test]
fn test_concurrent_hashmap_segment_distribution() {
    use std::collections::HashSet;

    // Create a map with 16 segments
    let map = ConcurrentHashMap::<i32, i32>::with_segments(16);

    // Track which segments are used
    let mut segments_used = HashSet::new();

    // Insert many keys and track segment distribution
    for i in 0..TEST_ELEMENT_COUNT {
        let key = i as i32;
        map.insert(key, key).unwrap();
        let segment_idx = map.segment_index(&key);
        segments_used.insert(segment_idx);
    }

    // With proper distribution, we should use most segments
    // With 1000 keys across 16 segments, we expect to use all segments
    assert!(
        segments_used.len() >= 14,
        "Poor segment distribution: only {} of 16 segments used",
        segments_used.len()
    );

    // Verify all keys can be retrieved
    for i in 0..TEST_ELEMENT_COUNT {
        let key = i as i32;
        assert_eq!(map.get(&key).unwrap(), Some(key));
    }
}

#[test]
fn test_spinlock_basic_functionality() {
    let lock = SpinLock::new(0);

    // Test basic lock/unlock
    {
        let mut guard = lock.lock();
        *guard = 42;
    }

    // Test that value was updated
    {
        let guard = lock.lock();
        assert_eq!(*guard, 42);
    }
}

#[test]
fn test_spinlock_try_lock() {
    let lock = SpinLock::new(0);

    // Should be able to try_lock on unlocked
    let guard1 = lock.try_lock();
    assert!(guard1.is_some());

    // Should fail to try_lock when locked
    let guard2 = lock.try_lock();
    assert!(guard2.is_none());

    // Should succeed after first guard is dropped
    drop(guard1);
    let guard3 = lock.try_lock();
    assert!(guard3.is_some());
}

#[test]
fn test_spinlock_contention() {
    let lock = Arc::new(SpinLock::new(0));
    let mut handles = vec![];

    // Spawn threads that increment a counter
    for _ in 0..TEST_THREAD_COUNT {
        let lock = lock.clone();
        handles.push(thread::spawn(move || {
            for _ in 0..OPERATIONS_PER_THREAD {
                let mut guard = lock.lock();
                *guard += 1;
                // Hold the lock briefly to create contention
                for _ in 0..10 {
                    hint::spin_loop();
                }
            }
        }));
    }

    // Wait for all threads to complete
    for handle in handles {
        handle.join().unwrap();
    }

    // Verify final count
    let guard = lock.lock();
    assert_eq!(*guard, TEST_THREAD_COUNT * OPERATIONS_PER_THREAD);
}

#[test]
fn test_spinlock_drop_behavior() {
    let lock = SpinLock::new(vec![1, 2, 3]);

    // Test that guard properly derefs
    {
        let guard = lock.lock();
        assert_eq!(guard.len(), 3);
        assert_eq!(guard[0], 1);
    }

    // Test that guard properly derefs mutably
    {
        let mut guard = lock.lock();
        guard.push(4);
        assert_eq!(guard.len(), 4);
    }

    // Verify changes persisted
    {
        let guard = lock.lock();
        assert_eq!(*guard, vec![1, 2, 3, 4]);
    }
}

#[test]
fn test_spinlock_send_sync() {
    // Test that SpinLock implements Send + Sync
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<SpinLock<i32>>();

    // Test that we can move SpinLock across threads
    let lock = SpinLock::new(42);
    let handle = thread::spawn(move || {
        let guard = lock.lock();
        *guard
    });

    assert_eq!(handle.join().unwrap(), 42);
}

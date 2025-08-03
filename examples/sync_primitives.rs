//! Synchronization primitives example for Moirai
//!
//! This example demonstrates:
//! - Fast mutex usage
//! - Concurrent hash map
//! - Lock-free stack and queue
//! - Barriers and wait groups

use moirai_sync::{
    FastMutex, WaitGroup, Barrier,
    ConcurrentHashMap, LockFreeStack
};
use moirai_utils::LockFreeQueue;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

fn main() {
    println!("Moirai Synchronization Primitives");
    println!("=================================");
    
    // Example 1: Fast Mutex
    println!("\n1. Fast Mutex Example:");
    
    let counter = Arc::new(FastMutex::new(0));
    let mut handles = vec![];
    
    for i in 0..10 {
        let counter_clone = Arc::clone(&counter);
        let handle = thread::spawn(move || {
            let mut guard = counter_clone.lock();
            *guard += 1;
            println!("  Thread {}: Counter = {}", i, *guard);
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    println!("  Final counter value: {}", *counter.lock());
    
    // Example 2: Concurrent HashMap
    println!("\n2. Concurrent HashMap Example:");
    
    let map = Arc::new(ConcurrentHashMap::<String, i32>::new());
    let mut handles = vec![];
    
    // Insert values concurrently
    for i in 0..5 {
        let map_clone = Arc::clone(&map);
        let handle = thread::spawn(move || {
            for j in 0..10 {
                let key = format!("thread_{}_item_{}", i, j);
                map_clone.insert(key.clone(), i * 10 + j);
                println!("  Thread {}: Inserted {}", i, key);
            }
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Read values
    // Note: ConcurrentHashMap doesn't have a len() method
    println!("  Map operations completed");
    if let Some(value) = map.get(&"thread_0_item_0".to_string()) {
        println!("  Sample value: thread_0_item_0 = {}", value);
    }
    
    // Example 3: Lock-Free Stack
    println!("\n3. Lock-Free Stack Example:");
    
    let stack = Arc::new(LockFreeStack::new());
    let mut handles = vec![];
    
    // Push values
    for i in 0..5 {
        let stack_clone = Arc::clone(&stack);
        let handle = thread::spawn(move || {
            for j in 0..5 {
                let value = i * 10 + j;
                stack_clone.push(value);
                println!("  Thread {}: Pushed {}", i, value);
            }
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Pop values
    println!("  Popping values from stack:");
    for _ in 0..5 {
        if let Some(value) = stack.pop() {
            println!("    Popped: {}", value);
        }
    }
    
    // Example 4: Lock-Free Queue
    println!("\n4. Lock-Free Queue Example:");
    
    let queue: Arc<LockFreeQueue<i32>> = Arc::new(LockFreeQueue::new());
    
    // Producer thread
    let queue_producer = Arc::clone(&queue);
    let producer = thread::spawn(move || {
        for i in 1..=10 {
            queue_producer.enqueue(i);
            println!("  Producer: Enqueued {}", i);
            thread::sleep(Duration::from_millis(10));
        }
    });
    
    // Consumer thread
    let queue_consumer = Arc::clone(&queue);
    let consumer = thread::spawn(move || {
        thread::sleep(Duration::from_millis(50)); // Let producer get ahead
        let mut consumed = 0;
        while consumed < 10 {
            if let Some(value) = queue_consumer.try_dequeue() {
                println!("  Consumer: Dequeued {}", value);
                consumed += 1;
            } else {
                thread::sleep(Duration::from_millis(20));
            }
        }
    });
    
    producer.join().unwrap();
    consumer.join().unwrap();
    
    // Example 5: Barrier
    println!("\n5. Barrier Example:");
    
    let barrier = Arc::new(Barrier::new(3));
    let mut handles = vec![];
    
    for i in 0..3 {
        let barrier_clone = Arc::clone(&barrier);
        let handle = thread::spawn(move || {
            println!("  Thread {}: Working...", i);
            thread::sleep(Duration::from_millis((i + 1) as u64 * 100));
            println!("  Thread {}: Waiting at barrier", i);
            
            barrier_clone.wait();
            
            println!("  Thread {}: Passed barrier!", i);
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    // Example 6: WaitGroup
    println!("\n6. WaitGroup Example:");
    
    let wg = Arc::new(WaitGroup::new());
    
    for i in 0..5 {
        wg.add(1);
        let wg_clone = Arc::clone(&wg);
        
        thread::spawn(move || {
            println!("  Worker {}: Starting task", i);
            thread::sleep(Duration::from_millis((5 - i) as u64 * 50));
            println!("  Worker {}: Task complete", i);
            wg_clone.done();
        });
    }
    
    println!("  Main: Waiting for all workers...");
    wg.wait();
    println!("  Main: All workers completed!");
    
    // Example 7: Performance comparison
    println!("\n7. Performance Comparison (FastMutex vs std::sync::Mutex):");
    
    let iterations = 100_000;
    
    // FastMutex benchmark
    let fast_mutex = Arc::new(FastMutex::new(0));
    let start = Instant::now();
    
    let mut handles = vec![];
    for _ in 0..4 {
        let mutex_clone = Arc::clone(&fast_mutex);
        let handle = thread::spawn(move || {
            for _ in 0..iterations / 4 {
                let mut guard = mutex_clone.lock();
                *guard += 1;
            }
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    let fast_time = start.elapsed();
    
    // std::sync::Mutex benchmark
    let std_mutex = Arc::new(std::sync::Mutex::new(0));
    let start = Instant::now();
    
    let mut handles = vec![];
    for _ in 0..4 {
        let mutex_clone = Arc::clone(&std_mutex);
        let handle = thread::spawn(move || {
            for _ in 0..iterations / 4 {
                let mut guard = mutex_clone.lock().unwrap();
                *guard += 1;
            }
        });
        handles.push(handle);
    }
    
    for handle in handles {
        handle.join().unwrap();
    }
    
    let std_time = start.elapsed();
    
    println!("  FastMutex:      {:?} ({} ops)", fast_time, *fast_mutex.lock());
    println!("  std::sync::Mutex: {:?} ({} ops)", std_time, *std_mutex.lock().unwrap());
    println!("  Speedup: {:.2}x", std_time.as_secs_f64() / fast_time.as_secs_f64());
}
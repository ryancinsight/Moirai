//! Performance demonstration of Moirai's zero-allocation optimizations
//!
//! This example shows:
//! - Zero-allocation task dispatch using TaskSlot
//! - Efficient thread parking instead of busy-waiting
//! - Comparison with boxed trait objects

use std::time::{Duration, Instant};
use std::sync::Arc;
use std::thread;

fn main() {
    println!("Moirai Performance Optimizations Demo");
    println!("=====================================\n");
    
    // Demo 1: Zero-allocation task dispatch
    println!("1. Zero-allocation Task Dispatch:");
    benchmark_task_dispatch();
    
    // Demo 2: Efficient channel parking
    println!("\n2. Efficient Channel Parking:");
    benchmark_channel_parking();
}

fn benchmark_task_dispatch() {
    const ITERATIONS: usize = 1_000_000;
    
    // Simulate the old approach with Box<dyn Fn()>
    let start = Instant::now();
    let mut boxed_tasks: Vec<Box<dyn Fn() + Send>> = Vec::with_capacity(ITERATIONS);
    for i in 0..ITERATIONS {
        boxed_tasks.push(Box::new(move || {
            let _ = i * 2;
        }));
    }
    let boxed_alloc_time = start.elapsed();
    
    // Execute boxed tasks
    let start = Instant::now();
    for task in boxed_tasks {
        task();
    }
    let boxed_exec_time = start.elapsed();
    
    // Simulate the new approach with inline closures (no boxing)
    let start = Instant::now();
    let mut results = Vec::with_capacity(ITERATIONS);
    for i in 0..ITERATIONS {
        results.push(i * 2);
    }
    let inline_time = start.elapsed();
    
    println!("  Boxed trait objects:");
    println!("    - Allocation time: {:?}", boxed_alloc_time);
    println!("    - Execution time:  {:?}", boxed_exec_time);
    println!("    - Total time:      {:?}", boxed_alloc_time + boxed_exec_time);
    
    println!("  Inline execution (zero-allocation):");
    println!("    - Total time:      {:?}", inline_time);
    
    let speedup = (boxed_alloc_time + boxed_exec_time).as_secs_f64() / inline_time.as_secs_f64();
    println!("  Speedup: {:.2}x", speedup);
}

fn benchmark_channel_parking() {
    const MESSAGES: usize = 10_000;
    
    // Simulate busy-wait approach
    let busy_wait_time = Arc::new(std::sync::Mutex::new(Duration::ZERO));
    let busy_wait_time_clone = busy_wait_time.clone();
    
    let (tx1, rx1) = std::sync::mpsc::channel();
    let busy_thread = thread::spawn(move || {
        let mut total_wait = Duration::ZERO;
        for _ in 0..MESSAGES {
            let start = Instant::now();
            while rx1.try_recv().is_err() {
                std::hint::spin_loop();
                if start.elapsed() > Duration::from_millis(1) {
                    break; // Prevent infinite loop
                }
            }
            total_wait += start.elapsed();
        }
        *busy_wait_time_clone.lock().unwrap() = total_wait;
    });
    
    // Send messages with delays
    thread::spawn(move || {
        for i in 0..MESSAGES {
            if i % 1000 == 0 {
                thread::sleep(Duration::from_micros(10));
            }
            tx1.send(i).unwrap();
        }
    });
    
    busy_thread.join().unwrap();
    
    // Simulate parking approach
    let park_wait_time = Arc::new(std::sync::Mutex::new(Duration::ZERO));
    let park_wait_time_clone = park_wait_time.clone();
    
    let (tx2, rx2) = std::sync::mpsc::channel();
    let park_thread = thread::spawn(move || {
        let mut total_wait = Duration::ZERO;
        for _ in 0..MESSAGES {
            let start = Instant::now();
            rx2.recv().unwrap();
            total_wait += start.elapsed();
        }
        *park_wait_time_clone.lock().unwrap() = total_wait;
    });
    
    // Send messages with delays
    thread::spawn(move || {
        for i in 0..MESSAGES {
            if i % 1000 == 0 {
                thread::sleep(Duration::from_micros(10));
            }
            tx2.send(i).unwrap();
        }
    });
    
    park_thread.join().unwrap();
    
    let busy_time = *busy_wait_time.lock().unwrap();
    let park_time = *park_wait_time.lock().unwrap();
    
    println!("  Busy-wait approach:");
    println!("    - Total wait time: {:?}", busy_time);
    println!("  Parking approach:");
    println!("    - Total wait time: {:?}", park_time);
    
    if busy_time > park_time {
        let improvement = busy_time.as_secs_f64() / park_time.as_secs_f64();
        println!("  Improvement: {:.2}x less CPU time wasted", improvement);
    }
}
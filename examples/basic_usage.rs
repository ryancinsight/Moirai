//! Basic usage example for Moirai concurrency library
//!
//! This example demonstrates:
//! - Creating a runtime
//! - Spawning parallel tasks
//! - Using channels for communication
//! - Async task execution

use moirai::{Moirai, Priority};
use std::time::Duration;

fn main() {
    // Create a new Moirai runtime with default configuration
    let runtime = Moirai::new().expect("Failed to create Moirai runtime");
    
    println!("Moirai Basic Usage Example");
    println!("==========================");
    
    // Example 1: Spawning simple tasks
    println!("\n1. Spawning simple tasks:");
    
    let handle1 = runtime.spawn_fn(|| {
        println!("  Task 1: Computing sum...");
        let sum: i32 = (1..=100).sum();
        println!("  Task 1: Sum of 1-100 = {}", sum);
        sum
    });
    
    let handle2 = runtime.spawn_fn(|| {
        println!("  Task 2: Computing product...");
        let product: i32 = (1..=5).product();
        println!("  Task 2: Product of 1-5 = {}", product);
        product
    });
    
    // Wait for tasks to complete
    let result1 = handle1.join().unwrap().unwrap();
    let result2 = handle2.join().unwrap().unwrap();
    println!("  Results: {} and {}", result1, result2);
    
    // Example 2: Using channels
    println!("\n2. Using channels for communication:");
    
    let (tx, rx) = runtime.channel::<i32>();
    
    runtime.spawn_fn(move || {
        println!("  Producer: Sending values...");
        for i in 1..=5 {
            tx.send(i).unwrap();
            println!("  Producer: Sent {}", i);
        }
    });
    
    runtime.spawn_fn(move || {
        println!("  Consumer: Receiving values...");
        while let Ok(value) = rx.recv() {
            println!("  Consumer: Received {}", value);
        }
    });
    
    // Give tasks time to execute
    std::thread::sleep(Duration::from_millis(100));
    
    // Example 3: Priority-based execution
    println!("\n3. Priority-based task execution:");
    
    runtime.spawn_fn_with_priority(|| {
        println!("  Low priority task executing");
    }, Priority::Low);
    
    runtime.spawn_fn_with_priority(|| {
        println!("  Critical priority task executing");
    }, Priority::Critical);
    
    runtime.spawn_fn_with_priority(|| {
        println!("  Normal priority task executing");
    }, Priority::Normal);
    
    // Example 4: Async execution
    println!("\n4. Async task execution:");
    
    let async_handle = runtime.spawn_async(async {
        println!("  Async task: Starting...");
        // Simulate async work
        moirai::sleep(Duration::from_millis(50)).await;
        println!("  Async task: Completed after delay");
        42
    });
    
    // Wait for the async task to complete and get its result.
    let result = async_handle.join().expect("Async task failed").expect("Task execution failed");
    println!("  Async result: {}", result);
    
    // Example 5: Parallel iteration (using moirai_iter if available)
    println!("\n5. Parallel iteration:");
    
    #[cfg(feature = "iter")]
    {

        
        let numbers: Vec<i32> = (1..=10).collect();
        
        // For now, use standard parallel iteration
        let squared: Vec<i32> = numbers
            .into_iter()
            .map(|n| {
                println!("  Squaring {} in parallel", n);
                n * n
            })
            .collect();
        
        println!("  Squared numbers: {:?}", squared);
    }
    
    #[cfg(not(feature = "iter"))]
    {
        println!("  (Iterator feature not enabled)");
    }
    
    // Shutdown the runtime gracefully
    runtime.shutdown();
    println!("\nRuntime shutdown complete.");
}
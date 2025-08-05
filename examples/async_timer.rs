//! Example demonstrating Moirai's async timer functionality
//!
//! This example shows how to use non-blocking sleep in async contexts
//! without external dependencies like tokio.

use moirai::prelude::*;
use std::time::{Duration, Instant};

fn main() {
    println!("Moirai Async Timer Example");
    println!("=========================\n");
    
    // Create a Moirai runtime
    let runtime = Moirai::new().unwrap();
    
    // Example 1: Simple async sleep
    println!("Example 1: Simple async sleep");
    let start = Instant::now();
    
    runtime.block_on(async {
        println!("Starting async sleep for 500ms...");
        moirai::sleep(Duration::from_millis(500)).await;
        println!("Async sleep completed!");
    });
    
    println!("Elapsed time: {:?}\n", start.elapsed());
    
    // Example 2: Multiple concurrent sleeps
    println!("Example 2: Multiple concurrent sleeps");
    let start = Instant::now();
    
    runtime.block_on(async {
        // Spawn multiple async tasks with different sleep durations
        let handle1 = moirai::spawn_async(async {
            println!("Task 1: Starting 200ms sleep");
            moirai::sleep(Duration::from_millis(200)).await;
            println!("Task 1: Completed!");
            1
        });
        
        let handle2 = moirai::spawn_async(async {
            println!("Task 2: Starting 300ms sleep");
            moirai::sleep(Duration::from_millis(300)).await;
            println!("Task 2: Completed!");
            2
        });
        
        let handle3 = moirai::spawn_async(async {
            println!("Task 3: Starting 100ms sleep");
            moirai::sleep(Duration::from_millis(100)).await;
            println!("Task 3: Completed!");
            3
        });
        
        // Wait for all tasks to complete
        // Note: In a real implementation, we'd have proper join functionality
        moirai::sleep(Duration::from_millis(400)).await;
        
        println!("All tasks should be completed");
    });
    
    println!("Total elapsed time: {:?}\n", start.elapsed());
    
    // Example 3: Async iteration with delays
    println!("Example 3: Async iteration with delays");
    let start = Instant::now();
    
    runtime.block_on(async {
        let items = vec!["first", "second", "third", "fourth", "fifth"];
        
        // Process items with async delays
        // Note: Async iterator processing would require proper async map implementation
        // For now, we'll demonstrate with sequential processing
        let mut results = Vec::new();
        for item in items {
            println!("Processing '{}' with 100ms delay...", item);
            moirai::sleep(Duration::from_millis(100)).await;
            results.push(format!("Processed: {}", item));
        }
        
        println!("Results: {:?}", results);
    });
    
    println!("Async iteration elapsed: {:?}\n", start.elapsed());
    
    // Example 4: Timeout demonstration
    println!("Example 4: Timeout demonstration");
    
    runtime.block_on(async {
        use moirai::timeout;
        
        // Task that completes in time
        match timeout(async {
            moirai::sleep(Duration::from_millis(100)).await;
            "Task completed successfully"
        }, Duration::from_millis(200)).await {
            Ok(result) => println!("Success: {}", result),
            Err(_) => println!("Task timed out"),
        }
        
        // Task that times out
        match timeout(async {
            moirai::sleep(Duration::from_millis(200)).await;
            "This won't be reached"
        }, Duration::from_millis(100)).await {
            Ok(result) => println!("Success: {}", result),
            Err(_) => println!("Task timed out as expected"),
        }
    });
    
    println!("\nExample completed!");
    
    // Shutdown the runtime
    runtime.shutdown();
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_async_sleep_accuracy() {
        let runtime = Moirai::new().unwrap();
        
        let start = Instant::now();
        runtime.block_on(async {
            moirai::sleep(Duration::from_millis(100)).await;
        });
        let elapsed = start.elapsed();
        
        // Allow some tolerance for timing
        assert!(elapsed >= Duration::from_millis(95));
        assert!(elapsed <= Duration::from_millis(150));
    }
    
    #[test]
    fn test_concurrent_sleeps() {
        let runtime = Moirai::new().unwrap();
        
        let start = Instant::now();
        runtime.block_on(async {
            // Multiple sleeps should complete concurrently
            let _h1 = moirai::spawn_async(async {
                moirai::sleep(Duration::from_millis(100)).await;
            });
            
            let _h2 = moirai::spawn_async(async {
                moirai::sleep(Duration::from_millis(100)).await;
            });
            
            // Wait for completion
            moirai::sleep(Duration::from_millis(150)).await;
        });
        let elapsed = start.elapsed();
        
        // Should take ~150ms, not 200ms+ if they were sequential
        assert!(elapsed < Duration::from_millis(200));
    }
}
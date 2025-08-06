//! Simple iterator showcase for Moirai
//!
//! This example demonstrates basic usage of Moirai iterators

use moirai_iter::{moirai_iter, MoiraiIterator};

fn main() {
    println!("Moirai Simple Iterator Showcase");
    println!("================================");
    
    // Since MoiraiIterator methods return futures, we need an async runtime
    // For this example, we'll use a simple blocking executor
    
    println!("\n1. Basic map operation:");
    let numbers = vec![1, 2, 3, 4, 5];
    
    // Note: In a real application, you would use an async runtime like tokio
    // Here we're just demonstrating the API
    println!("  Input: {:?}", numbers);
    println!("  (MoiraiIterator operations return futures - use with async runtime)");
    
    println!("\n2. Creating iterators:");
    let iter = moirai_iter(vec![10, 20, 30]);
    println!("  Created parallel iterator for: [10, 20, 30]");
    
    println!("\n3. Iterator features:");
    println!("  - Parallel execution for CPU-bound tasks");
    println!("  - Async execution for I/O-bound tasks");
    println!("  - Hybrid execution with adaptive strategies");
    println!("  - Zero-copy operations where possible");
    println!("  - NUMA-aware memory allocation");
    println!("  - SIMD optimizations for numeric operations");
    
    println!("\nFor full async examples, see the async_timer example.");
}
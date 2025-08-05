//! Simple Iterator showcase for Moirai
//!
//! This example demonstrates basic usage of Moirai iterators

use moirai_iter::{moirai_iter, moirai_iter_async, moirai_iter_hybrid, MoiraiIterator};
use std::time::Instant;

fn main() {
    println!("Moirai Iterator System - Simple Showcase");
    println!("========================================");
    
    // Create runtime
    let runtime = moirai::Moirai::new().unwrap();
    
    // Example 1: Basic parallel iteration
    println!("\n1. Basic Parallel Iterator:");
    let numbers: Vec<i32> = (1..=100).collect();
    let start = Instant::now();
    
    let result = runtime.block_on(async {
        moirai_iter(numbers)
            .for_each(|x| {
                // Process each item
                let _square = x * x;
            })
            .await
    });
    
    println!("  Parallel processing completed in {:?}", start.elapsed());
    
    // Example 2: Map operation
    println!("\n2. Map Operation:");
    let data: Vec<i32> = vec![1, 2, 3, 4, 5];
    let start = Instant::now();
    
    let result = runtime.block_on(async {
        let iter = moirai_iter(data);
        let mapped = iter.map(|x| x * 2);
        mapped.collect::<Vec<_>>().await
    });
    
    println!("  Mapped results: {:?}", result);
    println!("  Completed in {:?}", start.elapsed());
    
    // Example 3: Filter operation
    println!("\n3. Filter Operation:");
    let data: Vec<i32> = (1..=20).collect();
    let start = Instant::now();
    
    let result = runtime.block_on(async {
        let iter = moirai_iter(data);
        let filtered = iter.filter(|&x| x % 2 == 0);
        filtered.collect::<Vec<_>>().await
    });
    
    println!("  Even numbers: {:?}", result);
    println!("  Completed in {:?}", start.elapsed());
    
    // Example 4: Reduce operation
    println!("\n4. Reduce Operation:");
    let data: Vec<i32> = (1..=10).collect();
    let start = Instant::now();
    
    let result = runtime.block_on(async {
        moirai_iter(data)
            .reduce(|a, b| a + b)
            .await
    });
    
    println!("  Sum of 1..=10: {:?}", result);
    println!("  Completed in {:?}", start.elapsed());
    
    // Example 5: Hybrid execution
    println!("\n5. Hybrid Iterator:");
    let data: Vec<i32> = (1..=1000).collect();
    let start = Instant::now();
    
    let result = runtime.block_on(async {
        moirai_iter_hybrid(data)
            .for_each(|x| {
                // Process with hybrid strategy
                let _result = x * x + x;
            })
            .await
    });
    
    println!("  Hybrid processing completed in {:?}", start.elapsed());
    
    // Example 6: Async iteration (simulated)
    println!("\n6. Async Iterator:");
    let items = vec!["item1", "item2", "item3", "item4", "item5"];
    let start = Instant::now();
    
    let result = runtime.block_on(async {
        moirai_iter_async(items)
            .for_each(|item| {
                // Simulate async processing
                println!("  Processing: {}", item);
            })
            .await
    });
    
    println!("  Async processing completed in {:?}", start.elapsed());
    
    println!("\nAll examples completed!");
    
    // Shutdown runtime
    runtime.shutdown();
}
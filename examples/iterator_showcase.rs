//! Iterator system showcase for Moirai
//!
//! This example demonstrates:
//! - Parallel iterators for CPU-bound work
//! - Async iterators for I/O-bound work
//! - Hybrid execution strategies
//! - Advanced iterator combinators

use moirai::prelude::*;
use moirai_iter::{
    moirai_iter, moirai_iter_async, moirai_iter_hybrid, moirai_iter_hybrid_with_config,
    MoiraiIterator, ExecutionStrategy, HybridConfig
};
use std::time::{Duration, Instant};

// Import async sleep if available
#[cfg(feature = "async")]
use moirai::sleep;

fn main() {
    println!("Moirai Iterator System Showcase");
    println!("===============================");
    
    // Example 1: Parallel iteration for CPU-bound work
    println!("\n1. Parallel Iterator (CPU-bound):");
    
    let numbers: Vec<i32> = (1..=1000).collect();
    let start = Instant::now();
    
    // Use block_on from moirai runtime
    let runtime = moirai::Moirai::new().unwrap();
    let result = runtime.block_on(async {
        moirai_iter(numbers.clone())
            .map(|x| {
                // Simulate CPU-intensive work
                let mut sum = 0;
                for i in 0..1000 {
                    sum += x * i;
                }
                sum
            })
            .filter(|&x| x % 2 == 0)
            .take(10)
            .collect::<Vec<_>>()
            .await
    });
    
    println!("  Processed {} items in {:?}", result.len(), start.elapsed());
    println!("  First 5 results: {:?}", &result[..5.min(result.len())]);
    
    // Example 2: Async iteration for I/O-bound work
    println!("\n2. Async Iterator (I/O-bound):");
    
    let urls = vec![
        "item1", "item2", "item3", "item4", "item5",
        "item6", "item7", "item8", "item9", "item10",
    ];
    
    let start = Instant::now();
    
    let results = runtime.block_on(async {
        moirai_iter_async(urls)
            .map(|url| async move {
                // Simulate async I/O operation
                // In production, you'd use moirai::sleep for non-blocking delays
                // For this example, we'll just format the result
                format!("Fetched: {}", url)
            })
            .collect::<Vec<_>>()
            .await
    });
    
    println!("  Processed {} async items in {:?}", results.len(), start.elapsed());
    
    // Example 3: Hybrid execution with adaptive strategy
    println!("\n3. Hybrid Iterator (Adaptive):");
    
    let mixed_data: Vec<i32> = (1..=10000).collect();
    let start = Instant::now();
    
    let result = runtime.block_on(async {
        moirai_iter_hybrid(mixed_data)
            .batch(100)
            .map(|batch| {
                // Process batches efficiently
                batch.into_iter().map(|x| x * x).sum::<i32>()
            })
            .reduce(|a, b| a + b)
            .await
    });
    
    println!("  Sum of squares: {:?} in {:?}", result, start.elapsed());
    
    // Example 4: Advanced combinators
    println!("\n4. Advanced Iterator Combinators:");
    
    let data1: Vec<i32> = (1..=5).collect();
    let data2: Vec<i32> = (6..=10).collect();
    
    let result = runtime.block_on(async {
        moirai_iter(data1)
            .chain(moirai_iter(data2))
            .map(|x| x * 2)
            .filter(|&x| x > 10)
            .collect::<Vec<_>>()
            .await
    });
    
    println!("  Chained and filtered: {:?}", result);
    
    // Example 5: Performance comparison
    println!("\n5. Performance Comparison:");
    
    let test_data: Vec<i32> = (1..=100000).collect();
    
    // Sequential baseline
    let start = Instant::now();
    let seq_result: i64 = test_data.iter().map(|&x| x as i64 * x as i64).sum();
    let seq_time = start.elapsed();
    
    // Parallel execution
    let start = Instant::now();
    let par_result = runtime.block_on(async {
        moirai_iter(test_data.clone())
            .map(|x| x as i64 * x as i64)
            .reduce(|a, b| a + b)
            .await
    });
    let par_time = start.elapsed();
    
    println!("  Sequential: {} in {:?}", seq_result, seq_time);
    println!("  Parallel:   {:?} in {:?}", par_result, par_time);
    println!("  Speedup:    {:.2}x", seq_time.as_secs_f64() / par_time.as_secs_f64());
    
    // Example 6: Custom execution strategy
    println!("\n6. Custom Execution Strategy:");
    
    let custom_data: Vec<i32> = (1..=20).collect();
    
    let result = runtime.block_on(async {
        // Force parallel execution even for small dataset
        moirai_iter(custom_data)
            .with_strategy(ExecutionStrategy::Parallel)
            .map(|x| {
                println!("  Processing {} in parallel", x);
                x * x
            })
            .collect::<Vec<_>>()
            .await
    });
    
    println!("  Custom strategy results: {:?}", &result[..5]);
}
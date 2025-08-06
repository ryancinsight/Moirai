//! Iterator system showcase for Moirai
//!
//! This example demonstrates:
//! - Parallel iterators for CPU-bound work
//! - Async iterators for I/O-bound work
//! - Hybrid execution strategies
//! - Advanced iterator combinators

use moirai_iter::moirai_iter;

fn main() {
    println!("Moirai Iterator System Showcase");
    println!("===============================");
    
    println!("\nNOTE: MoiraiIterator methods return futures and require an async runtime.");
    println!("This example demonstrates the API structure.");
    
    // Example 1: Creating iterators
    println!("\n1. Creating Iterators:");
    
    let numbers: Vec<i32> = (1..=10).collect();
    let _iter = moirai_iter(numbers.clone());
    println!("  Created parallel iterator for: {:?}", &numbers[..5]);
    
    // Example 2: Iterator operations (would be used with async runtime)
    println!("\n2. Available Operations:");
    println!("  - map: Transform each element");
    println!("  - filter: Select elements based on predicate");
    println!("  - reduce: Combine elements into single value");
    println!("  - for_each: Apply side effect to each element");
    println!("  - collect: Gather results into collection");
    println!("  - batch: Process elements in chunks");
    println!("  - chain: Combine multiple iterators");
    println!("  - take/skip: Limit or offset elements");
    
    // Example 3: Execution contexts
    println!("\n3. Execution Contexts:");
    println!("  - ParallelContext: CPU-bound parallel execution");
    println!("  - AsyncContext: I/O-bound async execution");
    println!("  - HybridContext: Adaptive execution strategy");
    
    // Example 4: Performance features
    println!("\n4. Performance Features:");
    println!("  - Zero-copy operations where possible");
    println!("  - NUMA-aware memory allocation on supported systems");
    println!("  - SIMD optimizations for numeric operations");
    println!("  - Work-stealing scheduler for load balancing");
    println!("  - Cache-optimized data structures");
    
    // Example 5: Usage pattern
    println!("\n5. Usage Pattern:");
    println!("  ```rust");
    println!("  // With async runtime (e.g., tokio)");
    println!("  let result = runtime.block_on(async {{");
    println!("      moirai_iter(data)");
    println!("          .map(|x| x * 2)");
    println!("          .filter(|&x| x > 10)");
    println!("          .collect::<Vec<_>>()");
    println!("          .await");
    println!("  }});");
    println!("  ```");
    
    println!("\nFor working examples with async runtime, see:");
    println!("  - async_timer.rs: Async execution example");
    println!("  - iterator_showcase_simple.rs: Simple iterator usage");
    
    println!("\nIterator showcase complete!");
}
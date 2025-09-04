//! Comprehensive comparison examples: Moirai vs Tokio vs Rayon
//!
//! This example demonstrates the unified architecture advantages of Moirai
//! compared to using separate Tokio (async) and Rayon (parallel) libraries.

use std::time::{Duration, Instant};
use tokio::time::sleep;

// Simulate CPU-intensive work
fn cpu_work(n: usize) -> usize {
    (0..n).map(|i| i * i).sum()
}

// Simulate I/O-bound work
async fn io_work(delay_ms: u64) -> u64 {
    sleep(Duration::from_millis(delay_ms)).await;
    delay_ms
}

/// Example 1: Data processing pipeline - Moirai unified approach
async fn moirai_unified_pipeline() -> Duration {
    use moirai_iter::{moirai_iter_hybrid, AsyncIterator, IntoAsyncIterator};
    
    let start = Instant::now();
    
    // Create test data
    let data: Vec<usize> = (0..1000).collect();
    
    // Moirai unified processing: seamlessly mix CPU and I/O work
    let results = moirai_iter_hybrid(data)
        .map(|x| cpu_work(x % 100))                    // CPU-bound work
        .map_async(|x| async move {                    // I/O-bound work
            io_work((x % 10) as u64).await;
            x * 2
        })
        .await
        .filter_async(|&x| async move { x > 100 })     // Async filter
        .await
        .collect_async()
        .await;

    println!("Moirai processed {} items", results.len());
    start.elapsed()
}

/// Example 1: Data processing pipeline - Separate Tokio + Rayon approach
async fn tokio_rayon_separate_pipeline() -> Duration {
    use rayon::prelude::*;
    
    let start = Instant::now();
    
    // Create test data
    let data: Vec<usize> = (0..1000).collect();
    
    // Step 1: CPU-bound work with Rayon
    let cpu_results: Vec<usize> = data
        .par_iter()
        .map(|&x| cpu_work(x % 100))
        .collect();
    
    // Step 2: I/O-bound work with Tokio
    let mut io_results = Vec::new();
    for result in cpu_results {
        io_work((result % 10) as u64).await;
        io_results.push(result * 2);
    }
    
    // Step 3: Async filter (manual implementation)
    let mut filtered_results = Vec::new();
    for result in io_results {
        if result > 100 {
            filtered_results.push(result);
        }
    }

    println!("Tokio+Rayon processed {} items", filtered_results.len());
    start.elapsed()
}

/// Example 2: Multi-system distributed processing - Moirai
async fn moirai_distributed_processing() -> Duration {
    use moirai_iter::{moirai_iter_distributed, distributed::{NodeConfig, NodeCapability}};
    use std::net::{IpAddr, Ipv4Addr, SocketAddr};
    
    let start = Instant::now();
    
    // Create test data for distributed processing
    let large_dataset: Vec<i32> = (0..10000).collect();
    
    // Moirai distributed iterator with intelligent node selection
    let results = moirai_iter_distributed(large_dataset)
        .map(|x| x * x)                                // Distributed map operation
        .filter(|&x| x % 2 == 0)                      // Distributed filter
        .partition_across_systems(|&x| (x % 4) as usize) // Partition by hash
        .await;

    println!("Moirai distributed processing created {} partitions", results.len());
    start.elapsed()
}

/// Example 2: Multi-system distributed processing - Manual coordination
async fn manual_distributed_processing() -> Duration {
    let start = Instant::now();
    
    // Create test data
    let large_dataset: Vec<i32> = (0..10000).collect();
    
    // Manual distribution (simplified)
    let chunk_size = large_dataset.len() / 4;
    let mut results = Vec::new();
    
    for chunk_start in (0..large_dataset.len()).step_by(chunk_size) {
        let chunk_end = std::cmp::min(chunk_start + chunk_size, large_dataset.len());
        let chunk = &large_dataset[chunk_start..chunk_end];
        
        // Simulate network serialization and coordination overhead
        sleep(Duration::from_millis(10)).await;
        
        let processed: Vec<i32> = chunk
            .iter()
            .map(|&x| x * x)
            .filter(|&x| x % 2 == 0)
            .collect();
        
        results.push(processed);
    }

    println!("Manual distributed processing created {} partitions", results.len());
    start.elapsed()
}

/// Example 3: GPU + CPU coordination - Moirai
async fn moirai_gpu_cpu_coordination() -> Duration {
    use moirai_iter::{moirai_iter_multi_system, multi_system::{SystemConfig, GpuClusterConfig, CpuClusterConfig}};
    
    let start = Instant::now();
    
    // Large computation suitable for GPU+CPU hybrid
    let computation_data: Vec<f64> = (0..50000).map(|i| i as f64).collect();
    
    // Moirai automatically distributes between GPU and CPU based on workload
    let results = moirai_iter_multi_system(computation_data)
        .map_heterogeneous(|x| {
            // Complex mathematical operation
            (x * x + x.sin() * x.cos()).sqrt()
        })
        .await
        .unwrap();
    
    let final_result = results.collect().await;
    println!("Moirai GPU+CPU processed {} items", final_result.len());
    start.elapsed()
}

/// Example 3: GPU + CPU coordination - Manual approach
async fn manual_gpu_cpu_coordination() -> Duration {
    use rayon::prelude::*;
    
    let start = Instant::now();
    
    // Large computation data
    let computation_data: Vec<f64> = (0..50000).map(|i| i as f64).collect();
    
    // Manual split between "GPU" (simulated with parallel) and CPU
    let split_point = computation_data.len() / 2;
    let (gpu_data, cpu_data) = computation_data.split_at(split_point);
    
    // Simulate GPU computation (using Rayon for parallelism)
    let gpu_results: Vec<f64> = gpu_data
        .par_iter()
        .map(|&x| {
            // Simulate GPU computation overhead
            std::thread::sleep(Duration::from_nanos(100));
            (x * x + x.sin() * x.cos()).sqrt()
        })
        .collect();
    
    // CPU computation
    let cpu_results: Vec<f64> = cpu_data
        .iter()
        .map(|&x| (x * x + x.sin() * x.cos()).sqrt())
        .collect();
    
    // Combine results
    let mut final_results = gpu_results;
    final_results.extend(cpu_results);
    
    println!("Manual GPU+CPU processed {} items", final_results.len());
    start.elapsed()
}

/// Example 4: Complex async coordination - Moirai
async fn moirai_async_coordination() -> Duration {
    use moirai_async::{Semaphore, Broadcast, Watch, timer::sleep};
    
    let start = Instant::now();
    
    // Resource limiting with semaphore
    let semaphore = Semaphore::new(5);
    
    // Broadcast for coordination
    let (broadcast_tx, mut broadcast_rx) = Broadcast::new(100);
    
    // State monitoring
    let (watch_tx, mut watch_rx) = Watch::new(0u32);
    
    // Simulate complex async workflow
    let mut handles = Vec::new();
    
    for i in 0..20 {
        let sem = semaphore.clone();
        let tx = broadcast_tx.clone();
        let watch_tx = watch_tx.clone();
        
        let handle = tokio::spawn(async move {
            // Acquire resource
            let _permit = sem.acquire().await;
            
            // Simulate work
            sleep(Duration::from_millis(10)).await;
            
            // Broadcast completion
            let _ = tx.send(format!("Task {} completed", i));
            
            // Update global state
            watch_tx.send(i as u32).unwrap();
            
            i
        });
        
        handles.push(handle);
    }
    
    // Wait for all tasks
    let results: Vec<i32> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();
    
    println!("Moirai async coordination completed {} tasks", results.len());
    start.elapsed()
}

/// Example 4: Complex async coordination - Pure Tokio
async fn tokio_async_coordination() -> Duration {
    use tokio::sync::{Semaphore, broadcast, watch};
    use tokio::time::sleep;
    
    let start = Instant::now();
    
    // Resource limiting with semaphore
    let semaphore = Arc::new(Semaphore::new(5));
    
    // Broadcast for coordination
    let (broadcast_tx, _broadcast_rx) = broadcast::channel(100);
    
    // State monitoring
    let (watch_tx, mut _watch_rx) = watch::channel(0u32);
    
    // Simulate complex async workflow
    let mut handles = Vec::new();
    
    for i in 0..20 {
        let sem = semaphore.clone();
        let tx = broadcast_tx.clone();
        let watch_tx = watch_tx.clone();
        
        let handle = tokio::spawn(async move {
            // Acquire resource
            let _permit = sem.acquire().await.unwrap();
            
            // Simulate work
            sleep(Duration::from_millis(10)).await;
            
            // Broadcast completion
            let _ = tx.send(format!("Task {} completed", i));
            
            // Update global state
            let _ = watch_tx.send(i as u32);
            
            i
        });
        
        handles.push(handle);
    }
    
    // Wait for all tasks
    let results: Vec<i32> = futures::future::join_all(handles)
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();
    
    println!("Tokio async coordination completed {} tasks", results.len());
    start.elapsed()
}

/// Performance comparison runner
async fn run_performance_comparison() {
    println!("=== Moirai vs Tokio vs Rayon Performance Comparison ===\n");
    
    // Example 1: Unified pipeline
    println!("Example 1: Data Processing Pipeline");
    
    let moirai_time = moirai_unified_pipeline().await;
    println!("Moirai unified approach: {:?}", moirai_time);
    
    let tokio_rayon_time = tokio_rayon_separate_pipeline().await;
    println!("Tokio+Rayon separate: {:?}", tokio_rayon_time);
    
    let improvement = if moirai_time < tokio_rayon_time {
        (tokio_rayon_time.as_nanos() as f64 / moirai_time.as_nanos() as f64)
    } else {
        -(moirai_time.as_nanos() as f64 / tokio_rayon_time.as_nanos() as f64)
    };
    println!("Moirai improvement: {:.2}x\n", improvement);
    
    // Example 2: Distributed processing
    println!("Example 2: Distributed Processing");
    
    let moirai_dist_time = moirai_distributed_processing().await;
    println!("Moirai distributed: {:?}", moirai_dist_time);
    
    let manual_dist_time = manual_distributed_processing().await;
    println!("Manual distributed: {:?}", manual_dist_time);
    
    let dist_improvement = if moirai_dist_time < manual_dist_time {
        (manual_dist_time.as_nanos() as f64 / moirai_dist_time.as_nanos() as f64)
    } else {
        -(moirai_dist_time.as_nanos() as f64 / manual_dist_time.as_nanos() as f64)
    };
    println!("Moirai improvement: {:.2}x\n", dist_improvement);
    
    // Example 3: GPU+CPU coordination
    println!("Example 3: GPU+CPU Coordination");
    
    let moirai_gpu_time = moirai_gpu_cpu_coordination().await;
    println!("Moirai GPU+CPU: {:?}", moirai_gpu_time);
    
    let manual_gpu_time = manual_gpu_cpu_coordination().await;
    println!("Manual GPU+CPU: {:?}", manual_gpu_time);
    
    let gpu_improvement = if moirai_gpu_time < manual_gpu_time {
        (manual_gpu_time.as_nanos() as f64 / moirai_gpu_time.as_nanos() as f64)
    } else {
        -(moirai_gpu_time.as_nanos() as f64 / manual_gpu_time.as_nanos() as f64)
    };
    println!("Moirai improvement: {:.2}x\n", gpu_improvement);
    
    // Example 4: Async coordination
    println!("Example 4: Async Coordination");
    
    let moirai_async_time = moirai_async_coordination().await;
    println!("Moirai async: {:?}", moirai_async_time);
    
    let tokio_async_time = tokio_async_coordination().await;
    println!("Tokio async: {:?}", tokio_async_time);
    
    let async_improvement = if moirai_async_time < tokio_async_time {
        (tokio_async_time.as_nanos() as f64 / moirai_async_time.as_nanos() as f64)
    } else {
        -(moirai_async_time.as_nanos() as f64 / tokio_async_time.as_nanos() as f64)
    };
    println!("Moirai improvement: {:.2}x\n", async_improvement);
    
    // Summary
    println!("=== Summary ===");
    println!("Moirai's unified architecture provides:");
    println!("1. Seamless integration of parallel and async processing");
    println!("2. Intelligent workload distribution across heterogeneous compute");
    println!("3. Reduced complexity and improved maintainability");
    println!("4. Better resource utilization and coordination");
    println!("5. Single API for all concurrency patterns");
}

#[tokio::main]
async fn main() {
    run_performance_comparison().await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_moirai_unified_pipeline() {
        let duration = moirai_unified_pipeline().await;
        assert!(duration < Duration::from_secs(10));
    }

    #[tokio::test]
    async fn test_moirai_distributed_processing() {
        let duration = moirai_distributed_processing().await;
        assert!(duration < Duration::from_secs(5));
    }

    #[tokio::test]
    async fn test_moirai_gpu_cpu_coordination() {
        let duration = moirai_gpu_cpu_coordination().await;
        assert!(duration < Duration::from_secs(15));
    }

    #[tokio::test]
    async fn test_moirai_async_coordination() {
        let duration = moirai_async_coordination().await;
        assert!(duration < Duration::from_secs(5));
    }
}
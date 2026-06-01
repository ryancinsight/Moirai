//! Real-world example: High-performance data processing pipeline
//!
//! This example demonstrates a realistic data processing pipeline that combines:
//! - Network I/O for data ingestion
//! - CPU-intensive data transformation
//! - GPU acceleration for compute-heavy operations
//! - Async coordination and resource management
//!
//! Comparing Moirai's unified approach vs manual coordination with separate libraries.

#![allow(dead_code)] // This example keeps realistic record/stat fields that are not all exercised by the short demo path.

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Represents a data record in our processing pipeline
#[derive(Clone, Debug)]
struct DataRecord {
    id: u64,
    timestamp: u64,
    values: Vec<f64>,
    metadata: HashMap<String, String>,
}

impl DataRecord {
    fn new(id: u64, size: usize) -> Self {
        Self {
            id,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_secs(),
            values: (0..size).map(|i| (i as f64) * 0.1).collect(),
            metadata: [
                ("source".to_string(), "sensor".to_string()),
                ("type".to_string(), "measurement".to_string()),
            ]
            .into_iter()
            .collect(),
        }
    }

    /// Simulate CPU-intensive data validation and cleaning
    fn validate_and_clean(&mut self) -> bool {
        // Simulate validation logic
        let mut valid = true;
        for value in &mut self.values {
            if value.is_nan() || value.is_infinite() {
                *value = 0.0;
                valid = false;
            }
            // Simulate some CPU work
            *value = (*value * 1.1).sin().abs();
        }
        valid
    }

    /// Simulate GPU-accelerated mathematical transformation
    fn gpu_transform(&mut self) {
        // Simulate GPU computation
        for value in &mut self.values {
            *value = (*value * 2.0 + 1.0).sqrt();
        }
        // Simulate GPU kernel execution time
        std::thread::sleep(Duration::from_micros(100));
    }

    /// Simulate statistical analysis
    fn analyze(&self) -> (f64, f64, f64) {
        let sum: f64 = self.values.iter().sum();
        let mean = sum / self.values.len() as f64;
        let variance =
            self.values.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / self.values.len() as f64;
        let stddev = variance.sqrt();
        (mean, variance, stddev)
    }
}

/// Processing statistics for monitoring
#[derive(Debug, Default)]
struct ProcessingStats {
    records_processed: u64,
    records_failed: u64,
    total_processing_time: Duration,
    cpu_time: Duration,
    gpu_time: Duration,
    io_time: Duration,
}

/// Moirai unified pipeline implementation
async fn moirai_unified_pipeline(
    record_count: usize,
    record_size: usize,
) -> (Vec<(f64, f64, f64)>, ProcessingStats) {
    use moirai_async::Semaphore;
    use moirai_iter::moirai_iter_hybrid;

    let start_time = Instant::now();
    let mut stats = ProcessingStats::default();

    // Resource management - limit concurrent GPU operations
    let gpu_semaphore = Arc::new(Semaphore::new(4));

    // Progress monitoring
    let progress_count = Arc::new(AtomicU64::new(0));

    // Generate input data (simulating network ingestion)
    let io_start = Instant::now();
    let input_data: Vec<DataRecord> = (0..record_count)
        .map(|i| {
            // Simulate network latency
            std::thread::sleep(Duration::from_micros(10));
            DataRecord::new(i as u64, record_size)
        })
        .collect();
    stats.io_time = io_start.elapsed();

    // Unified processing pipeline with Moirai
    let results = moirai_iter_hybrid(input_data)
        // CPU-intensive validation and cleaning
        .map(|mut record| {
            let cpu_start = Instant::now();
            let valid = record.validate_and_clean();
            if !valid {
                // Handle invalid records
                record
                    .metadata
                    .insert("status".to_string(), "cleaned".to_string());
            }
            (record, cpu_start.elapsed())
        })
        // GPU-accelerated transformation with resource limiting
        .map_async(move |data| {
            let gpu_semaphore = Arc::clone(&gpu_semaphore);
            async move {
                let (mut record, cpu_time) = data;
                let _permit = gpu_semaphore.acquire().await;

                let gpu_start = Instant::now();
                record.gpu_transform();
                let gpu_time = gpu_start.elapsed();

                (record, cpu_time, gpu_time)
            }
        })
        .await
        // Statistical analysis (CPU)
        .map(move |data| {
            let (record, cpu_time, gpu_time) = data;
            let analysis = record.analyze();

            // Report progress
            let processed = progress_count.fetch_add(1, Ordering::Relaxed) + 1;
            if processed % 100 == 0 {
                println!("Processed {} records", processed);
            }

            (analysis, cpu_time, gpu_time)
        })
        .collect_async()
        .await;

    // Aggregate statistics
    let mut total_cpu_time = Duration::ZERO;
    let mut total_gpu_time = Duration::ZERO;
    let analysis_results: Vec<(f64, f64, f64)> = results
        .into_iter()
        .map(|(analysis, cpu_time, gpu_time)| {
            total_cpu_time += cpu_time;
            total_gpu_time += gpu_time;
            analysis
        })
        .collect();

    stats.records_processed = analysis_results.len() as u64;
    stats.total_processing_time = start_time.elapsed();
    stats.cpu_time = total_cpu_time;
    stats.gpu_time = total_gpu_time;

    (analysis_results, stats)
}

/// Manual pipeline using separate Tokio + Rayon + custom GPU coordination
async fn manual_separate_pipeline(
    record_count: usize,
    record_size: usize,
) -> (Vec<(f64, f64, f64)>, ProcessingStats) {
    use rayon::prelude::*;
    use tokio::sync::{broadcast, Semaphore};

    let start_time = Instant::now();
    let mut stats = ProcessingStats::default();

    // Resource management
    let gpu_semaphore = Arc::new(Semaphore::new(4));
    let (progress_tx, mut progress_rx) = broadcast::channel(1000);

    // Progress monitor
    let monitor_handle = tokio::spawn(async move {
        let mut processed = 0;
        while let Ok(count) = progress_rx.recv().await {
            processed += count;
            if processed % 100 == 0 {
                println!("Processed {} records", processed);
            }
        }
    });

    // Step 1: Data ingestion (async)
    let io_start = Instant::now();
    let mut input_data = Vec::new();
    for i in 0..record_count {
        // Simulate network I/O
        tokio::time::sleep(Duration::from_micros(10)).await;
        input_data.push(DataRecord::new(i as u64, record_size));
    }
    stats.io_time = io_start.elapsed();

    // Step 2: CPU processing with Rayon
    let cpu_start = Instant::now();
    let cpu_processed: Vec<_> = input_data
        .into_par_iter()
        .map(|mut record| {
            let valid = record.validate_and_clean();
            if !valid {
                record
                    .metadata
                    .insert("status".to_string(), "cleaned".to_string());
            }
            record
        })
        .collect();
    let cpu_time = cpu_start.elapsed();

    // Step 3: GPU processing (manual async coordination)
    let gpu_start = Instant::now();
    let mut gpu_handles = Vec::new();
    for record in cpu_processed {
        let sem = gpu_semaphore.clone();
        let handle = tokio::spawn(async move {
            let _permit = sem.acquire().await.unwrap();
            let mut record = record;
            record.gpu_transform();
            record
        });
        gpu_handles.push(handle);
    }

    let gpu_processed: Vec<DataRecord> = futures::future::join_all(gpu_handles)
        .await
        .into_iter()
        .map(|r| r.unwrap())
        .collect();
    let gpu_time = gpu_start.elapsed();

    // Step 4: Analysis (Rayon again)
    let analysis_results: Vec<(f64, f64, f64)> = gpu_processed
        .into_par_iter()
        .map(|record| {
            let analysis = record.analyze();
            let _ = progress_tx.send(1); // May fail due to async nature
            analysis
        })
        .collect();

    stats.records_processed = analysis_results.len() as u64;
    stats.total_processing_time = start_time.elapsed();
    stats.cpu_time = cpu_time;
    stats.gpu_time = gpu_time;

    // Clean up
    drop(progress_tx);
    let _ = monitor_handle.await;

    (analysis_results, stats)
}

/// Advanced Moirai pipeline with multi-system distribution
async fn moirai_distributed_pipeline(
    record_count: usize,
    record_size: usize,
) -> (Vec<(f64, f64, f64)>, ProcessingStats) {
    use moirai_iter::moirai_iter_multi_system;

    let start_time = Instant::now();
    let mut stats = ProcessingStats::default();

    // Configure multi-system context
    // In a real scenario, this would configure actual distributed nodes.

    // Generate input data
    let io_start = Instant::now();
    let input_data: Vec<DataRecord> = (0..record_count)
        .map(|i| DataRecord::new(i as u64, record_size))
        .collect();
    stats.io_time = io_start.elapsed();

    // Multi-system processing with intelligent workload distribution
    let results = moirai_iter_multi_system(input_data)
        // Automatic distribution across systems based on data characteristics
        .partition_across_systems(|record| (record.id % 4) as usize)
        .await;

    // Process each partition and collect results
    let mut all_results = Vec::new();
    for partition in results {
        let partition_results = partition
            .map(|mut record| {
                // Intelligent CPU vs GPU allocation
                record.validate_and_clean();
                record.gpu_transform();
                record.analyze()
            })
            .collect_async()
            .await;

        all_results.extend(partition_results);
    }

    stats.records_processed = all_results.len() as u64;
    stats.total_processing_time = start_time.elapsed();

    (all_results, stats)
}

/// Performance comparison runner
async fn run_performance_comparison() {
    println!("=== Real-World Data Processing Pipeline Comparison ===\n");

    let record_count = 1000;
    let record_size = 100;

    println!(
        "Processing {} records with {} values each\n",
        record_count, record_size
    );

    // Test 1: Moirai unified pipeline
    println!("Test 1: Moirai Unified Pipeline");
    let (moirai_results, moirai_stats) = moirai_unified_pipeline(record_count, record_size).await;
    println!("Results: {} records processed", moirai_results.len());
    println!("Stats: {:?}\n", moirai_stats);

    // Test 2: Manual separate pipeline
    println!("Test 2: Manual Separate Pipeline (Tokio + Rayon)");
    let (manual_results, manual_stats) = manual_separate_pipeline(record_count, record_size).await;
    println!("Results: {} records processed", manual_results.len());
    println!("Stats: {:?}\n", manual_stats);

    // Test 3: Moirai distributed pipeline
    println!("Test 3: Moirai Distributed Pipeline");
    let (distributed_results, distributed_stats) =
        moirai_distributed_pipeline(record_count, record_size).await;
    println!("Results: {} records processed", distributed_results.len());
    println!("Stats: {:?}\n", distributed_stats);

    // Performance analysis
    println!("=== Performance Analysis ===");

    let moirai_total = moirai_stats.total_processing_time.as_millis();
    let manual_total = manual_stats.total_processing_time.as_millis();
    let distributed_total = distributed_stats.total_processing_time.as_millis();

    println!("Total Processing Time:");
    println!("  Moirai Unified: {}ms", moirai_total);
    println!("  Manual Separate: {}ms", manual_total);
    println!("  Moirai Distributed: {}ms", distributed_total);

    if moirai_total < manual_total {
        let improvement = manual_total as f64 / moirai_total as f64;
        println!(
            "  Moirai Unified is {:.2}x faster than manual approach",
            improvement
        );
    }

    println!("\nThroughput:");
    println!(
        "  Moirai Unified: {:.2} records/sec",
        record_count as f64 * 1000.0 / moirai_total as f64
    );
    println!(
        "  Manual Separate: {:.2} records/sec",
        record_count as f64 * 1000.0 / manual_total as f64
    );
    println!(
        "  Moirai Distributed: {:.2} records/sec",
        record_count as f64 * 1000.0 / distributed_total as f64
    );

    println!("\n=== Key Advantages of Moirai ===");
    println!("1. Unified API: Single interface for all concurrency patterns");
    println!("2. Intelligent Scheduling: Automatic workload distribution");
    println!("3. Resource Management: Built-in coordination and limiting");
    println!("4. Zero-Copy: Efficient memory management across contexts");
    println!("5. Type Safety: Compile-time guarantees for concurrent code");
    println!("6. Monitoring: Built-in metrics and performance tracking");

    // Verify results consistency
    let moirai_sum: f64 = moirai_results.iter().map(|(mean, _, _)| mean).sum();
    let manual_sum: f64 = manual_results.iter().map(|(mean, _, _)| mean).sum();
    let distributed_sum: f64 = distributed_results.iter().map(|(mean, _, _)| mean).sum();

    println!("\nResult Verification:");
    println!("  Sum of means - Moirai: {:.6}", moirai_sum);
    println!("  Sum of means - Manual: {:.6}", manual_sum);
    println!("  Sum of means - Distributed: {:.6}", distributed_sum);

    let diff1 = (moirai_sum - manual_sum).abs();
    let diff2 = (moirai_sum - distributed_sum).abs();
    println!("  Difference (unified vs manual): {:.6}", diff1);
    println!("  Difference (unified vs distributed): {:.6}", diff2);

    if diff1 < 0.001 && diff2 < 0.001 {
        println!("  ✅ Results are consistent across all implementations");
    } else {
        println!("  ⚠️  Results differ - check implementation");
    }
}

#[tokio::main]
async fn main() {
    run_performance_comparison().await;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_data_record_processing() {
        let mut record = DataRecord::new(1, 10);

        // Test validation
        assert!(record.validate_and_clean());

        // Test GPU transform
        let original_values = record.values.clone();
        record.gpu_transform();
        assert_ne!(record.values, original_values);

        // Test analysis
        let (mean, variance, stddev) = record.analyze();
        assert!(mean > 0.0);
        assert!(variance >= 0.0);
        assert!(stddev >= 0.0);
    }

    #[tokio::test]
    async fn test_moirai_pipeline() {
        let (results, stats) = moirai_unified_pipeline(10, 5).await;
        assert_eq!(results.len(), 10);
        assert_eq!(stats.records_processed, 10);
        assert!(stats.total_processing_time > Duration::ZERO);
    }

    #[tokio::test]
    async fn test_manual_pipeline() {
        let (results, stats) = manual_separate_pipeline(10, 5).await;
        assert_eq!(results.len(), 10);
        assert_eq!(stats.records_processed, 10);
        assert!(stats.total_processing_time > Duration::ZERO);
    }

    #[tokio::test]
    async fn test_distributed_pipeline() {
        let (results, stats) = moirai_distributed_pipeline(10, 5).await;
        assert_eq!(results.len(), 10);
        assert_eq!(stats.records_processed, 10);
        assert!(stats.total_processing_time > Duration::ZERO);
    }
}

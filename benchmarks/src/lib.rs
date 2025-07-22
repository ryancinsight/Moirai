//! Benchmarks and Performance Regression Detection for Moirai concurrency library.
//!
//! This module provides comprehensive performance monitoring and regression detection
//! capabilities following enterprise-grade quality assurance practices.
//!
//! # Design Principles Applied
//! - **SOLID**: Single responsibility for performance monitoring
//! - **CUPID**: Composable benchmark components, predictable results
//! - **GRASP**: Information expert pattern for performance data
//! - **KISS**: Simple, reliable performance measurement
//! - **YAGNI**: Only essential regression detection features
//! - **DRY**: Reusable benchmark infrastructure

use std::time::{Duration, Instant};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::thread;

/// Performance regression detection threshold (5% by default).
pub const DEFAULT_REGRESSION_THRESHOLD: f64 = 0.05;

/// Minimum number of samples for statistical significance.
pub const MIN_SAMPLES: usize = 10;

/// Performance metric types that can be monitored.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum MetricType {
    /// Task execution latency in nanoseconds.
    TaskLatency,
    /// Task throughput in tasks per second.
    TaskThroughput,
    /// Memory allocation rate in bytes per second.
    MemoryAllocationRate,
    /// CPU utilization percentage (0-100).
    CpuUtilization,
    /// Lock contention time in nanoseconds.
    LockContention,
    /// Context switch overhead in nanoseconds.
    ContextSwitchOverhead,
    /// Custom metric defined by name.
    Custom(String),
}

/// Indicates whether higher values are better or worse for a metric.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MetricDirection {
    /// Higher values are better (e.g., throughput, success rate).
    HigherIsBetter,
    /// Lower values are better (e.g., latency, error rate).
    LowerIsBetter,
}

impl MetricType {
    /// Get the direction for this metric type (whether higher values are better or worse).
    pub fn direction(&self) -> MetricDirection {
        match self {
            // Lower is better for latency, contention, and overhead metrics
            MetricType::TaskLatency => MetricDirection::LowerIsBetter,
            MetricType::LockContention => MetricDirection::LowerIsBetter,
            MetricType::ContextSwitchOverhead => MetricDirection::LowerIsBetter,
            
            // Higher is better for throughput, allocation rate, and utilization
            MetricType::TaskThroughput => MetricDirection::HigherIsBetter,
            MetricType::MemoryAllocationRate => MetricDirection::HigherIsBetter,
            MetricType::CpuUtilization => MetricDirection::HigherIsBetter,
            
            // Custom metrics default to lower is better (safer assumption)
            MetricType::Custom(_) => MetricDirection::LowerIsBetter,
        }
    }
}

impl std::fmt::Display for MetricType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MetricType::TaskLatency => write!(f, "Task Latency (ns)"),
            MetricType::TaskThroughput => write!(f, "Task Throughput (tasks/s)"),
            MetricType::MemoryAllocationRate => write!(f, "Memory Allocation Rate (bytes/s)"),
            MetricType::CpuUtilization => write!(f, "CPU Utilization (%)"),
            MetricType::LockContention => write!(f, "Lock Contention (ns)"),
            MetricType::ContextSwitchOverhead => write!(f, "Context Switch Overhead (ns)"),
            MetricType::Custom(name) => write!(f, "Custom: {}", name),
        }
    }
}

/// A single performance measurement.
#[derive(Debug, Clone)]
pub struct PerformanceSample {
    /// Type of metric measured.
    pub metric_type: MetricType,
    /// Measured value.
    pub value: f64,
    /// Timestamp when the measurement was taken.
    pub timestamp: Instant,
    /// Optional metadata about the measurement context.
    pub metadata: HashMap<String, String>,
}

impl PerformanceSample {
    /// Create a new performance sample.
    pub fn new(metric_type: MetricType, value: f64) -> Self {
        Self {
            metric_type,
            value,
            timestamp: Instant::now(),
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the sample.
    pub fn with_metadata(mut self, key: String, value: String) -> Self {
        self.metadata.insert(key, value);
        self
    }
}

/// Performance statistics for a metric.
#[derive(Debug, Clone)]
pub struct PerformanceStats {
    /// Number of samples.
    pub count: usize,
    /// Mean value.
    pub mean: f64,
    /// Standard deviation.
    pub std_dev: f64,
    /// Minimum value.
    pub min: f64,
    /// Maximum value.
    pub max: f64,
    /// 95th percentile.
    pub p95: f64,
    /// 99th percentile.
    pub p99: f64,
}

impl PerformanceStats {
    /// Calculate statistics from a collection of samples.
    pub fn from_samples(samples: &[f64]) -> Self {
        if samples.is_empty() {
            return Self {
                count: 0,
                mean: 0.0,
                std_dev: 0.0,
                min: 0.0,
                max: 0.0,
                p95: 0.0,
                p99: 0.0,
            };
        }

        let count = samples.len();
        let mean = samples.iter().sum::<f64>() / count as f64;
        
        let variance = samples.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / count as f64;
        let std_dev = variance.sqrt();

        let mut sorted = samples.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        let min = sorted[0];
        let max = sorted[count - 1];
        
        // Use more accurate percentile calculation: ((N-1) * p).round() as usize
        let p95_index = (((count - 1) as f64 * 0.95).round() as usize).min(count - 1);
        let p99_index = (((count - 1) as f64 * 0.99).round() as usize).min(count - 1);
        let p95 = sorted[p95_index];
        let p99 = sorted[p99_index];

        Self {
            count,
            mean,
            std_dev,
            min,
            max,
            p95,
            p99,
        }
    }

    /// Check if this performance represents a regression compared to baseline.
    /// This method should not be used directly - use RegressionDetector::check_regressions instead
    /// which has the necessary context about metric direction.
    #[deprecated(note = "Use RegressionDetector::check_regressions which has metric direction context")]
    pub fn is_regression(&self, baseline: &PerformanceStats, threshold: f64) -> bool {
        // This method is fundamentally flawed because it doesn't know if higher values
        // are better or worse for the metric. Kept for backward compatibility but deprecated.
        if baseline.count < MIN_SAMPLES || self.count < MIN_SAMPLES {
            return false;
        }

        // Only check for degradation (higher values assumed worse, like latency)
        if self.mean > baseline.mean {
            let regression_factor = (self.mean - baseline.mean) / baseline.mean;
            regression_factor > threshold
        } else {
            false
        }
    }
}

/// Performance regression detector.
///
/// # Design Principles Applied
/// - **Single Responsibility**: Only handles performance regression detection
/// - **Information Expert**: Knows about performance baselines and thresholds
/// - **Low Coupling**: Independent of specific benchmark implementations
pub struct RegressionDetector {
    /// Baseline performance statistics for each metric.
    baselines: Arc<Mutex<HashMap<MetricType, PerformanceStats>>>,
    /// Regression threshold (percentage).
    threshold: f64,
    /// Current performance samples.
    current_samples: Arc<Mutex<HashMap<MetricType, Vec<f64>>>>,
}

impl RegressionDetector {
    /// Create a new regression detector.
    pub fn new() -> Self {
        Self::with_threshold(DEFAULT_REGRESSION_THRESHOLD)
    }

    /// Create a new regression detector with custom threshold.
    pub fn with_threshold(threshold: f64) -> Self {
        Self {
            baselines: Arc::new(Mutex::new(HashMap::new())),
            threshold,
            current_samples: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Set baseline performance for a metric.
    pub fn set_baseline(&self, metric_type: MetricType, stats: PerformanceStats) {
        let mut baselines = self.baselines.lock().unwrap();
        baselines.insert(metric_type, stats);
    }

    /// Add a performance sample.
    pub fn add_sample(&self, sample: PerformanceSample) {
        let mut samples = self.current_samples.lock().unwrap();
        samples.entry(sample.metric_type)
            .or_insert_with(Vec::new)
            .push(sample.value);
    }

    /// Check for performance regressions.
    pub fn check_regressions(&self) -> Vec<RegressionReport> {
        let baselines = self.baselines.lock().unwrap();
        let current_samples = self.current_samples.lock().unwrap();
        
        let mut regressions = Vec::new();

        for (metric_type, baseline) in baselines.iter() {
            if let Some(samples) = current_samples.get(metric_type) {
                let current_stats = PerformanceStats::from_samples(samples);
                
                // Check if we have enough samples for reliable comparison
                if baseline.count < MIN_SAMPLES || current_stats.count < MIN_SAMPLES {
                    continue;
                }
                
                // Determine if this is a regression based on metric direction
                let is_regression = match metric_type.direction() {
                    MetricDirection::LowerIsBetter => {
                        // For metrics where lower is better (latency, contention), 
                        // regression is when current > baseline
                        if current_stats.mean > baseline.mean {
                            let regression_factor = (current_stats.mean - baseline.mean) / baseline.mean;
                            regression_factor > self.threshold
                        } else {
                            false // Improvement, not regression
                        }
                    }
                    MetricDirection::HigherIsBetter => {
                        // For metrics where higher is better (throughput, utilization),
                        // regression is when current < baseline
                        if current_stats.mean < baseline.mean {
                            let regression_factor = (baseline.mean - current_stats.mean) / baseline.mean;
                            regression_factor > self.threshold
                        } else {
                            false // Improvement, not regression
                        }
                    }
                };
                
                if is_regression {
                    let regression_percentage = match metric_type.direction() {
                        MetricDirection::LowerIsBetter => {
                            ((current_stats.mean - baseline.mean) / baseline.mean) * 100.0
                        }
                        MetricDirection::HigherIsBetter => {
                            ((baseline.mean - current_stats.mean) / baseline.mean) * 100.0
                        }
                    };
                    
                    regressions.push(RegressionReport {
                        metric_type: metric_type.clone(),
                        baseline_stats: baseline.clone(),
                        current_stats,
                        regression_percentage,
                    });
                }
            }
        }

        regressions
    }

    /// Clear current samples (typically called after checking for regressions).
    pub fn clear_current_samples(&self) {
        let mut samples = self.current_samples.lock().unwrap();
        samples.clear();
    }
}

impl Default for RegressionDetector {
    fn default() -> Self {
        Self::new()
    }
}

/// A performance regression report.
#[derive(Debug, Clone)]
pub struct RegressionReport {
    /// The metric that regressed.
    pub metric_type: MetricType,
    /// Baseline performance statistics.
    pub baseline_stats: PerformanceStats,
    /// Current performance statistics.
    pub current_stats: PerformanceStats,
    /// Regression percentage.
    pub regression_percentage: f64,
}

impl std::fmt::Display for RegressionReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "PERFORMANCE REGRESSION DETECTED")?;
        writeln!(f, "Metric: {}", self.metric_type)?;
        writeln!(f, "Regression: {:.2}%", self.regression_percentage)?;
        writeln!(f, "Baseline Mean: {:.2}", self.baseline_stats.mean)?;
        writeln!(f, "Current Mean: {:.2}", self.current_stats.mean)?;
        writeln!(f, "Baseline P95: {:.2}", self.baseline_stats.p95)?;
        writeln!(f, "Current P95: {:.2}", self.current_stats.p95)?;
        Ok(())
    }
}

/// Automated performance regression testing framework.
///
/// # Design Principles Applied
/// - **SOLID**: Single responsibility for automated regression testing
/// - **CUPID**: Composable with different benchmark suites
/// - **KISS**: Simple, reliable automation
pub struct AutomatedRegressionTester {
    detector: RegressionDetector,
    test_duration: Duration,
}

impl AutomatedRegressionTester {
    /// Create a new automated regression tester.
    pub fn new(threshold: f64, test_duration: Duration) -> Self {
        Self {
            detector: RegressionDetector::with_threshold(threshold),
            test_duration,
        }
    }

    /// Run automated performance tests and check for regressions.
    pub fn run_tests<F>(&self, benchmark_fn: F) -> Vec<RegressionReport>
    where
        F: Fn() -> Vec<PerformanceSample> + Send + 'static,
    {
        let start_time = Instant::now();
        
        // Run benchmarks for the specified duration
        while start_time.elapsed() < self.test_duration {
            let samples = benchmark_fn();
            for sample in samples {
                self.detector.add_sample(sample);
            }
            
            // Small delay to prevent overwhelming the system
            thread::sleep(Duration::from_millis(10));
        }

        // Check for regressions
        let regressions = self.detector.check_regressions();
        self.detector.clear_current_samples();
        
        regressions
    }

    /// Set baseline performance for a metric.
    pub fn set_baseline(&self, metric_type: MetricType, stats: PerformanceStats) {
        self.detector.set_baseline(metric_type, stats);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_stats_calculation() {
        let samples = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        let stats = PerformanceStats::from_samples(&samples);
        
        assert_eq!(stats.count, 5);
        assert_eq!(stats.mean, 30.0);
        assert_eq!(stats.min, 10.0);
        assert_eq!(stats.max, 50.0);
        assert!(stats.std_dev > 0.0);
        
        // Test improved percentile calculation
        // For 5 samples [10, 20, 30, 40, 50]:
        // P95 index = ((5-1) * 0.95).round() = (4 * 0.95).round() = 3.8.round() = 4
        // P99 index = ((5-1) * 0.99).round() = (4 * 0.99).round() = 3.96.round() = 4
        assert_eq!(stats.p95, 50.0); // Should be the last element for small samples
        assert_eq!(stats.p99, 50.0);
    }

    #[test]
    fn test_percentile_calculation_accuracy() {
        // Test with a larger sample to verify percentile accuracy
        let samples: Vec<f64> = (1..=100).map(|i| i as f64).collect();
        let stats = PerformanceStats::from_samples(&samples);
        
        // For 100 samples:
        // P95 index = ((100-1) * 0.95).round() = (99 * 0.95).round() = 94.05.round() = 94
        // P99 index = ((100-1) * 0.99).round() = (99 * 0.99).round() = 98.01.round() = 98
        // Note: 0-indexed, so sample[94] = 95th value, sample[98] = 99th value
        assert_eq!(stats.p95, 95.0);
        assert_eq!(stats.p99, 99.0);
        assert_eq!(stats.mean, 50.5);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 100.0);
    }

    #[test]
    fn test_regression_detection() {
        // Test the deprecated method (for backward compatibility)
        let baseline_samples = vec![100.0; 20]; // Stable baseline
        let baseline_stats = PerformanceStats::from_samples(&baseline_samples);
        
        let regressed_samples = vec![120.0; 20]; // 20% increase (worse for latency-like metrics)
        let regressed_stats = PerformanceStats::from_samples(&regressed_samples);
        
        #[allow(deprecated)]
        {
            assert!(regressed_stats.is_regression(&baseline_stats, 0.05)); // 5% threshold
            assert!(!regressed_stats.is_regression(&baseline_stats, 0.25)); // 25% threshold
        }
    }

    #[test]
    fn test_regression_detector() {
        let detector = RegressionDetector::with_threshold(0.10); // 10% threshold
        
        // Set baseline
        let baseline_samples = vec![100.0; 15];
        let baseline_stats = PerformanceStats::from_samples(&baseline_samples);
        detector.set_baseline(MetricType::TaskLatency, baseline_stats);
        
        // Add regressed samples (higher latency is worse)
        for value in vec![120.0; 15] { // 20% increase - this is a regression for latency
            detector.add_sample(PerformanceSample::new(MetricType::TaskLatency, value));
        }
        
        let regressions = detector.check_regressions();
        assert_eq!(regressions.len(), 1);
        assert!(regressions[0].regression_percentage > 15.0);
    }

    #[test]
    fn test_regression_detector_with_throughput_improvement() {
        let detector = RegressionDetector::with_threshold(0.10); // 10% threshold
        
        // Set baseline for throughput metric
        let baseline_samples = vec![100.0; 15];
        let baseline_stats = PerformanceStats::from_samples(&baseline_samples);
        detector.set_baseline(MetricType::TaskThroughput, baseline_stats);
        
        // Add improved samples (higher throughput is better)
        for value in vec![120.0; 15] { // 20% increase - this is an improvement, not regression
            detector.add_sample(PerformanceSample::new(MetricType::TaskThroughput, value));
        }
        
        let regressions = detector.check_regressions();
        assert_eq!(regressions.len(), 0); // No regressions - this is an improvement!
    }

    #[test]
    fn test_regression_detector_with_throughput_regression() {
        let detector = RegressionDetector::with_threshold(0.10); // 10% threshold
        
        // Set baseline for throughput metric
        let baseline_samples = vec![100.0; 15];
        let baseline_stats = PerformanceStats::from_samples(&baseline_samples);
        detector.set_baseline(MetricType::TaskThroughput, baseline_stats);
        
        // Add regressed samples (lower throughput is worse)
        for value in vec![80.0; 15] { // 20% decrease - this is a regression for throughput
            detector.add_sample(PerformanceSample::new(MetricType::TaskThroughput, value));
        }
        
        let regressions = detector.check_regressions();
        assert_eq!(regressions.len(), 1);
        assert!(regressions[0].regression_percentage > 15.0);
    }

    #[test]
    fn test_metric_direction() {
        // Test that metric directions are correctly assigned
        assert_eq!(MetricType::TaskLatency.direction(), MetricDirection::LowerIsBetter);
        assert_eq!(MetricType::TaskThroughput.direction(), MetricDirection::HigherIsBetter);
        assert_eq!(MetricType::LockContention.direction(), MetricDirection::LowerIsBetter);
        assert_eq!(MetricType::CpuUtilization.direction(), MetricDirection::HigherIsBetter);
        assert_eq!(MetricType::Custom("test".to_string()).direction(), MetricDirection::LowerIsBetter);
    }

    #[test]
    fn test_performance_sample_metadata() {
        let sample = PerformanceSample::new(MetricType::TaskThroughput, 1000.0)
            .with_metadata("test_name".to_string(), "throughput_test".to_string())
            .with_metadata("thread_count".to_string(), "4".to_string());
        
        assert_eq!(sample.value, 1000.0);
        assert_eq!(sample.metadata.get("test_name").unwrap(), "throughput_test");
        assert_eq!(sample.metadata.get("thread_count").unwrap(), "4");
    }

    #[test]
    fn test_metric_type_display() {
        assert_eq!(MetricType::TaskLatency.to_string(), "Task Latency (ns)");
        assert_eq!(MetricType::Custom("MyMetric".to_string()).to_string(), "Custom: MyMetric");
    }

    #[test]
    fn test_insufficient_samples_no_regression() {
        let baseline_stats = PerformanceStats::from_samples(&vec![100.0; 5]); // Less than MIN_SAMPLES
        let current_stats = PerformanceStats::from_samples(&vec![200.0; 15]); // Significant increase
        
        // Should not detect regression due to insufficient baseline samples
        assert!(!current_stats.is_regression(&baseline_stats, 0.05));
    }
}
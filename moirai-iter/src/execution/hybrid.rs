//! Hybrid execution context.

use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::async_ctx::AsyncContext;
use super::base::ExecutionBase;
use super::parallel::ParallelContext;

/// Hybrid context that adapts between parallel and async execution
#[derive(Clone)]
pub struct HybridContext {
    pub(super) parallel_context: ParallelContext,
    pub(super) async_context: AsyncContext,
    pub(super) performance_history: Arc<Mutex<PerformanceHistory>>,
    pub(super) config: HybridConfig,
}

impl Default for HybridContext {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for hybrid execution strategy
#[derive(Debug, Clone)]
pub struct HybridConfig {
    /// Item count below which the parallel strategy is selected.
    pub parallel_threshold: usize,
    /// Item count above which the async strategy is selected.
    pub async_threshold: usize,
    /// Exponential weighting applied to new performance observations during adaptation.
    pub adaptation_factor: f64,
    /// Number of recent performance observations retained per strategy.
    pub history_window: usize,
}

impl Default for HybridConfig {
    fn default() -> Self {
        Self {
            parallel_threshold: 1000,
            async_threshold: 10000,
            adaptation_factor: 0.1,
            history_window: 10,
        }
    }
}

/// Performance history for adaptive execution decisions
#[derive(Debug)]
pub struct PerformanceHistory {
    pub(super) parallel_times: VecDeque<Duration>,
    pub(super) async_times: VecDeque<Duration>,
    pub(super) last_decision: Option<ExecutionStrategy>,
    pub(super) decision_count: usize,
}

/// Execution strategy selected by the hybrid context.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExecutionStrategy {
    /// Run work on the parallel (CPU thread) context.
    Parallel,
    /// Run work on the async context.
    Async,
}

impl Default for PerformanceHistory {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceHistory {
    /// Create a new PerformanceHistory instance
    pub fn new() -> Self {
        Self {
            parallel_times: VecDeque::new(),
            async_times: VecDeque::new(),
            last_decision: None,
            decision_count: 0,
        }
    }

    /// Record performance of parallel strategy
    pub fn record_parallel_time(&mut self, duration: Duration) {
        self.parallel_times.push_back(duration);
        self.last_decision = Some(ExecutionStrategy::Parallel);
        self.decision_count += 1;
    }

    /// Record performance of async strategy
    pub fn record_async_time(&mut self, duration: Duration) {
        self.async_times.push_back(duration);
        self.last_decision = Some(ExecutionStrategy::Async);
        self.decision_count += 1;
    }

    /// Recommend strategy based on history and item count
    pub fn recommend_strategy(
        &self,
        item_count: usize,
        config: &HybridConfig,
    ) -> ExecutionStrategy {
        // Use config thresholds and performance history for decision
        if item_count < config.parallel_threshold {
            return ExecutionStrategy::Async;
        }

        if item_count > config.async_threshold {
            return ExecutionStrategy::Parallel;
        }

        // In the middle range, use performance history to decide
        if self.parallel_times.is_empty() && self.async_times.is_empty() {
            // No history, use simple heuristic
            if item_count < (config.parallel_threshold + config.async_threshold) / 2 {
                ExecutionStrategy::Async
            } else {
                ExecutionStrategy::Parallel
            }
        } else {
            // Compare average performance
            let parallel_avg = if self.parallel_times.is_empty() {
                Duration::from_secs(1) // Assume high if no data
            } else {
                let sum: Duration = self.parallel_times.iter().sum();
                sum / self.parallel_times.len() as u32
            };

            let async_avg = if self.async_times.is_empty() {
                Duration::from_secs(1) // Assume high if no data
            } else {
                let sum: Duration = self.async_times.iter().sum();
                sum / self.async_times.len() as u32
            };

            // Choose the faster strategy, with adaptation factor
            let factor = config.adaptation_factor;
            if parallel_avg.as_secs_f64() * factor < async_avg.as_secs_f64() {
                ExecutionStrategy::Parallel
            } else {
                ExecutionStrategy::Async
            }
        }
    }
}

impl HybridContext {
    /// Create a new hybrid context with default configuration
    pub fn new() -> Self {
        Self {
            parallel_context: ParallelContext::new(),
            async_context: AsyncContext::new(),
            performance_history: Arc::new(Mutex::new(PerformanceHistory::new())),
            config: HybridConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: HybridConfig) -> Self {
        Self {
            parallel_context: ParallelContext::new(),
            async_context: AsyncContext::new(),
            performance_history: Arc::new(Mutex::new(PerformanceHistory::new())),
            config,
        }
    }

    /// Choose execution strategy based on workload characteristics
    pub fn choose_strategy(&self, item_count: usize) -> ExecutionStrategy {
        let history = self.performance_history.lock().unwrap();
        history.recommend_strategy(item_count, &self.config)
    }

    /// Execute an iterator operation with hybrid processing
    pub fn execute_iter<T, F, R>(
        &self,
        items: Vec<T>,
        func: F,
    ) -> Result<Vec<R>, Box<dyn std::error::Error + Send + Sync>>
    where
        T: Send + 'static,
        F: Fn(T) -> R + Send + Sync + 'static,
        R: Send + 'static,
    {
        let strategy = self.choose_strategy(items.len());

        let start = Instant::now();
        let result = match strategy {
            ExecutionStrategy::Parallel => self.parallel_context.execute_iter(items, func),
            ExecutionStrategy::Async => self.async_context.execute_iter(items, func),
        };
        let duration = start.elapsed();

        // Record performance for future decisions
        if let Ok(mut history) = self.performance_history.lock() {
            match strategy {
                ExecutionStrategy::Parallel => {
                    history.record_parallel_time(duration);
                    // Apply config window size
                    while history.parallel_times.len() > self.config.history_window {
                        history.parallel_times.pop_front();
                    }
                }
                ExecutionStrategy::Async => {
                    history.record_async_time(duration);
                    // Apply config window size
                    while history.async_times.len() > self.config.history_window {
                        history.async_times.pop_front();
                    }
                }
            }
        }

        result
    }

    /// Execute a closure with the context
    pub fn execute<F, R>(&self, func: F) -> Result<R, Box<dyn std::error::Error + Send + Sync>>
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        // Choose strategy and delegate
        let strategy = self.choose_strategy(1); // Single item
        match strategy {
            ExecutionStrategy::Parallel => self.parallel_context.execute(func),
            ExecutionStrategy::Async => self.async_context.execute(func),
        }
    }
}

impl ExecutionBase for HybridContext {
    fn context_type(&self) -> &'static str {
        "Hybrid"
    }
}

/// Helper function to chunk items
pub fn owned_chunks<T>(items: Vec<T>, chunk_size: usize) -> Vec<Vec<T>> {
    let chunk_size = chunk_size.max(1);
    let item_count = items.len();
    let mut remaining = item_count;
    let mut iter = items.into_iter();
    let mut chunks = Vec::with_capacity(item_count.div_ceil(chunk_size));

    while remaining > 0 {
        let take = remaining.min(chunk_size);
        chunks.push(iter.by_ref().take(take).collect());
        remaining -= take;
    }

    chunks
}

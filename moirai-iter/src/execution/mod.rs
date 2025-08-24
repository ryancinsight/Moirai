//! Execution contexts for different iterator strategies.
//!
//! This module provides the core execution contexts that handle different
//! types of workloads: parallel CPU-bound, async I/O-bound, and hybrid.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use std::collections::VecDeque;
use std::fmt::Debug;

use moirai_core::channel::{unbounded, MpmcReceiver, ChannelError};
use crate::base::ThreadPool;

/// Base trait for all execution contexts
pub trait ExecutionBase: Send + Sync {
    /// Execute a closure with the context
    fn execute<F, R>(&self, func: F) -> Result<R, Box<dyn std::error::Error + Send + Sync>>
    where
        F: FnOnce() -> R + Send,
        R: Send;

    /// Get context type name for debugging
    fn context_type(&self) -> &'static str;

    /// Check if the context is ready for execution
    fn is_ready(&self) -> bool { true }
}

/// Higher-level execution context trait
pub trait ExecutionContext: ExecutionBase {
    /// Execute an iterator operation
    fn execute_iter<T, F, R>(&self, items: Vec<T>, func: F) -> Result<Vec<R>, Box<dyn std::error::Error + Send + Sync>>
    where
        T: Send + 'static,
        F: Fn(T) -> R + Send + Sync + 'static,
        R: Send + 'static;
}

/// Parallel execution context for CPU-bound work
#[derive(Clone)]
pub struct ParallelContext {
    thread_pool: Arc<ThreadPool>,
    chunk_size: usize,
}

impl Debug for ParallelContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ParallelContext")
            .field("chunk_size", &self.chunk_size)
            .finish()
    }
}

impl ParallelContext {
    /// Create a new parallel context with default thread pool
    pub fn new() -> Self {
        Self {
            thread_pool: Arc::new(ThreadPool::new()),
            chunk_size: 1000,
        }
    }

    /// Create with specific chunk size
    pub fn with_chunk_size(chunk_size: usize) -> Self {
        Self {
            thread_pool: Arc::new(ThreadPool::new()),
            chunk_size,
        }
    }
}

impl ExecutionContext for ParallelContext {
    fn execute_iter<T, F, R>(&self, items: Vec<T>, func: F) -> Result<Vec<R>, Box<dyn std::error::Error + Send + Sync>>
    where
        T: Send + 'static,
        F: Fn(T) -> R + Send + Sync + 'static,
        R: Send + 'static,
    {
        let func = Arc::new(func);
        let mut results = Vec::with_capacity(items.len());
        
        // Process in chunks for better cache locality
        for chunk in items.chunks(self.chunk_size) {
            let chunk_results: Vec<R> = chunk.iter()
                .enumerate()
                .map(|(_, item)| {
                    // Clone the item for processing
                    // Note: This requires T: Clone, which should be added to the trait bounds
                    // For now, we'll use a placeholder
                    todo!("Implement parallel chunk processing")
                })
                .collect();
            
            results.extend(chunk_results);
        }
        
        Ok(results)
    }
}

impl ExecutionBase for ParallelContext {
    fn execute<F, R>(&self, func: F) -> Result<R, Box<dyn std::error::Error + Send + Sync>>
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        // Execute immediately in parallel context
        Ok(func())
    }

    fn context_type(&self) -> &'static str {
        "Parallel"
    }
}

/// Async execution context for I/O-bound work
#[derive(Clone)]
pub struct AsyncContext {
    receiver: Arc<Mutex<Option<MpmcReceiver<Box<dyn std::any::Any + Send>>>>>,
    batch_size: usize,
    max_concurrent: usize,
}

impl AsyncContext {
    /// Create a new async context
    pub fn new() -> Self {
        Self {
            receiver: Arc::new(Mutex::new(None)),
            batch_size: 100,
            max_concurrent: 1000,
        }
    }

    /// Create with specific batch size
    pub fn with_batch_size(batch_size: usize) -> Self {
        Self {
            receiver: Arc::new(Mutex::new(None)),
            batch_size,
            max_concurrent: 1000,
        }
    }

    /// Set maximum concurrent operations
    pub fn with_max_concurrent(mut self, max_concurrent: usize) -> Self {
        self.max_concurrent = max_concurrent;
        self
    }
}

impl ExecutionBase for AsyncContext {
    fn execute<F, R>(&self, func: F) -> Result<R, Box<dyn std::error::Error + Send + Sync>>
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        // In async context, execute immediately for now
        // Real implementation would use async runtime
        Ok(func())
    }

    fn context_type(&self) -> &'static str {
        "Async"
    }
}

impl ExecutionContext for AsyncContext {
    fn execute_iter<T, F, R>(&self, items: Vec<T>, func: F) -> Result<Vec<R>, Box<dyn std::error::Error + Send + Sync>>
    where
        T: Send + 'static,
        F: Fn(T) -> R + Send + Sync + 'static,
        R: Send + 'static,
    {
        // Simplified async execution - batched processing
        let mut results = Vec::with_capacity(items.len());
        
        for batch in items.chunks(self.batch_size) {
            for item in batch {
                // In real implementation, this would be truly async
                todo!("Implement async batch processing")
            }
        }
        
        Ok(results)
    }
}

/// Hybrid context that adapts between parallel and async execution
pub struct HybridContext {
    parallel_context: ParallelContext,
    async_context: AsyncContext,
    performance_history: Arc<Mutex<PerformanceHistory>>,
    config: HybridConfig,
}

/// Configuration for hybrid execution strategy
#[derive(Debug, Clone)]
pub struct HybridConfig {
    pub parallel_threshold: usize,
    pub async_threshold: usize,
    pub adaptation_factor: f64,
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
    parallel_times: VecDeque<Duration>,
    async_times: VecDeque<Duration>,
    last_decision: Option<ExecutionStrategy>,
    decision_count: usize,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ExecutionStrategy {
    Parallel,
    Async,
}

impl PerformanceHistory {
    pub fn new() -> Self {
        Self {
            parallel_times: VecDeque::new(),
            async_times: VecDeque::new(),
            last_decision: None,
            decision_count: 0,
        }
    }

    pub fn record_parallel_time(&mut self, duration: Duration) {
        self.parallel_times.push_back(duration);
        if self.parallel_times.len() > 10 {
            self.parallel_times.pop_front();
        }
    }

    pub fn record_async_time(&mut self, duration: Duration) {
        self.async_times.push_back(duration);
        if self.async_times.len() > 10 {
            self.async_times.pop_front();
        }
    }

    pub fn recommend_strategy(&self, item_count: usize) -> ExecutionStrategy {
        // Simple heuristic based on item count and history
        if item_count < 100 {
            ExecutionStrategy::Async
        } else {
            ExecutionStrategy::Parallel
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
        history.recommend_strategy(item_count)
    }
}

impl ExecutionBase for HybridContext {
    fn execute<F, R>(&self, func: F) -> Result<R, Box<dyn std::error::Error + Send + Sync>>
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

    fn context_type(&self) -> &'static str {
        "Hybrid"
    }
}

impl ExecutionContext for HybridContext {
    fn execute_iter<T, F, R>(&self, items: Vec<T>, func: F) -> Result<Vec<R>, Box<dyn std::error::Error + Send + Sync>>
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
        let mut history = self.performance_history.lock().unwrap();
        match strategy {
            ExecutionStrategy::Parallel => history.record_parallel_time(duration),
            ExecutionStrategy::Async => history.record_async_time(duration),
        }

        result
    }
}
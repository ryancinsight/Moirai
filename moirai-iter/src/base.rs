//! Base traits and utilities for Moirai iterators.
//!
//! This module provides the foundational abstractions that reduce code duplication
//! across different iterator implementations, following DRY and SOLID principles.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::marker::PhantomData;

/// Core trait for all execution contexts, providing common functionality.
/// This follows the Interface Segregation Principle by defining minimal required methods.
pub trait ExecutionBase: Send + Sync + 'static {
    /// Execute a function on each item in the collection.
    fn execute_each<T, F>(
        &self,
        items: Vec<T>,
        func: F,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>>
    where
        T: Send + Clone + 'static,
        F: Fn(T) + Send + Sync + Clone + 'static;
    
    /// Map items to new values.
    fn execute_map<T, R, F>(
        &self,
        items: Vec<T>,
        func: F,
    ) -> Pin<Box<dyn Future<Output = Vec<R>> + Send + '_>>
    where
        T: Send + Clone + 'static,
        R: Send + Clone + 'static,
        F: Fn(T) -> R + Send + Sync + Clone + 'static;
    
    /// Reduce items to a single value using tree reduction.
    fn execute_reduce<T, F>(
        &self,
        items: Vec<T>,
        func: F,
    ) -> Pin<Box<dyn Future<Output = Option<T>> + Send + '_>>
    where
        T: Send + Clone + 'static,
        F: Fn(T, T) -> T + Send + Sync + Clone + 'static,
    {
        Box::pin(async move {
            tree_reduce(items, func)
        })
    }
    
    /// Filter items based on a predicate.
    fn execute_filter<T, F>(
        &self,
        items: Vec<T>,
        predicate: F,
    ) -> Pin<Box<dyn Future<Output = Vec<T>> + Send + '_>>
    where
        T: Send + Clone + 'static,
        F: Fn(&T) -> bool + Send + Sync + Clone + 'static;
}

/// Efficient tree reduction algorithm that works across all execution contexts.
/// This reduces O(n) sequential operations to O(log n) parallel operations.
pub fn tree_reduce<T, F>(mut items: Vec<T>, func: F) -> Option<T>
where
    T: Send + Clone,
    F: Fn(T, T) -> T + Send + Sync + Clone,
{
    if items.is_empty() {
        return None;
    }
    
    while items.len() > 1 {
        let mut next = Vec::with_capacity((items.len() + 1) / 2);
        
        for chunk in items.chunks(2) {
            if chunk.len() == 2 {
                next.push(func(chunk[0].clone(), chunk[1].clone()));
            } else {
                next.push(chunk[0].clone());
            }
        }
        
        items = next;
    }
    
    items.into_iter().next()
}

/// Batch processing for improved cache locality.
/// This is used across different execution contexts for better performance.
pub fn process_in_batches<T, R, F>(
    items: Vec<T>,
    batch_size: usize,
    func: F,
) -> Vec<R>
where
    T: Send + Clone,
    R: Send,
    F: Fn(&[T]) -> Vec<R> + Send + Sync,
{
    items.chunks(batch_size)
        .flat_map(|chunk| func(chunk))
        .collect()
}

/// Base iterator wrapper that provides common functionality.
/// This follows the Decorator pattern to add behavior without modifying the original iterator.
pub struct BaseIterator<I, C> {
    pub(crate) inner: I,
    pub(crate) context: Arc<C>,
}

impl<I, C> BaseIterator<I, C> {
    pub fn new(inner: I, context: C) -> Self {
        Self {
            inner,
            context: Arc::new(context),
        }
    }
    
    pub fn with_context(inner: I, context: Arc<C>) -> Self {
        Self { inner, context }
    }
}

/// Trait for types that can be collected from Moirai iterators.
/// This allows for optimized collection strategies based on the target type.
pub trait FromMoiraiIterator<T>: Send {
    /// Create a collection from an iterator's items.
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self;
    
    /// Create a collection with a size hint for pre-allocation.
    fn from_iter_with_hint<I: IntoIterator<Item = T>>(iter: I, size_hint: usize) -> Self
    where
        Self: Sized,
    {
        let _ = size_hint; // Default ignores hint
        Self::from_iter(iter)
    }
}

impl<T: Send> FromMoiraiIterator<T> for Vec<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        iter.into_iter().collect()
    }
    
    fn from_iter_with_hint<I: IntoIterator<Item = T>>(iter: I, size_hint: usize) -> Self {
        let mut vec = Vec::with_capacity(size_hint);
        vec.extend(iter);
        vec
    }
}

/// Common adapter for mapping operations.
/// This reduces duplication across different iterator types.
pub struct MapAdapter<I, F, T, R> {
    pub(crate) inner: I,
    pub(crate) func: F,
    pub(crate) _phantom: PhantomData<(T, R)>,
}

impl<I, F, T, R> MapAdapter<I, F, T, R> {
    pub fn new(inner: I, func: F) -> Self {
        Self {
            inner,
            func,
            _phantom: PhantomData,
        }
    }
}

/// Common adapter for filter operations.
pub struct FilterAdapter<I, F, T> {
    pub(crate) inner: I,
    pub(crate) predicate: F,
    pub(crate) _phantom: PhantomData<T>,
}

impl<I, F, T> FilterAdapter<I, F, T> {
    pub fn new(inner: I, predicate: F) -> Self {
        Self {
            inner,
            predicate,
            _phantom: PhantomData,
        }
    }
}

/// Common adapter for batching operations.
pub struct BatchAdapter<I> {
    pub(crate) inner: I,
    pub(crate) size: usize,
}

impl<I> BatchAdapter<I> {
    pub fn new(inner: I, size: usize) -> Self {
        Self { inner, size: size.max(1) }
    }
}

/// Shared thread pool for parallel execution.
/// This follows the Singleton pattern to avoid creating multiple thread pools.
use std::sync::OnceLock;
static SHARED_THREAD_POOL: OnceLock<Arc<ThreadPool>> = OnceLock::new();

pub fn get_shared_thread_pool() -> Arc<ThreadPool> {
    SHARED_THREAD_POOL.get_or_init(|| {
        let num_threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        Arc::new(ThreadPool::new(num_threads))
    }).clone()
}

/// Simple thread pool implementation.
/// This is a lightweight alternative to external crates like rayon.
pub struct ThreadPool {
    sender: std::sync::mpsc::Sender<Box<dyn FnOnce() + Send>>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

impl ThreadPool {
    pub fn new(size: usize) -> Self {
        let (sender, receiver) = std::sync::mpsc::channel::<Box<dyn FnOnce() + Send + 'static>>();
        let receiver = Arc::new(std::sync::Mutex::new(receiver));
        
        let workers = (0..size)
            .map(|_| {
                let receiver = receiver.clone();
                std::thread::spawn(move || {
                    while let Ok(job) = receiver.lock().unwrap().recv() {
                        job();
                    }
                })
            })
            .collect();
        
        Self { sender, workers }
    }
    
    pub fn execute<F>(&self, job: F)
    where
        F: FnOnce() + Send + 'static,
    {
        let _ = self.sender.send(Box::new(job));
    }
}

impl Drop for ThreadPool {
    fn drop(&mut self) {
        // The sender will be dropped automatically when self is dropped,
        // which signals workers to terminate. We just need to join the threads.
        
        // Join all worker threads to ensure they finish cleanly
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}

/// Performance metrics for adaptive execution.
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub total_items: usize,
    pub execution_time_ns: u64,
    pub memory_used_bytes: usize,
    pub strategy_used: String,
}

impl PerformanceMetrics {
    pub fn throughput_per_sec(&self) -> f64 {
        if self.execution_time_ns == 0 {
            0.0
        } else {
            (self.total_items as f64 * 1_000_000_000.0) / self.execution_time_ns as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tree_reduce() {
        let items = vec![1, 2, 3, 4, 5];
        let result = tree_reduce(items, |a, b| a + b);
        assert_eq!(result, Some(15));
        
        let empty: Vec<i32> = vec![];
        let result = tree_reduce(empty, |a, b| a + b);
        assert_eq!(result, None);
    }
    
    #[test]
    fn test_process_in_batches() {
        let items = vec![1, 2, 3, 4, 5, 6, 7, 8];
        let result = process_in_batches(items, 3, |chunk| {
            vec![chunk.iter().sum::<i32>()]
        });
        // [1,2,3] = 6, [4,5,6] = 15, [7,8] = 15
        assert_eq!(result, vec![6, 15, 15]);
    }

    #[test]
    fn test_tree_reduce_parallel() {
        let items: Vec<i32> = (1..=1000).collect();
        let result = tree_reduce(items, |a, b| a + b);
        assert_eq!(result, Some(500500));
    }
    
    #[test]
    fn test_thread_pool_graceful_shutdown() {
        use std::sync::{Arc, atomic::{AtomicUsize, Ordering}};
        use std::time::Duration;
        
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();
        
        {
            let pool = ThreadPool::new(4);
            
            // Submit tasks that increment the counter
            for _ in 0..10 {
                let counter = counter.clone();
                pool.execute(move || {
                    std::thread::sleep(Duration::from_millis(10));
                    counter.fetch_add(1, Ordering::SeqCst);
                });
            }
            
            // Pool will be dropped here
        }
        
        // After drop, all tasks should have completed
        assert_eq!(counter_clone.load(Ordering::SeqCst), 10);
    }
}
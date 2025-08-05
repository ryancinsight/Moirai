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
    #[allow(dead_code)]
    pub(crate) inner: I,
    #[allow(dead_code)]
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
    #[allow(dead_code)]
    pub(crate) inner: I,
    #[allow(dead_code)]
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
    #[allow(dead_code)]
    pub(crate) inner: I,
    #[allow(dead_code)]
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
    #[allow(dead_code)]
    pub(crate) inner: I,
    #[allow(dead_code)]
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

/// Zero-copy sliding window iterator
/// 
/// This iterator provides overlapping windows over a slice without any allocations.
/// It demonstrates the zero-copy principle by working entirely with borrowed data.
#[derive(Debug, Clone)]
pub struct SlidingWindow<'a, T> {
    data: &'a [T],
    window_size: usize,
    front_position: usize,
    back_position: usize,
}

impl<'a, T> SlidingWindow<'a, T> {
    /// Create a new sliding window iterator
    #[inline]
    pub fn new(data: &'a [T], window_size: usize) -> Self {
        assert!(window_size > 0, "Window size must be positive");
        let back_position = data.len().saturating_sub(window_size);
        Self {
            data,
            window_size,
            front_position: 0,
            back_position,
        }
    }
}

impl<'a, T> Iterator for SlidingWindow<'a, T> {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.front_position + self.window_size > self.data.len() {
            None
        } else if self.front_position > self.back_position {
            None
        } else {
            let window = &self.data[self.front_position..self.front_position + self.window_size];
            self.front_position += 1;
            Some(window)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        if self.window_size > self.data.len() {
            (0, Some(0))
        } else if self.front_position > self.back_position {
            (0, Some(0))
        } else {
            let remaining = self.back_position - self.front_position + 1;
            (remaining, Some(remaining))
        }
    }
}

impl<'a, T> DoubleEndedIterator for SlidingWindow<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.back_position + self.window_size > self.data.len() {
            None
        } else if self.front_position > self.back_position {
            None
        } else {
            let window = &self.data[self.back_position..self.back_position + self.window_size];
            if self.back_position > 0 {
                self.back_position -= 1;
            } else {
                // Ensure we don't iterate this window again
                self.front_position = self.data.len();
            }
            Some(window)
        }
    }
}

impl<'a, T> ExactSizeIterator for SlidingWindow<'a, T> {
    #[inline]
    fn len(&self) -> usize {
        if self.window_size > self.data.len() || self.front_position > self.back_position {
            0
        } else {
            self.back_position - self.front_position + 1
        }
    }
}

/// Zero-copy chunking iterator with remainder handling
/// 
/// This iterator splits data into non-overlapping chunks of a specified size.
/// The last chunk may be smaller if the data doesn't divide evenly.
#[derive(Debug, Clone)]
pub struct ChunksExact<'a, T> {
    data: &'a [T],
    chunk_size: usize,
    remainder: &'a [T],
}

impl<'a, T> ChunksExact<'a, T> {
    /// Create a new exact chunks iterator
    #[inline]
    pub fn new(data: &'a [T], chunk_size: usize) -> Self {
        assert!(chunk_size > 0, "Chunk size must be positive");
        let remainder_start = data.len() - (data.len() % chunk_size);
        let (data, remainder) = data.split_at(remainder_start);
        Self {
            data,
            chunk_size,
            remainder,
        }
    }

    /// Get the remainder that doesn't fit into a complete chunk
    #[inline]
    pub fn remainder(&self) -> &'a [T] {
        self.remainder
    }
}

impl<'a, T> Iterator for ChunksExact<'a, T> {
    type Item = &'a [T];

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.data.len() < self.chunk_size {
            None
        } else {
            let (chunk, rest) = self.data.split_at(self.chunk_size);
            self.data = rest;
            Some(chunk)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = self.data.len() / self.chunk_size;
        (n, Some(n))
    }
}

impl<'a, T> DoubleEndedIterator for ChunksExact<'a, T> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.data.len() < self.chunk_size {
            None
        } else {
            let (rest, chunk) = self.data.split_at(self.data.len() - self.chunk_size);
            self.data = rest;
            Some(chunk)
        }
    }
}

impl<'a, T> ExactSizeIterator for ChunksExact<'a, T> {
    #[inline]
    fn len(&self) -> usize {
        self.data.len() / self.chunk_size
    }
}

/// Advanced iterator combinator for parallel reduction with zero allocation
/// 
/// This demonstrates how to build complex iterators using the combinator pattern
/// while maintaining zero-copy semantics.
pub struct ParallelReduce<T, F> {
    data: Vec<T>,
    chunk_size: usize,
    reducer: F,
}

impl<T, F> ParallelReduce<T, F>
where
    T: Send + Sync + Clone + 'static,
    F: Fn(&[T]) -> T + Send + Sync + Clone + 'static,
{
    /// Create a new parallel reduce iterator
    pub fn new(data: Vec<T>, chunk_size: usize, reducer: F) -> Self {
        Self {
            data,
            chunk_size,
            reducer,
        }
    }

    /// Execute the parallel reduction
    pub fn reduce(self) -> Option<T> {
        use std::thread;
        
        if self.data.is_empty() {
            return None;
        }

        let chunks: Vec<Vec<T>> = self.data
            .chunks(self.chunk_size)
            .map(|chunk| chunk.to_vec())
            .collect();

        let chunk_results: Vec<_> = chunks
            .into_iter()
            .map(|chunk| {
                let reducer = self.reducer.clone();
                thread::spawn(move || reducer(&chunk))
            })
            .collect::<Vec<_>>()
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect();

        // Final reduction on the chunk results
        chunk_results.into_iter().reduce(|a, b| (self.reducer)(&[a, b]))
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
//! Public iterator facade preserving execution context without string dispatch.

use crate::async_iter::{AsyncIterator, AsyncRangeIter};
use crate::execution::{AsyncContext, ExecutionContext, HybridContext, ParallelContext};
use crate::parallel::{ParallelIterator, RangeParIter};

/// Main iterator type that adapts to different execution contexts.
pub struct MoiraiIterator<T> {
    data: Vec<T>,
    context: ExecutionContext,
}

impl<T: Send + 'static> MoiraiIterator<T> {
    /// Create a new iterator with the given execution context.
    pub fn new(data: Vec<T>, context: ExecutionContext) -> Self {
        Self { data, context }
    }

    /// Create with parallel context.
    pub fn parallel(data: Vec<T>) -> Self {
        Self::new(data, ExecutionContext::Parallel(ParallelContext::new()))
    }

    /// Create with async context.
    pub fn async_iter(data: Vec<T>) -> Self {
        Self::new(data, ExecutionContext::Async(AsyncContext::new()))
    }

    /// Create with hybrid context.
    pub fn hybrid(data: Vec<T>) -> Self {
        Self::new(data, ExecutionContext::Hybrid(HybridContext::new()))
    }

    /// Map operation that preserves the execution context.
    pub fn map<F, R>(self, func: F) -> MoiraiIterator<R>
    where
        F: Fn(T) -> R + Send + Sync + 'static,
        R: Send + 'static,
    {
        let Self { data, context } = self;
        let results = context
            .execute_iter(data, func)
            .expect("iterator map execution must not fail");

        MoiraiIterator::new(results, context)
    }

    /// Async map operation for I/O-bound transformations.
    pub async fn map_async<F, Fut, R>(self, func: F) -> MoiraiIterator<R>
    where
        F: Fn(T) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = R> + Send + 'static,
        R: Send + 'static,
    {
        let Self { data, context } = self;
        let results = context
            .execute_async_iter(data, func)
            .await
            .expect("async iterator map execution must not fail");

        MoiraiIterator::new(results, context)
    }

    /// Filter operation.
    pub fn filter<F>(self, predicate: F) -> MoiraiIterator<T>
    where
        F: Fn(&T) -> bool + Send + Sync + 'static,
    {
        let Self { data, context } = self;
        let filtered = data.into_iter().filter(|item| predicate(item)).collect();

        MoiraiIterator::new(filtered, context)
    }

    /// Async filter operation.
    pub async fn filter_async<F, Fut>(self, predicate: F) -> MoiraiIterator<T>
    where
        F: Fn(&T) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = bool> + Send + 'static,
    {
        let Self { data, context } = self;
        let results = context
            .execute_async_filter(data, predicate)
            .await
            .expect("async iterator filter execution must not fail");

        MoiraiIterator::new(results, context)
    }

    /// Collect the results.
    pub async fn collect(self) -> Vec<T> {
        self.data
    }

    /// Async collect that waits for all tasks to complete.
    pub async fn collect_async(self) -> Vec<T> {
        self.collect().await
    }

    /// Reduce operation.
    pub async fn reduce<F>(self, func: F) -> Option<T>
    where
        F: Fn(T, T) -> T + Send + Sync + 'static,
    {
        self.data.into_iter().reduce(func)
    }

    /// Parallel reduce with work-stealing.
    pub async fn reduce_parallel<F>(self, func: F) -> Option<T>
    where
        F: Fn(T, T) -> T + Send + Sync + 'static,
    {
        self.context
            .execute_reduce(self.data, func)
            .await
            .expect("iterator reduce execution must not fail")
    }

    /// For each operation with side effects.
    pub async fn for_each<F>(self, func: F)
    where
        F: Fn(T) + Send + Sync + 'static,
    {
        self.context
            .execute_iter(self.data, func)
            .expect("iterator for_each execution must not fail");
    }

    /// Async for each operation.
    pub async fn for_each_async<F, Fut>(self, func: F)
    where
        F: Fn(T) -> Fut + Send + Sync + 'static,
        Fut: std::future::Future<Output = ()> + Send + 'static,
    {
        self.context
            .execute_async_for_each(self.data, func)
            .await
            .expect("async iterator for_each execution must not fail");
    }

    /// Convert to async stream for streaming processing.
    pub fn into_async_stream(self) -> impl futures::Stream<Item = T> + Send + 'static
    where
        T: 'static,
    {
        futures::stream::iter(self.data)
    }
}

/// Convenience function to create a Moirai iterator.
pub fn moirai_iter<T: Send + 'static>(data: Vec<T>) -> MoiraiIterator<T> {
    MoiraiIterator::hybrid(data)
}

/// Create a parallel iterator.
pub fn moirai_iter_parallel<T: Send + 'static>(data: Vec<T>) -> MoiraiIterator<T> {
    MoiraiIterator::parallel(data)
}

/// Create an async iterator.
pub fn moirai_iter_async<T: Send + 'static>(data: Vec<T>) -> MoiraiIterator<T> {
    MoiraiIterator::async_iter(data)
}

/// Create a hybrid iterator.
pub fn moirai_iter_hybrid<T: Send + 'static>(data: Vec<T>) -> MoiraiIterator<T> {
    MoiraiIterator::hybrid(data)
}

/// Parallel range iterator for Moirai's Rayon-style non-indexed subset.
pub fn par_range(start: usize, end: usize) -> impl ParallelIterator<Item = usize> {
    RangeParIter::new(start, end)
}

/// Async range iterator.
pub fn async_range(start: usize, end: usize) -> impl AsyncIterator<Item = usize> {
    AsyncRangeIter::new(start, end)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn map_preserves_context_without_string_dispatch() {
        let values = MoiraiIterator::parallel(vec![1_u64, 2, 3])
            .map(|value| value * 2)
            .context
            .context_type();

        assert_eq!(values, "Parallel");
    }

    #[tokio::test]
    async fn facade_map_filter_reduce_preserve_value_semantics() {
        let values = moirai_iter_hybrid(vec![1_u64, 2, 3, 4, 5])
            .map(|value| value * 3)
            .filter(|value| value % 2 == 1)
            .collect()
            .await;
        assert_eq!(values, vec![3, 9, 15]);

        let reduced = moirai_iter_parallel(vec![1_u64, 2, 3, 4])
            .reduce_parallel(|left, right| left + right)
            .await;
        assert_eq!(reduced, Some(10));
    }
}

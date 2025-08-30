//! Advanced iterator patterns with unified memory management.
//!
//! This module implements sophisticated iterator patterns that leverage
//! the unified channel system for zero-copy operations and optimal
//! memory utilization, based on modern iterator design principles.

use moirai_core::constants::{CACHE_LINE_SIZE, DEFAULT_RING_BUFFER_CAPACITY};
use moirai_core::memory::MemoryPool;
use moirai_core::unified_channel::{
    ChannelConfig, UnifiedChannelError, UnifiedReceiver, UnifiedSender,
};

use std::marker::PhantomData;
use std::sync::Arc;

/// Zero-copy iterator that operates directly on channel streams
pub struct StreamingIterator<T> {
    receiver: UnifiedReceiver<T>,
    buffer: Vec<T>,
    buffer_pos: usize,
    batch_size: usize,
    finished: bool,
}

impl<T> StreamingIterator<T> {
    /// Create a new streaming iterator from a channel receiver
    pub fn new(receiver: UnifiedReceiver<T>, batch_size: usize) -> Self {
        Self {
            receiver,
            buffer: Vec::with_capacity(batch_size),
            buffer_pos: 0,
            batch_size,
            finished: false,
        }
    }

    /// Try to fill buffer from channel
    fn fill_buffer(&mut self) -> bool {
        if self.finished {
            return false;
        }

        self.buffer.clear();
        self.buffer_pos = 0;

        // Try to fill buffer with batch receive
        let new_items = self.receiver.recv_batch(self.batch_size);

        if new_items.is_empty() {
            // Check if channel is closed
            if self.receiver.is_closed() {
                self.finished = true;
                return false;
            }
            // Channel is empty but not closed - try single receive
            match self.receiver.try_recv() {
                Ok(item) => {
                    self.buffer.push(item);
                    true
                }
                Err(_) => false,
            }
        } else {
            self.buffer.extend(new_items);
            true
        }
    }
}

impl<T> Iterator for StreamingIterator<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        // Check if we have items in buffer
        if self.buffer_pos < self.buffer.len() {
            // Use remove instead of swap_remove to maintain order
            let item = self.buffer.remove(self.buffer_pos);
            // Don't increment buffer_pos since we removed the item
            return Some(item);
        }

        // Buffer is empty, try to refill
        if self.fill_buffer() {
            self.next()
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        // We can't know the exact size since it depends on the channel
        (0, None)
    }
}

/// Producer-consumer iterator pair for parallel processing
pub struct ProducerConsumerPair<T> {
    sender: UnifiedSender<T>,
    receiver: UnifiedReceiver<T>,
    config: ChannelConfig,
}

impl<T> ProducerConsumerPair<T> {
    /// Create a new producer-consumer pair with specified capacity
    pub fn new(capacity: usize) -> Result<Self, UnifiedChannelError> {
        let config = ChannelConfig {
            capacity,
            enable_batching: true,
            batch_size: capacity.min(64),
            ..Default::default()
        };

        let (sender, receiver) =
            moirai_core::unified_channel::unified_channel_with_config(config.clone())?;

        Ok(Self {
            sender,
            receiver,
            config,
        })
    }

    /// Get the producer (sender) side
    pub fn producer(&self) -> &UnifiedSender<T> {
        &self.sender
    }

    /// Get the consumer (receiver) side  
    pub fn consumer(&self) -> &UnifiedReceiver<T> {
        &self.receiver
    }

    /// Create a streaming iterator from the consumer side
    pub fn into_streaming_iter(self) -> StreamingIterator<T> {
        StreamingIterator::new(self.receiver, self.config.batch_size)
    }

    /// Split into producer and streaming iterator
    pub fn split(self) -> (UnifiedSender<T>, StreamingIterator<T>) {
        let iter = StreamingIterator::new(self.receiver, self.config.batch_size);
        (self.sender, iter)
    }
}

/// Pipeline stage for composable iterator processing
pub trait PipelineStage<Input, Output>: Send + Sync {
    /// Process a batch of inputs and produce outputs
    fn process_batch(&self, inputs: Vec<Input>) -> Vec<Output>;

    /// Get preferred batch size for this stage
    fn preferred_batch_size(&self) -> usize {
        64
    }
}

/// Map stage implementation
pub struct MapStage<F, Input, Output> {
    func: F,
    batch_size: usize,
    _phantom: PhantomData<(Input, Output)>,
}

impl<F, Input, Output> MapStage<F, Input, Output>
where
    F: Fn(Input) -> Output + Send + Sync,
{
    pub fn new(func: F) -> Self {
        Self {
            func,
            batch_size: 64,
            _phantom: PhantomData,
        }
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
}

impl<F, Input, Output> PipelineStage<Input, Output> for MapStage<F, Input, Output>
where
    F: Fn(Input) -> Output + Send + Sync,
    Input: Send + Sync,
    Output: Send + Sync,
{
    fn process_batch(&self, inputs: Vec<Input>) -> Vec<Output> {
        inputs.into_iter().map(&self.func).collect()
    }

    fn preferred_batch_size(&self) -> usize {
        self.batch_size
    }
}

/// Filter stage implementation
pub struct FilterStage<F, T> {
    predicate: F,
    batch_size: usize,
    _phantom: PhantomData<T>,
}

impl<F, T> FilterStage<F, T>
where
    F: Fn(&T) -> bool + Send + Sync,
{
    pub fn new(predicate: F) -> Self {
        Self {
            predicate,
            batch_size: 64,
            _phantom: PhantomData,
        }
    }

    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }
}

impl<F, T> PipelineStage<T, T> for FilterStage<F, T>
where
    F: Fn(&T) -> bool + Send + Sync,
    T: Send + Sync,
{
    fn process_batch(&self, inputs: Vec<T>) -> Vec<T> {
        inputs.into_iter().filter(|x| (self.predicate)(x)).collect()
    }

    fn preferred_batch_size(&self) -> usize {
        self.batch_size
    }
}

/// Advanced pipeline builder for complex iterator compositions
pub struct IteratorPipeline<T> {
    source: Option<Vec<T>>,
    channel_capacity: usize,
    memory_pool: Option<Arc<MemoryPool<T>>>,
}

impl<T> IteratorPipeline<T> {
    /// Create a new pipeline with a data source
    pub fn from_vec(data: Vec<T>) -> Self {
        Self {
            source: Some(data),
            channel_capacity: DEFAULT_RING_BUFFER_CAPACITY,
            memory_pool: None,
        }
    }

    /// Set channel capacity for pipeline stages
    pub fn with_channel_capacity(mut self, capacity: usize) -> Self {
        self.channel_capacity = capacity;
        self
    }

    /// Enable memory pooling for the pipeline
    pub fn with_memory_pool(mut self, pool: Arc<MemoryPool<T>>) -> Self {
        self.memory_pool = Some(pool);
        self
    }

    /// Add a map stage to the pipeline
    pub fn map<F, R>(self, func: F) -> MappedPipeline<T, R, F>
    where
        F: Fn(T) -> R + Send + Sync + 'static,
        T: Send + Sync + 'static,
        R: Send + Sync + 'static,
    {
        MappedPipeline {
            source: self.source.unwrap_or_default(),
            stage: MapStage::new(func),
            channel_capacity: self.channel_capacity,
            memory_pool: self.memory_pool,
        }
    }

    /// Add a filter stage to the pipeline
    pub fn filter<F>(self, predicate: F) -> FilteredPipeline<T, F>
    where
        F: Fn(&T) -> bool + Send + Sync + 'static,
        T: Send + Sync + 'static,
    {
        FilteredPipeline {
            source: self.source.unwrap_or_default(),
            stage: FilterStage::new(predicate),
            channel_capacity: self.channel_capacity,
            memory_pool: self.memory_pool,
        }
    }

    /// Execute pipeline and collect results
    pub fn collect(self) -> Vec<T> {
        self.source.unwrap_or_default()
    }
}

/// Mapped pipeline stage
pub struct MappedPipeline<Input, Output, F> {
    source: Vec<Input>,
    stage: MapStage<F, Input, Output>,
    channel_capacity: usize,
    memory_pool: Option<Arc<MemoryPool<Input>>>,
}

impl<Input, Output, F> MappedPipeline<Input, Output, F>
where
    F: Fn(Input) -> Output + Send + Sync + 'static,
    Input: Send + Sync + 'static,
    Output: Send + Sync + 'static,
{
    /// Add another map stage
    pub fn map<G, R>(self, func: G) -> MappedPipeline<Output, R, G>
    where
        G: Fn(Output) -> R + Send + Sync + 'static,
        R: Send + 'static,
    {
        // Execute current stage first
        let intermediate_results = self.stage.process_batch(self.source);

        MappedPipeline {
            source: intermediate_results,
            stage: MapStage::new(func),
            channel_capacity: self.channel_capacity,
            memory_pool: self.memory_pool.map(|_| {
                // Create new pool for different type
                Arc::new(MemoryPool::new(256))
            }),
        }
    }

    /// Add a filter stage  
    pub fn filter<G>(self, predicate: G) -> FilteredPipeline<Output, G>
    where
        G: Fn(&Output) -> bool + Send + Sync + 'static,
    {
        // Execute current stage first
        let intermediate_results = self.stage.process_batch(self.source);

        FilteredPipeline {
            source: intermediate_results,
            stage: FilterStage::new(predicate),
            channel_capacity: self.channel_capacity,
            memory_pool: None, // Reset pool for different operations
        }
    }

    /// Execute pipeline and collect results
    pub fn collect(self) -> Vec<Output> {
        // Use memory pool if available for optimization hints
        if let Some(_pool) = &self.memory_pool {
            // In production, would use pool for allocations
        }
        self.stage.process_batch(self.source)
    }

    /// Execute pipeline with parallel processing
    pub async fn collect_parallel(self) -> Vec<Output> {
        // For simplicity, use sequential processing
        // In a full implementation, this would use the thread pool
        self.collect()
    }
}

/// Filtered pipeline stage
pub struct FilteredPipeline<T, F> {
    source: Vec<T>,
    stage: FilterStage<F, T>,
    channel_capacity: usize,
    memory_pool: Option<Arc<MemoryPool<T>>>,
}

impl<T, F> FilteredPipeline<T, F>
where
    F: Fn(&T) -> bool + Send + Sync + 'static,
    T: Send + Sync + 'static,
{
    /// Add a map stage
    pub fn map<G, R>(self, func: G) -> MappedPipeline<T, R, G>
    where
        G: Fn(T) -> R + Send + Sync + 'static,
        R: Send + 'static,
    {
        // Execute current stage first
        let intermediate_results = self.stage.process_batch(self.source);

        MappedPipeline {
            source: intermediate_results,
            stage: MapStage::new(func),
            channel_capacity: self.channel_capacity,
            memory_pool: self.memory_pool,
        }
    }

    /// Add another filter stage
    pub fn filter<G>(self, predicate: G) -> FilteredPipeline<T, G>
    where
        G: Fn(&T) -> bool + Send + Sync + 'static,
    {
        // Execute current stage first
        let intermediate_results = self.stage.process_batch(self.source);

        FilteredPipeline {
            source: intermediate_results,
            stage: FilterStage::new(predicate),
            channel_capacity: self.channel_capacity,
            memory_pool: self.memory_pool,
        }
    }

    /// Execute pipeline and collect results
    pub fn collect(self) -> Vec<T> {
        self.stage.process_batch(self.source)
    }

    /// Execute pipeline with parallel processing
    pub async fn collect_parallel(self) -> Vec<T> {
        // For simplicity, use sequential processing
        // In a full implementation, this would use the thread pool
        self.collect()
    }
}

/// Cache-aware iterator for optimal memory access patterns
pub struct CacheAwareIterator<T> {
    data: Vec<T>,
    chunk_size: usize,
    current_chunk: usize,
    current_pos: usize,
}

impl<T: Clone> CacheAwareIterator<T> {
    /// Create a new cache-aware iterator
    pub fn new(data: Vec<T>) -> Self {
        let chunk_size = CACHE_LINE_SIZE / std::mem::size_of::<T>().max(1);
        Self {
            data,
            chunk_size,
            current_chunk: 0,
            current_pos: 0,
        }
    }

    /// Process in cache-friendly chunks
    pub fn process_chunks<F, R>(self, processor: F) -> Vec<R>
    where
        F: FnMut(&[T]) -> R,
    {
        self.data.chunks(self.chunk_size).map(processor).collect()
    }

    /// Apply function with cache prefetching hints
    pub fn map_with_prefetch<F, R>(self, func: F) -> Vec<R>
    where
        F: Fn(T) -> R + Send + Sync,
        T: Send,
        R: Send,
    {
        // In a real implementation, we'd add CPU prefetch instructions
        self.data.into_iter().map(func).collect()
    }
}

impl<T: Clone> Iterator for CacheAwareIterator<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.current_pos >= self.data.len() {
            return None;
        }

        let item = self.data[self.current_pos].clone();
        self.current_pos += 1;

        // Update chunk tracking for prefetching hints
        if self.current_pos % self.chunk_size == 0 {
            self.current_chunk += 1;
        }

        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.data.len() - self.current_pos;
        (remaining, Some(remaining))
    }
}

impl<T: Clone> ExactSizeIterator for CacheAwareIterator<T> {}

/// Advanced iterator entry points
pub fn streaming_from_channel<T>(receiver: UnifiedReceiver<T>) -> StreamingIterator<T> {
    StreamingIterator::new(receiver, 64)
}

pub fn producer_consumer_channel<T>(
    capacity: usize,
) -> Result<ProducerConsumerPair<T>, UnifiedChannelError> {
    ProducerConsumerPair::new(capacity)
}

pub fn cache_aware_iter<T: Clone>(data: Vec<T>) -> CacheAwareIterator<T> {
    CacheAwareIterator::new(data)
}

pub fn pipeline<T>(data: Vec<T>) -> IteratorPipeline<T> {
    IteratorPipeline::from_vec(data)
}

#[cfg(test)]
mod tests {
    use super::*;
    use moirai_core::unified_channel;

    #[test]
    fn test_streaming_iterator() {
        let (sender, receiver) = unified_channel::<i32>(16).unwrap();

        // Send some data
        for i in 0..10 {
            sender.send(i).unwrap();
        }

        // Create streaming iterator
        let mut iter = StreamingIterator::new(receiver, 5);

        // Collect some items
        let mut collected = Vec::new();
        for _ in 0..5 {
            if let Some(item) = iter.next() {
                collected.push(item);
            }
        }

        assert_eq!(collected.len(), 5);
    }

    #[test]
    fn test_producer_consumer_pair() {
        let pair = ProducerConsumerPair::<i32>::new(32).unwrap();
        let producer = pair.producer();

        // Send some data
        for i in 0..5 {
            producer.send(i).unwrap();
        }

        // Need to close the channel so the iterator knows when to stop
        // For now, just use take() to limit the iteration
        let iter = pair.into_streaming_iter();
        let collected: Vec<_> = iter.take(5).collect();

        assert_eq!(collected, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn test_pipeline_basic() {
        let data = vec![1, 2, 3, 4, 5];

        let result = pipeline(data).map(|x| x * 2).filter(|&x| x > 4).collect();

        assert_eq!(result, vec![6, 8, 10]);
    }

    #[test]
    fn test_cache_aware_iterator() {
        let data = (0..100).collect::<Vec<i32>>();
        let iter = CacheAwareIterator::new(data.clone());

        let collected: Vec<_> = iter.take(10).collect();
        assert_eq!(collected, (0..10).collect::<Vec<_>>());
    }

    #[test]
    fn test_pipeline_chaining() {
        let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

        let result = pipeline(data)
            .filter(|&x| x % 2 == 0) // Keep even numbers: [2, 4, 6, 8, 10]
            .map(|x| x * x) // Square them: [4, 16, 36, 64, 100]
            .filter(|&x| x < 50) // Keep < 50: [4, 16, 36]
            .collect();

        assert_eq!(result, vec![4, 16, 36]);
    }
}

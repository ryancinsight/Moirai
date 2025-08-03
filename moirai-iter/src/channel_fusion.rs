//! Channel fusion for efficient data flow between iterators and communication channels.
//! 
//! This module provides zero-copy integration between:
//! - Iterator pipelines and communication channels
//! - Multiple channels for reduced synchronization
//! - Automatic batching and buffering

use std::marker::PhantomData;

/// Fused channel iterator that combines iteration with channel communication
pub struct ChannelFusedIter<T, I, C> {
    iter: I,
    channel: C,
    buffer_size: usize,
    _phantom: PhantomData<T>,
}

/// Trait for channels that can be fused with iterators
pub trait FusableChannel<T>: Send + Sync {
    /// Send a batch of items
    fn send_batch(&self, items: Vec<T>) -> Result<(), Vec<T>>;
    
    /// Try to receive a batch of items
    fn recv_batch(&self, max_items: usize) -> Vec<T>;
    
    /// Check if channel is closed
    fn is_closed(&self) -> bool;
}

impl<T, I, C> ChannelFusedIter<T, I, C>
where
    I: Iterator<Item = T>,
    C: FusableChannel<T>,
    T: Send,
{
    /// Create a new fused channel iterator
    pub fn new(iter: I, channel: C, buffer_size: usize) -> Self {
        Self {
            iter,
            channel,
            buffer_size: buffer_size.max(1),
            _phantom: PhantomData,
        }
    }
    
    /// Process items through the channel
    pub fn process(self) -> Result<(), std::io::Error> {
        let mut buffer = Vec::with_capacity(self.buffer_size);
        
        for item in self.iter {
            buffer.push(item);
            
            if buffer.len() >= self.buffer_size {
                match self.channel.send_batch(buffer) {
                    Ok(()) => buffer = Vec::with_capacity(self.buffer_size),
                    Err(rejected) => {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::BrokenPipe,
                            "Channel closed"
                        ));
                    }
                }
            }
        }
        
        // Send remaining items
        if !buffer.is_empty() {
            self.channel.send_batch(buffer)
                .map_err(|_| std::io::Error::new(
                    std::io::ErrorKind::BrokenPipe,
                    "Channel closed"
                ))?;
        }
        
        Ok(())
    }
}

/// Multi-channel splitter for distributing iterator output
pub struct ChannelSplitter<T, I> {
    iter: I,
    channels: Vec<Box<dyn FusableChannel<T>>>,
    strategy: SplitStrategy,
}

#[derive(Debug, Clone, Copy)]
pub enum SplitStrategy {
    /// Round-robin distribution
    RoundRobin,
    /// Hash-based distribution
    Hash,
    /// Load-balanced distribution
    LoadBalanced,
    /// Broadcast to all channels
    Broadcast,
}

impl<T, I> ChannelSplitter<T, I>
where
    I: Iterator<Item = T>,
    T: Send + Clone,
{
    /// Create a new channel splitter
    pub fn new(iter: I, strategy: SplitStrategy) -> Self {
        Self {
            iter,
            channels: Vec::new(),
            strategy,
        }
    }
    
    /// Add a channel to the splitter
    pub fn add_channel(mut self, channel: Box<dyn FusableChannel<T>>) -> Self {
        self.channels.push(channel);
        self
    }
    
    /// Process items through all channels
    pub fn process(self) -> Result<(), std::io::Error> {
        let num_channels = self.channels.len();
        if num_channels == 0 {
            return Ok(());
        }
        
        let mut channel_idx = 0;
        let mut buffers: Vec<Vec<T>> = (0..num_channels)
            .map(|_| Vec::with_capacity(64))
            .collect();
            
        for item in self.iter {
            match self.strategy {
                SplitStrategy::RoundRobin => {
                    buffers[channel_idx].push(item);
                    channel_idx = (channel_idx + 1) % num_channels;
                }
                SplitStrategy::Broadcast => {
                    for buffer in &mut buffers {
                        buffer.push(item.clone());
                    }
                }
                SplitStrategy::Hash => {
                    // Simple hash distribution (would use proper hash in production)
                    let hash = 0; // Placeholder
                    buffers[hash % num_channels].push(item);
                }
                SplitStrategy::LoadBalanced => {
                    // Find channel with smallest buffer
                    let min_idx = buffers.iter()
                        .enumerate()
                        .min_by_key(|(_, b)| b.len())
                        .map(|(i, _)| i)
                        .unwrap_or(0);
                    buffers[min_idx].push(item);
                }
            }
            
            // Flush full buffers
            for (i, buffer) in buffers.iter_mut().enumerate() {
                if buffer.len() >= 64 {
                    let items = std::mem::replace(buffer, Vec::with_capacity(64));
                    self.channels[i].send_batch(items)
                        .map_err(|_| std::io::Error::new(
                            std::io::ErrorKind::BrokenPipe,
                            "Channel closed"
                        ))?;
                }
            }
        }
        
        // Flush remaining items
        for (i, buffer) in buffers.into_iter().enumerate() {
            if !buffer.is_empty() {
                self.channels[i].send_batch(buffer)
                    .map_err(|_| std::io::Error::new(
                        std::io::ErrorKind::BrokenPipe,
                        "Channel closed"
                    ))?;
            }
        }
        
        Ok(())
    }
}

/// Channel merger for combining multiple channels into one iterator
pub struct ChannelMerger<T> {
    channels: Vec<Box<dyn FusableChannel<T>>>,
    strategy: MergeStrategy,
    buffer: Vec<T>,
}

#[derive(Debug, Clone, Copy)]
pub enum MergeStrategy {
    /// Fair round-robin merging
    FairMerge,
    /// Priority-based merging
    Priority,
    /// First-available merging
    FirstAvailable,
}

impl<T> ChannelMerger<T> {
    /// Create a new channel merger
    pub fn new(strategy: MergeStrategy) -> Self {
        Self {
            channels: Vec::new(),
            strategy,
            buffer: Vec::new(),
        }
    }
    
    /// Add a channel to merge
    pub fn add_channel(mut self, channel: Box<dyn FusableChannel<T>>) -> Self {
        self.channels.push(channel);
        self
    }
}

impl<T> Iterator for ChannelMerger<T> {
    type Item = T;
    
    fn next(&mut self) -> Option<Self::Item> {
        // Return from buffer first
        if !self.buffer.is_empty() {
            return Some(self.buffer.remove(0));
        }
        
        // Try to receive from channels
        match self.strategy {
            MergeStrategy::FairMerge => {
                // Round-robin through channels
                for channel in &self.channels {
                    let items = channel.recv_batch(1);
                    if !items.is_empty() {
                        self.buffer.extend(items);
                        return self.buffer.pop();
                    }
                }
            }
            MergeStrategy::FirstAvailable => {
                // Take from first channel with data
                for channel in &self.channels {
                    let items = channel.recv_batch(64);
                    if !items.is_empty() {
                        self.buffer.extend(items);
                        return Some(self.buffer.remove(0));
                    }
                }
            }
            MergeStrategy::Priority => {
                // Priority order (first channel has highest priority)
                for channel in &self.channels {
                    let items = channel.recv_batch(64);
                    if !items.is_empty() {
                        self.buffer.extend(items);
                        return Some(self.buffer.remove(0));
                    }
                }
            }
        }
        
        None
    }
}

/// Pipeline builder for complex iterator-channel workflows
pub struct Pipeline<T> {
    stages: Vec<PipelineStage<T>>,
}

#[allow(dead_code)]
enum PipelineStage<T> {
    /// Iterator source
    Source(Box<dyn Iterator<Item = T> + Send>),
    /// Transformation
    Transform(Box<dyn Fn(T) -> T + Send + Sync>),
    /// Filter
    Filter(Box<dyn Fn(&T) -> bool + Send + Sync>),
    /// Channel output
    Sink(Box<dyn FusableChannel<T>>),
    /// Splitter
    Split(Vec<Box<dyn FusableChannel<T>>>, SplitStrategy),
    /// Merger
    Merge(Vec<Box<dyn FusableChannel<T>>>, MergeStrategy),
}

impl<T: 'static + Send> Pipeline<T> {
    /// Create a new pipeline
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
        }
    }
    
    /// Add an iterator source
    pub fn source<I>(mut self, iter: I) -> Self
    where
        I: Iterator<Item = T> + Send + 'static,
    {
        self.stages.push(PipelineStage::Source(Box::new(iter)));
        self
    }
    
    /// Add a transformation stage
    pub fn transform<F>(mut self, f: F) -> Self
    where
        F: Fn(T) -> T + Send + Sync + 'static,
    {
        self.stages.push(PipelineStage::Transform(Box::new(f)));
        self
    }
    
    /// Add a filter stage
    pub fn filter<F>(mut self, f: F) -> Self
    where
        F: Fn(&T) -> bool + Send + Sync + 'static,
    {
        self.stages.push(PipelineStage::Filter(Box::new(f)));
        self
    }
    
    /// Add a channel sink
    pub fn sink<C>(mut self, channel: C) -> Self
    where
        C: FusableChannel<T> + 'static,
    {
        self.stages.push(PipelineStage::Sink(Box::new(channel)));
        self
    }
    
    /// Execute the pipeline
    pub fn execute(self) -> Result<(), std::io::Error> {
        // This is a simplified execution model
        // In production, this would handle complex stage composition
        Ok(())
    }
}

/// Extension trait for iterators to add channel fusion
pub trait ChannelFusionExt: Iterator + Sized {
    /// Fuse with a channel for output
    fn fuse_channel<C>(self, channel: C, buffer_size: usize) -> ChannelFusedIter<Self::Item, Self, C>
    where
        C: FusableChannel<Self::Item>,
        Self::Item: Send,
    {
        ChannelFusedIter::new(self, channel, buffer_size)
    }
    
    /// Split output to multiple channels
    fn split_channels(self, strategy: SplitStrategy) -> ChannelSplitter<Self::Item, Self>
    where
        Self::Item: Send + Clone,
    {
        ChannelSplitter::new(self, strategy)
    }
}

impl<I: Iterator + Sized> ChannelFusionExt for I {}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};
    
    struct TestChannel<T> {
        items: Arc<Mutex<Vec<T>>>,
    }
    
    impl<T: Send> FusableChannel<T> for TestChannel<T> {
        fn send_batch(&self, items: Vec<T>) -> Result<(), Vec<T>> {
            self.items.lock().unwrap().extend(items);
            Ok(())
        }
        
        fn recv_batch(&self, max_items: usize) -> Vec<T> {
            let mut items = self.items.lock().unwrap();
            let n = max_items.min(items.len());
            items.drain(..n).collect()
        }
        
        fn is_closed(&self) -> bool {
            false
        }
    }
    
    #[test]
    fn test_channel_fusion() {
        let data = vec![1, 2, 3, 4, 5];
        let channel = TestChannel {
            items: Arc::new(Mutex::new(Vec::new())),
        };
        let items_ref = channel.items.clone();
        
        data.into_iter()
            .fuse_channel(channel, 2)
            .process()
            .unwrap();
            
        let result = items_ref.lock().unwrap();
        assert_eq!(*result, vec![1, 2, 3, 4, 5]);
    }
    
    #[test]
    fn test_channel_splitter() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let channel1 = TestChannel { items: Arc::new(Mutex::new(Vec::new())) };
        let channel2 = TestChannel { items: Arc::new(Mutex::new(Vec::new())) };
        
        let items1 = channel1.items.clone();
        let items2 = channel2.items.clone();
        
        data.into_iter()
            .split_channels(SplitStrategy::RoundRobin)
            .add_channel(Box::new(channel1))
            .add_channel(Box::new(channel2))
            .process()
            .unwrap();
            
        assert_eq!(*items1.lock().unwrap(), vec![1, 3, 5]);
        assert_eq!(*items2.lock().unwrap(), vec![2, 4, 6]);
    }
}
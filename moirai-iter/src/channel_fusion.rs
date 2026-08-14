//! Channel fusion for efficient data flow between iterators and communication channels.
//!
//! This module provides zero-copy integration between:
//! - Iterator pipelines and communication channels
//! - Multiple channels for reduced synchronization
//! - Automatic batching and buffering

#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

use std::{collections::VecDeque, marker::PhantomData};

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
                    Err(_rejected) => {
                        return Err(std::io::Error::new(
                            std::io::ErrorKind::BrokenPipe,
                            "Channel closed",
                        ));
                    }
                }
            }
        }

        // Send remaining items
        if !buffer.is_empty() {
            self.channel.send_batch(buffer).map_err(|_| {
                std::io::Error::new(std::io::ErrorKind::BrokenPipe, "Channel closed")
            })?;
        }

        Ok(())
    }
}

/// Multi-channel splitter for distributing iterator output.
///
/// The channel type is generic so each splitter monomorphizes to direct calls
/// into the concrete channel implementation.
pub struct ChannelSplitter<T, I, C> {
    iter: I,
    channels: Vec<C>,
    strategy: SplitStrategy,
    _phantom: PhantomData<T>,
}

/// Distribution strategy used by a channel splitter.
#[derive(Debug, Clone, Copy)]
pub enum SplitStrategy {
    /// Round-robin distribution
    RoundRobin,
    /// Load-balanced distribution
    LoadBalanced,
    /// Broadcast to all channels
    Broadcast,
}

impl<T, I, C> ChannelSplitter<T, I, C>
where
    I: Iterator<Item = T>,
    C: FusableChannel<T>,
    T: Send + Clone,
{
    /// Create a new channel splitter
    pub fn new(iter: I, strategy: SplitStrategy) -> Self {
        Self {
            iter,
            channels: Vec::new(),
            strategy,
            _phantom: PhantomData,
        }
    }

    /// Add a channel to the splitter
    pub fn add_channel(mut self, channel: C) -> Self {
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
        let mut buffers: Vec<Vec<T>> = (0..num_channels).map(|_| Vec::with_capacity(64)).collect();

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
                SplitStrategy::LoadBalanced => {
                    let min_idx = buffers
                        .iter()
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
                    self.channels[i].send_batch(items).map_err(|_| {
                        std::io::Error::new(std::io::ErrorKind::BrokenPipe, "Channel closed")
                    })?;
                }
            }
        }

        // Flush remaining items
        for (i, buffer) in buffers.into_iter().enumerate() {
            if !buffer.is_empty() {
                self.channels[i].send_batch(buffer).map_err(|_| {
                    std::io::Error::new(std::io::ErrorKind::BrokenPipe, "Channel closed")
                })?;
            }
        }

        Ok(())
    }
}

/// Channel merger for combining multiple channels into one iterator
pub struct ChannelMerger<T, C> {
    channels: Vec<C>,
    strategy: MergeStrategy,
    buffer: VecDeque<T>,
    next_channel: usize,
}

/// Merge strategy used by a channel merger.
#[derive(Debug, Clone, Copy)]
pub enum MergeStrategy {
    /// Fair round-robin merging
    FairMerge,
    /// Priority-based merging
    Priority,
    /// First-available merging
    FirstAvailable,
}

impl<T, C> ChannelMerger<T, C>
where
    C: FusableChannel<T>,
{
    /// Create a new channel merger
    pub fn new(strategy: MergeStrategy) -> Self {
        Self {
            channels: Vec::new(),
            strategy,
            buffer: VecDeque::new(),
            next_channel: 0,
        }
    }

    /// Add a channel to merge
    pub fn add_channel(mut self, channel: C) -> Self {
        self.channels.push(channel);
        self
    }
}

impl<T, C> Iterator for ChannelMerger<T, C>
where
    C: FusableChannel<T>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(item) = self.buffer.pop_front() {
            return Some(item);
        }

        match self.strategy {
            MergeStrategy::FairMerge => {
                let channel_count = self.channels.len();
                if channel_count == 0 {
                    return None;
                }

                for _ in 0..channel_count {
                    let channel_index = self.next_channel % channel_count;
                    self.next_channel = (channel_index + 1) % channel_count;

                    let items = self.channels[channel_index].recv_batch(1);
                    if !items.is_empty() {
                        self.buffer.extend(items);
                        return self.buffer.pop_front();
                    }
                }
            }
            MergeStrategy::FirstAvailable => {
                for channel in &self.channels {
                    let items = channel.recv_batch(64);
                    if !items.is_empty() {
                        self.buffer.extend(items);
                        return self.buffer.pop_front();
                    }
                }
            }
            MergeStrategy::Priority => {
                for channel in &self.channels {
                    let items = channel.recv_batch(64);
                    if !items.is_empty() {
                        self.buffer.extend(items);
                        return self.buffer.pop_front();
                    }
                }
            }
        }

        None
    }
}

/// Extension trait for iterators to add channel fusion
pub trait ChannelFusionExt: Iterator + Sized {
    /// Fuse with a channel for output
    fn fuse_channel<C>(
        self,
        channel: C,
        buffer_size: usize,
    ) -> ChannelFusedIter<Self::Item, Self, C>
    where
        C: FusableChannel<Self::Item>,
        Self::Item: Send,
    {
        ChannelFusedIter::new(self, channel, buffer_size)
    }

    /// Split output to multiple channels
    fn split_channels<C>(self, strategy: SplitStrategy) -> ChannelSplitter<Self::Item, Self, C>
    where
        C: FusableChannel<Self::Item>,
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

    impl<T> TestChannel<T> {
        fn from_items(items: Vec<T>) -> Self {
            Self {
                items: Arc::new(Mutex::new(items)),
            }
        }
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

        data.into_iter().fuse_channel(channel, 2).process().unwrap();

        let result = items_ref.lock().unwrap();
        assert_eq!(*result, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_channel_splitter() {
        let data = vec![1, 2, 3, 4, 5, 6];
        let channel1 = TestChannel {
            items: Arc::new(Mutex::new(Vec::new())),
        };
        let channel2 = TestChannel {
            items: Arc::new(Mutex::new(Vec::new())),
        };

        let items1 = channel1.items.clone();
        let items2 = channel2.items.clone();

        data.into_iter()
            .split_channels(SplitStrategy::RoundRobin)
            .add_channel(channel1)
            .add_channel(channel2)
            .process()
            .unwrap();

        assert_eq!(*items1.lock().unwrap(), vec![1, 3, 5]);
        assert_eq!(*items2.lock().unwrap(), vec![2, 4, 6]);
    }

    #[test]
    fn test_channel_merger_fair_merge_uses_fifo_order() {
        let channel1 = TestChannel::from_items(vec![1, 3]);
        let channel2 = TestChannel::from_items(vec![2, 4]);

        let merged = ChannelMerger::new(MergeStrategy::FairMerge)
            .add_channel(channel1)
            .add_channel(channel2)
            .collect::<Vec<_>>();

        assert_eq!(merged, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_channel_splitter_broadcast_clones_to_every_channel() {
        let data = vec![1, 2, 3];
        let channel1 = TestChannel {
            items: Arc::new(Mutex::new(Vec::new())),
        };
        let channel2 = TestChannel {
            items: Arc::new(Mutex::new(Vec::new())),
        };
        let items1 = channel1.items.clone();
        let items2 = channel2.items.clone();

        data.into_iter()
            .split_channels(SplitStrategy::Broadcast)
            .add_channel(channel1)
            .add_channel(channel2)
            .process()
            .unwrap();

        assert_eq!(*items1.lock().unwrap(), vec![1, 2, 3]);
        assert_eq!(*items2.lock().unwrap(), vec![1, 2, 3]);
    }
}

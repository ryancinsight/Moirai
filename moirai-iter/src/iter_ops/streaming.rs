//! Bounded streaming iterator adapter.

use std::collections::VecDeque;

/// Memory-efficient streaming iterator backed by a FIFO buffer.
pub struct StreamingIter<T, F> {
    buffer: VecDeque<T>,
    capacity: usize,
    producer: F,
}

impl<T, F> StreamingIter<T, F>
where
    F: FnMut() -> Option<T>,
{
    /// Create a new streaming iterator.
    pub fn new(capacity: usize, producer: F) -> Self {
        Self {
            buffer: VecDeque::with_capacity(capacity),
            capacity: capacity.max(1),
            producer,
        }
    }

    fn fill_buffer(&mut self) {
        while self.buffer.len() < self.capacity {
            match (self.producer)() {
                Some(item) => self.buffer.push_back(item),
                None => break,
            }
        }
    }
}

impl<T, F> Iterator for StreamingIter<T, F>
where
    F: FnMut() -> Option<T>,
{
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        if self.buffer.is_empty() {
            self.fill_buffer();
        }

        self.buffer.pop_front()
    }
}

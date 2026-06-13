//! Async range and vector iterators.

use super::traits::{AsyncIterator, IntoAsyncIterator};

/// Async iterator over a vector
pub struct AsyncVecIter<T> {
    items: Vec<T>,
}

impl<T: Send + 'static> AsyncVecIter<T> {
    /// Create a new async vector iterator
    pub fn new(items: Vec<T>) -> Self {
        Self { items }
    }
}

impl<T: Send + 'static> AsyncIterator for AsyncVecIter<T> {
    type Item = T;

    fn into_vec(self) -> Vec<Self::Item> {
        self.items
    }
}

impl<T: Send + 'static> IntoAsyncIterator for Vec<T> {
    type Item = T;
    type IntoAsyncIter = AsyncVecIter<T>;

    fn into_async_iter(self) -> Self::IntoAsyncIter {
        AsyncVecIter::new(self)
    }
}

/// Async range iterator
pub struct AsyncRangeIter {
    start: usize,
    end: usize,
}

impl AsyncRangeIter {
    /// Create a new async range iterator
    pub fn new(start: usize, end: usize) -> Self {
        Self { start, end }
    }
}

impl AsyncIterator for AsyncRangeIter {
    type Item = usize;

    fn into_vec(self) -> Vec<Self::Item> {
        (self.start..self.end).collect()
    }
}

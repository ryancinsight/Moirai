//! Core types for the executor module.
//!
//! This module contains fundamental types and identifiers used throughout
//! the executor system, following the Single Responsibility Principle.

/// A unique identifier for worker threads.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct WorkerId(usize);

impl WorkerId {
    /// Create a new worker ID.
    pub const fn new(id: usize) -> Self {
        Self(id)
    }

    /// Get the raw ID value.
    pub const fn get(self) -> usize {
        self.0
    }
}

/// I/O event types for the async runtime
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoEvent {
    /// Ready for reading
    Read,
    /// Ready for writing  
    Write,
    /// Error condition
    Error,
}
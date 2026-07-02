//! Zero-dependency stackless coroutine adapters for Moirai.
//!
//! This module provides a lightweight stackless coroutine abstraction built on
//! a re-invoked boxed `FnMut`: [`FunctionCoroutine`] resumes by calling the
//! closure again, which returns either a yielded value or a completion. It does
//! not preserve a suspended call stack — cooperative state is carried by the
//! closure's captured environment across resumes.
//!
//! # Design Principles
//!
//! - **Zero Dependencies**: Pure Rust standard library implementation
//! - **Zero-Cost Abstractions**: Compile-time optimizations with no runtime overhead
//! - **Memory Safety**: Safe resume protocol with Rust's ownership model
//! - **Unified Execution**: Works with async (via [`CoroutineFuture`]) and
//!   iterator (via [`CoroutineIterator`]) consumers
//!
//! # Architecture
//!
//! The coroutine system consists of:
//! - [`Coroutine`]: The core trait for types that can be resumed to yield values
//! - [`CoroutineState`]: Tracks coroutine execution state
//! - [`CoroutineResult`]: The outcome of a single resume (yield, complete, error)
//! - [`FunctionCoroutine`]: A closure-backed stackless coroutine
//! - [`CoroutineIterator`] / [`CoroutineFuture`]: Adapters exposing a coroutine
//!   as a standard iterator or future

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

use crate::error::TaskError;
use crate::platform::*;
use crate::{TaskContext, TaskId};

/// The state of a coroutine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoroutineState {
    /// Coroutine is created but not started
    Created,
    /// Coroutine is ready to run
    Ready,
    /// Coroutine is currently running
    Running,
    /// Coroutine has yielded a value
    Yielded,
    /// Coroutine has completed
    Completed,
    /// Coroutine encountered an error
    Error,
}

/// Core coroutine trait for types that can be executed as coroutines.
pub trait Coroutine {
    /// The type of value yielded by this coroutine
    type Yield;
    /// The type of value returned when the coroutine completes
    type Return;

    /// Resume execution of the coroutine.
    ///
    /// Returns either a yielded value or the final return value.
    fn resume(&mut self) -> CoroutineResult<Self::Yield, Self::Return>;

    /// Get the current state of the coroutine.
    fn state(&self) -> CoroutineState;

    /// Check if the coroutine can be resumed.
    fn is_resumable(&self) -> bool {
        matches!(
            self.state(),
            CoroutineState::Created | CoroutineState::Ready | CoroutineState::Yielded
        )
    }
}

/// Result type for coroutine execution.
pub enum CoroutineResult<Y, R> {
    /// Coroutine yielded a value and can be resumed
    Yielded(Y),
    /// Coroutine completed with a return value
    Complete(R),
    /// Coroutine encountered an error
    Error(TaskError),
}

/// A coroutine implementation using function pointers.
///
/// This provides a zero-cost abstraction over cooperative multitasking,
/// allowing functions to yield control and resume later.
pub struct FunctionCoroutine<Y, R> {
    /// The coroutine state machine
    state_fn: Option<Box<dyn FnMut() -> CoroutineResult<Y, R> + Send>>,
    /// Current state
    state: CoroutineState,
    /// Task context for scheduling
    _context: TaskContext,
}

impl<Y, R> FunctionCoroutine<Y, R>
where
    Y: Send + 'static,
    R: Send + 'static,
{
    /// Create a new simple coroutine.
    pub fn new<F>(func: F) -> Self
    where
        F: FnMut() -> CoroutineResult<Y, R> + Send + 'static,
    {
        Self {
            state_fn: Some(Box::new(func)),
            state: CoroutineState::Created,
            _context: TaskContext::new(TaskId::new(0)),
        }
    }
}

impl<Y, R> Coroutine for FunctionCoroutine<Y, R>
where
    Y: Send + 'static,
    R: Send + 'static,
{
    type Yield = Y;
    type Return = R;

    fn resume(&mut self) -> CoroutineResult<Self::Yield, Self::Return> {
        if let Some(mut func) = self.state_fn.take() {
            self.state = CoroutineState::Running;
            let result = func();

            match &result {
                CoroutineResult::Yielded(_) => {
                    self.state = CoroutineState::Yielded;
                    self.state_fn = Some(func);
                }
                CoroutineResult::Complete(_) => {
                    self.state = CoroutineState::Completed;
                }
                CoroutineResult::Error(_) => {
                    self.state = CoroutineState::Error;
                }
            }

            result
        } else {
            CoroutineResult::Error(TaskError::InvalidState)
        }
    }

    fn state(&self) -> CoroutineState {
        self.state
    }
}

/// Coroutine iterator adapter for using coroutines as iterators.
pub struct CoroutineIterator<C> {
    coroutine: C,
}

impl<C> CoroutineIterator<C>
where
    C: Coroutine,
{
    /// Create a new coroutine iterator.
    pub fn new(coroutine: C) -> Self {
        Self { coroutine }
    }
}

impl<C> Iterator for CoroutineIterator<C>
where
    C: Coroutine,
{
    type Item = C::Yield;

    fn next(&mut self) -> Option<Self::Item> {
        if !self.coroutine.is_resumable() {
            return None;
        }

        match self.coroutine.resume() {
            CoroutineResult::Yielded(value) => Some(value),
            CoroutineResult::Complete(_) | CoroutineResult::Error(_) => None,
        }
    }
}

/// Future adapter for coroutines, allowing them to be awaited.
pub struct CoroutineFuture<C> {
    coroutine: Option<C>,
}

impl<C> CoroutineFuture<C>
where
    C: Coroutine,
{
    /// Create a new coroutine future.
    pub fn new(coroutine: C) -> Self {
        Self {
            coroutine: Some(coroutine),
        }
    }
}

impl<C> Future for CoroutineFuture<C>
where
    C: Coroutine + Unpin,
{
    type Output = Result<C::Return, TaskError>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let Some(coroutine) = self.coroutine.as_mut() else {
            return Poll::Ready(Err(TaskError::AlreadyCompleted));
        };

        match coroutine.resume() {
            CoroutineResult::Yielded(_) => {
                cx.waker().wake_by_ref();
                Poll::Pending
            }
            CoroutineResult::Complete(value) => {
                self.coroutine = None;
                Poll::Ready(Ok(value))
            }
            CoroutineResult::Error(e) => {
                self.coroutine = None;
                Poll::Ready(Err(e))
            }
        }
    }
}

/// Extension trait for creating coroutines from closures.
pub trait CoroutineExt: Sized {
    /// The yield type of the coroutine
    type Yield;
    /// The return type of the coroutine
    type Return;

    /// Convert this value into a coroutine.
    fn into_coroutine(self) -> FunctionCoroutine<Self::Yield, Self::Return>;
}

impl<F, Y, R> CoroutineExt for F
where
    F: FnMut() -> CoroutineResult<Y, R> + Send + 'static,
    Y: Send + 'static,
    R: Send + 'static,
{
    type Yield = Y;
    type Return = R;

    fn into_coroutine(self) -> FunctionCoroutine<Y, R> {
        FunctionCoroutine::new(self)
    }
}

/// Macro for creating coroutines with yield syntax.
#[macro_export]
macro_rules! coroutine {
    ($($body:tt)*) => {{
        move || {
            $($body)*
        }
    }};
}

/// Macro for yielding from within a coroutine.
#[macro_export]
macro_rules! co_yield {
    ($value:expr) => {{
        // Cooperative yield using the simplified coroutine protocol
        return $crate::coroutine::CoroutineResult::Yielded($value);
    }};
}

/// Macro for returning from a coroutine.
#[macro_export]
macro_rules! co_return {
    ($value:expr) => {{
        return $crate::coroutine::CoroutineResult::Complete($value);
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_coroutine() {
        let mut counter = 0;
        let mut coro = FunctionCoroutine::new(move || {
            counter += 1;
            if counter < 3 {
                CoroutineResult::Yielded(counter)
            } else {
                CoroutineResult::Complete(counter)
            }
        });

        assert_eq!(coro.state(), CoroutineState::Created);

        match coro.resume() {
            CoroutineResult::Yielded(1) => {}
            _ => panic!("Expected yield of 1"),
        }

        match coro.resume() {
            CoroutineResult::Yielded(2) => {}
            _ => panic!("Expected yield of 2"),
        }

        match coro.resume() {
            CoroutineResult::Complete(3) => {}
            _ => panic!("Expected completion with 3"),
        }

        assert_eq!(coro.state(), CoroutineState::Completed);
    }

    #[test]
    fn test_coroutine_iterator() {
        let mut counter = 0;
        let coro = FunctionCoroutine::new(move || {
            counter += 1;
            if counter <= 3 {
                CoroutineResult::Yielded(counter)
            } else {
                CoroutineResult::Complete(())
            }
        });

        let iter = CoroutineIterator::new(coro);
        let values: Vec<i32> = iter.collect();
        assert_eq!(values, vec![1, 2, 3]);
    }
}

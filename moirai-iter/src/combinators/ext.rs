//! Extension trait adding combinator methods to iterators.

use super::cycle::Cycle;
use super::flat_map::FlatMap;
use super::inspect::Inspect;
use super::peekable::Peekable;
use super::scan::Scan;
use super::skip::Skip;
use super::skip_while::SkipWhile;
use super::step_by::StepBy;

/// Extension trait adding combinator methods to iterators
pub trait CombinatorExt: Iterator + Sized {
    /// Creates an iterator that maintains state while iterating
    #[inline]
    fn scan<St, B, F>(self, initial_state: St, f: F) -> Scan<Self, St, F>
    where
        F: FnMut(&mut St, Self::Item) -> Option<B>,
    {
        Scan::new(self, initial_state, f)
    }

    /// Creates an iterator that flattens nested iterators
    #[inline]
    fn flat_map<U, F>(self, f: F) -> FlatMap<Self, U, F>
    where
        U: Iterator,
        F: FnMut(Self::Item) -> U,
    {
        FlatMap::new(self, f)
    }

    /// Creates an iterator that calls a closure on each element
    #[inline]
    fn inspect<F>(self, f: F) -> Inspect<Self, F>
    where
        F: FnMut(&Self::Item),
    {
        Inspect::new(self, f)
    }

    /// Creates an iterator that can peek at the next element
    #[inline]
    fn peekable(self) -> Peekable<Self> {
        Peekable::new(self)
    }

    /// Creates an iterator that skips the first n elements
    #[inline]
    fn skip(self, n: usize) -> Skip<Self> {
        Skip::new(self, n)
    }

    /// Creates an iterator that skips elements based on a predicate
    #[inline]
    fn skip_while<P>(self, predicate: P) -> SkipWhile<Self, P>
    where
        P: FnMut(&Self::Item) -> bool,
    {
        SkipWhile::new(self, predicate)
    }

    /// Creates an iterator that yields every nth element
    #[inline]
    fn step_by(self, step: usize) -> StepBy<Self> {
        StepBy::new(self, step)
    }

    /// Creates an iterator that cycles through elements infinitely
    #[inline]
    fn cycle(self) -> Cycle<Self>
    where
        Self: Clone,
    {
        Cycle::new(self)
    }
}

impl<I: Iterator + Sized> CombinatorExt for I {}

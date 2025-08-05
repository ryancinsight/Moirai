//! Zero-cost iterator combinators for functional programming patterns
//!
//! This module provides advanced iterator combinators that enable functional
//! programming patterns with zero runtime overhead. All combinators are designed
//! to be inlined and optimized away by the compiler.
//!
//! # Design Principles
//! - **Zero-cost**: All abstractions compile to optimal machine code
//! - **Lazy evaluation**: Computation only happens when values are consumed
//! - **Composable**: Combinators can be chained without performance penalty
//! - **Type-safe**: Compile-time guarantees prevent runtime errors
//!
//! # Literature References
//! - "Iterators" by Stepanov & Lee (1995)
//! - "Stream Fusion" by Coutts, Leshchinskiy & Stewart (2007)
//! - "Functional Programming in C++" by Cukic (2018)



/// Iterator adapter that maintains state while iterating
///
/// Similar to fold but yields intermediate results
#[derive(Clone)]
pub struct Scan<I, St, F> {
    iter: I,
    state: Option<St>,
    f: F,
}

impl<I, St, F> Scan<I, St, F> {
    #[inline]
    pub fn new(iter: I, initial_state: St, f: F) -> Self {
        Scan {
            iter,
            state: Some(initial_state),
            f,
        }
    }
}

impl<B, I, St, F> Iterator for Scan<I, St, F>
where
    I: Iterator,
    F: FnMut(&mut St, I::Item) -> Option<B>,
    St: Clone,
{
    type Item = B;

    #[inline]
    fn next(&mut self) -> Option<B> {
        let item = self.iter.next()?;
        let state = self.state.as_mut()?;
        (self.f)(state, item)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper)
    }
}

/// Iterator adapter that flattens nested iterators
#[derive(Clone)]
pub struct FlatMap<I, U, F> {
    iter: I,
    f: F,
    frontiter: Option<U>,
    backiter: Option<U>,
}

impl<I, U, F> FlatMap<I, U, F> {
    #[inline]
    pub fn new(iter: I, f: F) -> Self {
        FlatMap {
            iter,
            f,
            frontiter: None,
            backiter: None,
        }
    }
}

impl<I, U, F> Iterator for FlatMap<I, U, F>
where
    I: Iterator,
    U: Iterator,
    F: FnMut(I::Item) -> U,
{
    type Item = U::Item;

    #[inline]
    fn next(&mut self) -> Option<U::Item> {
        loop {
            if let Some(ref mut inner) = self.frontiter {
                if let Some(item) = inner.next() {
                    return Some(item);
                }
            }
            match self.iter.next() {
                None => return self.backiter.as_mut()?.next(),
                Some(item) => self.frontiter = Some((self.f)(item)),
            }
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (flo, fhi) = self.frontiter.as_ref()
            .map_or((0, Some(0)), |it| it.size_hint());
        let (blo, bhi) = self.backiter.as_ref()
            .map_or((0, Some(0)), |it| it.size_hint());
        let lo = flo.saturating_add(blo);
        match (self.iter.size_hint(), fhi, bhi) {
            ((0, Some(0)), Some(fhi), Some(bhi)) => (lo, fhi.checked_add(bhi)),
            _ => (lo, None),
        }
    }
}

/// Iterator adapter for inspecting elements without consuming
#[derive(Clone)]
pub struct Inspect<I, F> {
    iter: I,
    f: F,
}

impl<I, F> Inspect<I, F> {
    #[inline]
    pub fn new(iter: I, f: F) -> Self {
        Inspect { iter, f }
    }
}

impl<I, F> Iterator for Inspect<I, F>
where
    I: Iterator,
    F: FnMut(&I::Item),
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<I::Item> {
        self.iter.next().map(|item| {
            (self.f)(&item);
            item
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }

    #[inline]
    fn count(self) -> usize {
        self.iter.count()
    }
}

impl<I, F> DoubleEndedIterator for Inspect<I, F>
where
    I: DoubleEndedIterator,
    F: FnMut(&I::Item),
{
    #[inline]
    fn next_back(&mut self) -> Option<I::Item> {
        self.iter.next_back().map(|item| {
            (self.f)(&item);
            item
        })
    }
}

impl<I, F> ExactSizeIterator for Inspect<I, F>
where
    I: ExactSizeIterator,
    F: FnMut(&I::Item),
{
    #[inline]
    fn len(&self) -> usize {
        self.iter.len()
    }
}

/// Iterator adapter that allows peeking at the next element
#[derive(Clone)]
pub struct Peekable<I: Iterator> {
    iter: I,
    peeked: Option<Option<I::Item>>,
}

impl<I: Iterator> Peekable<I> {
    #[inline]
    pub fn new(iter: I) -> Self {
        Peekable { iter, peeked: None }
    }

    /// Returns a reference to the next element without consuming it
    #[inline]
    pub fn peek(&mut self) -> Option<&I::Item> {
        let iter = &mut self.iter;
        self.peeked.get_or_insert_with(|| iter.next()).as_ref()
    }

    /// Returns a mutable reference to the next element without consuming it
    #[inline]
    pub fn peek_mut(&mut self) -> Option<&mut I::Item> {
        let iter = &mut self.iter;
        self.peeked.get_or_insert_with(|| iter.next()).as_mut()
    }
}

impl<I: Iterator> Iterator for Peekable<I> {
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<I::Item> {
        match self.peeked.take() {
            Some(v) => v,
            None => self.iter.next(),
        }
    }

    #[inline]
    fn count(mut self) -> usize {
        match self.peeked.take() {
            Some(None) => 0,
            Some(Some(_)) => 1 + self.iter.count(),
            None => self.iter.count(),
        }
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<I::Item> {
        match self.peeked.take() {
            Some(None) => None,
            Some(v @ Some(_)) if n == 0 => v,
            Some(Some(_)) => self.iter.nth(n - 1),
            None => self.iter.nth(n),
        }
    }

    #[inline]
    fn last(mut self) -> Option<I::Item> {
        let peek_opt = match self.peeked.take() {
            Some(None) => return None,
            Some(v) => v,
            None => None,
        };
        self.iter.last().or(peek_opt)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let peek_len = match self.peeked {
            Some(None) => return (0, Some(0)),
            Some(Some(_)) => 1,
            None => 0,
        };
        let (lo, hi) = self.iter.size_hint();
        let lo = lo.saturating_add(peek_len);
        let hi = match hi {
            Some(x) => x.checked_add(peek_len),
            None => None,
        };
        (lo, hi)
    }
}

/// Iterator adapter that skips a fixed number of elements
#[derive(Clone)]
pub struct Skip<I> {
    iter: I,
    n: usize,
}

impl<I> Skip<I> {
    #[inline]
    pub fn new(iter: I, n: usize) -> Self {
        Skip { iter, n }
    }
}

impl<I> Iterator for Skip<I>
where
    I: Iterator,
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<I::Item> {
        if self.n > 0 {
            let to_skip = self.n;
            self.n = 0;
            self.iter.nth(to_skip - 1)?;
        }
        self.iter.next()
    }

    #[inline]
    fn nth(&mut self, n: usize) -> Option<I::Item> {
        if self.n > 0 {
            let to_skip = self.n;
            self.n = 0;
            self.iter.nth(to_skip.saturating_add(n))
        } else {
            self.iter.nth(n)
        }
    }

    #[inline]
    fn count(mut self) -> usize {
        if self.n > 0 {
            if let Some(_) = self.iter.nth(self.n - 1) {
                1 + self.iter.count()
            } else {
                0
            }
        } else {
            self.iter.count()
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (lower, upper) = self.iter.size_hint();
        let lower = lower.saturating_sub(self.n);
        let upper = upper.map(|x| x.saturating_sub(self.n));
        (lower, upper)
    }
}

impl<I> ExactSizeIterator for Skip<I>
where
    I: ExactSizeIterator,
{
    #[inline]
    fn len(&self) -> usize {
        self.iter.len().saturating_sub(self.n)
    }
}

/// Iterator adapter that skips elements based on a predicate
#[derive(Clone)]
pub struct SkipWhile<I, P> {
    iter: I,
    flag: bool,
    predicate: P,
}

impl<I, P> SkipWhile<I, P> {
    #[inline]
    pub fn new(iter: I, predicate: P) -> Self {
        SkipWhile {
            iter,
            flag: false,
            predicate,
        }
    }
}

impl<I, P> Iterator for SkipWhile<I, P>
where
    I: Iterator,
    P: FnMut(&I::Item) -> bool,
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<I::Item> {
        let flag = &mut self.flag;
        let pred = &mut self.predicate;
        self.iter.find(move |x| {
            if *flag || !pred(x) {
                *flag = true;
                true
            } else {
                false
            }
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper)
    }
}

/// Iterator adapter that yields every nth element
#[derive(Clone)]
pub struct StepBy<I> {
    iter: I,
    step: usize,
    first_take: bool,
}

impl<I> StepBy<I> {
    #[inline]
    pub fn new(iter: I, step: usize) -> Self {
        assert!(step != 0);
        StepBy {
            iter,
            step: step - 1,
            first_take: true,
        }
    }
}

impl<I> Iterator for StepBy<I>
where
    I: Iterator,
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.first_take {
            self.first_take = false;
            self.iter.next()
        } else {
            self.iter.nth(self.step)
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (low, high) = self.iter.size_hint();
        let step = self.step + 1;
        let low = if self.first_take {
            (low + step - 1) / step
        } else {
            low / step
        };
        let high = high.map(|h| {
            if self.first_take {
                (h + step - 1) / step
            } else {
                h / step
            }
        });
        (low, high)
    }

    #[inline]
    fn nth(&mut self, mut n: usize) -> Option<Self::Item> {
        if self.first_take {
            self.first_take = false;
            let first = self.iter.next();
            if n == 0 {
                return first;
            }
            n -= 1;
        }
        let step = self.step + 1;
        if let Some(x) = n.checked_mul(step) {
            self.iter.nth(x.saturating_add(self.step))
        } else {
            self.iter.nth(self.step).and_then(|_| {
                for _ in 0..n - 1 {
                    self.iter.nth(self.step)?;
                }
                self.iter.nth(self.step)
            })
        }
    }
}

impl<I> ExactSizeIterator for StepBy<I>
where
    I: ExactSizeIterator,
{
    #[inline]
    fn len(&self) -> usize {
        let len = self.iter.len();
        let step = self.step + 1;
        if self.first_take {
            (len + step - 1) / step
        } else {
            len / step
        }
    }
}

/// Iterator adapter that cycles through the iterator infinitely
#[derive(Clone)]
pub struct Cycle<I> {
    orig: I,
    iter: I,
}

impl<I: Clone> Cycle<I> {
    #[inline]
    pub fn new(iter: I) -> Self {
        let orig = iter.clone();
        Cycle { orig, iter }
    }
}

impl<I> Iterator for Cycle<I>
where
    I: Clone + Iterator,
{
    type Item = I::Item;

    #[inline]
    fn next(&mut self) -> Option<I::Item> {
        match self.iter.next() {
            None => {
                self.iter = self.orig.clone();
                self.iter.next()
            }
            Some(x) => Some(x),
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        match self.orig.size_hint() {
            (0, Some(0)) => (0, Some(0)),
            (0, _) => (0, None),
            _ => (usize::MAX, None),
        }
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scan() {
        let numbers = vec![1, 2, 3, 4, 5];
        let sums: Vec<_> = numbers.into_iter()
            .scan(0, |state, x| {
                *state += x;
                Some(*state)
            })
            .collect();
        assert_eq!(sums, vec![1, 3, 6, 10, 15]);
    }

    #[test]
    fn test_flat_map() {
        let data = vec![vec![1, 2], vec![3, 4], vec![5]];
        let flattened: Vec<_> = data.into_iter()
            .flat_map(|v| v.into_iter())
            .collect();
        assert_eq!(flattened, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_inspect() {
        let mut inspected = Vec::new();
        let data: Vec<_> = (1..=5)
            .inspect(|x| inspected.push(*x))
            .map(|x| x * 2)
            .collect();
        assert_eq!(data, vec![2, 4, 6, 8, 10]);
        assert_eq!(inspected, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_peekable() {
        let mut iter = vec![1, 2, 3].into_iter().peekable();
        assert_eq!(iter.peek(), Some(&1));
        assert_eq!(iter.peek(), Some(&1));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.peek(), Some(&2));
    }

    #[test]
    fn test_skip() {
        let data: Vec<_> = (1..=10).skip(5).collect();
        assert_eq!(data, vec![6, 7, 8, 9, 10]);
    }

    #[test]
    fn test_skip_while() {
        let data: Vec<_> = (1..=10)
            .skip_while(|&x| x < 5)
            .collect();
        assert_eq!(data, vec![5, 6, 7, 8, 9, 10]);
    }

    #[test]
    fn test_step_by() {
        let data: Vec<_> = (0..10).step_by(2).collect();
        assert_eq!(data, vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_cycle() {
        let data: Vec<_> = vec![1, 2, 3].into_iter()
            .cycle()
            .take(10)
            .collect();
        assert_eq!(data, vec![1, 2, 3, 1, 2, 3, 1, 2, 3, 1]);
    }
}
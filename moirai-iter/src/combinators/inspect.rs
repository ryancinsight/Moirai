//! Inspect iterator adapter.

/// Iterator adapter for inspecting elements without consuming.
#[derive(Clone)]
pub struct Inspect<I, F> {
    iter: I,
    f: F,
}

impl<I, F> Inspect<I, F> {
    /// Creates a new `Inspect` adapter.
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
        self.iter.next().inspect(|item| {
            (self.f)(item);
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
        self.iter.next_back().inspect(|item| {
            (self.f)(item);
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

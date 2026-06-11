//! Cycle iterator adapter.

/// Iterator adapter that cycles through the iterator infinitely.
#[derive(Clone)]
pub struct Cycle<I> {
    orig: I,
    iter: I,
}

impl<I: Clone> Cycle<I> {
    /// Creates a new `Cycle` adapter.
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

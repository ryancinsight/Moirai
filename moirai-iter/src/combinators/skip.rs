//! Skip iterator adapter.

/// Iterator adapter that skips a fixed number of elements.
#[derive(Clone)]
pub struct Skip<I> {
    iter: I,
    n: usize,
}

impl<I> Skip<I> {
    /// Creates a new `Skip` adapter.
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
            if self.iter.nth(self.n - 1).is_some() {
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

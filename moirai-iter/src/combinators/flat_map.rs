//! FlatMap iterator adapter.

/// Iterator adapter that flattens nested iterators.
#[derive(Clone)]
pub struct FlatMap<I, U, F> {
    iter: I,
    f: F,
    frontiter: Option<U>,
    backiter: Option<U>,
}

impl<I, U, F> FlatMap<I, U, F> {
    /// Creates a new `FlatMap` adapter.
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
        let (flo, fhi) = self
            .frontiter
            .as_ref()
            .map_or((0, Some(0)), |it| it.size_hint());
        let (blo, bhi) = self
            .backiter
            .as_ref()
            .map_or((0, Some(0)), |it| it.size_hint());
        let lo = flo.saturating_add(blo);
        match (self.iter.size_hint(), fhi, bhi) {
            ((0, Some(0)), Some(fhi), Some(bhi)) => (lo, fhi.checked_add(bhi)),
            _ => (lo, None),
        }
    }
}

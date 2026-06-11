//! Scan iterator adapter.

/// Iterator adapter that maintains state while iterating.
///
/// Similar to fold but yields intermediate results.
#[derive(Clone)]
pub struct Scan<I, St, F> {
    iter: I,
    state: Option<St>,
    f: F,
}

impl<I, St, F> Scan<I, St, F> {
    /// Creates a new `Scan` adapter.
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

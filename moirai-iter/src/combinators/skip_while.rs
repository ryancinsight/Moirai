//! SkipWhile iterator adapter.

/// Iterator adapter that skips elements based on a predicate.
#[derive(Clone)]
pub struct SkipWhile<I, P> {
    iter: I,
    flag: bool,
    predicate: P,
}

impl<I, P> SkipWhile<I, P> {
    /// Creates a new `SkipWhile` adapter.
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

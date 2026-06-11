//! StepBy iterator adapter.

/// Iterator adapter that yields every nth element.
#[derive(Clone)]
pub struct StepBy<I> {
    iter: I,
    step: usize,
    first_take: bool,
}

impl<I> StepBy<I> {
    /// Creates a new `StepBy` adapter.
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
            low.div_ceil(step)
        } else {
            low / step
        };
        let high = high.map(|h| {
            if self.first_take {
                h.div_ceil(step)
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
            len.div_ceil(step)
        } else {
            len / step
        }
    }
}

//! Stateful iterator adapters.

use std::marker::PhantomData;

/// Zero-copy scan iterator that maintains state without cloning.
pub struct ScanRef<I, St, F> {
    pub(crate) iter: I,
    pub(crate) state: St,
    pub(crate) f: F,
}

impl<I, St, F, B> Iterator for ScanRef<I, St, F>
where
    I: Iterator,
    F: FnMut(&mut St, I::Item) -> Option<B>,
{
    type Item = B;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        let item = self.iter.next()?;
        (self.f)(&mut self.state, item)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let (_, upper) = self.iter.size_hint();
        (0, upper)
    }
}

/// Fold an iterator while passing each item to the accumulator by reference.
pub fn fold_ref<I, B, F>(iter: I, init: B, mut f: F) -> B
where
    I: Iterator,
    F: FnMut(B, &I::Item) -> B,
{
    let mut accum = init;
    for item in iter {
        accum = f(accum, &item);
    }
    accum
}

/// Zero-copy partition iterator.
pub struct PartitionRef<I, F> {
    pub(crate) iter: I,
    pub(crate) predicate: F,
}

impl<I, F> PartitionRef<I, F>
where
    I: Iterator,
    F: FnMut(&I::Item) -> bool,
{
    /// Consume the iterator and partition into two collections.
    pub fn partition<A, B>(mut self) -> (A, B)
    where
        A: Default + Extend<I::Item>,
        B: Default + Extend<I::Item>,
    {
        let mut left = A::default();
        let mut right = B::default();

        for item in self.iter {
            if (self.predicate)(&item) {
                left.extend(Some(item));
            } else {
                right.extend(Some(item));
            }
        }

        (left, right)
    }
}

/// Iterator adapter for in-place modification.
pub struct UpdateInPlace<'a, T, I, F> {
    iter: I,
    updater: F,
    _phantom: PhantomData<&'a mut T>,
}

impl<'a, T, I, F> UpdateInPlace<'a, T, I, F> {
    /// Create an in-place update adapter.
    pub fn new(iter: I, updater: F) -> Self {
        Self {
            iter,
            updater,
            _phantom: PhantomData,
        }
    }
}

impl<'a, T, I, F> Iterator for UpdateInPlace<'a, T, I, F>
where
    I: Iterator<Item = &'a mut T>,
    F: FnMut(&mut T),
{
    type Item = &'a mut T;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|item| {
            (self.updater)(item);
            item
        })
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

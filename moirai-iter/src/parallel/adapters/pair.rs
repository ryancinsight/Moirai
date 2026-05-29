use super::super::{Consumer, ParallelIterator, VecParIter};

/// Zip adapter with standard shortest-input value semantics.
pub struct Zip<I, J> {
    left: I,
    right: J,
}

impl<I, J> Zip<I, J> {
    pub(in crate::parallel) fn new(left: I, right: J) -> Self {
        Self { left, right }
    }
}

impl<I, J> ParallelIterator for Zip<I, J>
where
    I: ParallelIterator,
    J: ParallelIterator,
    I::Item: Sync + 'static,
    J::Item: Sync + 'static,
{
    type Item = (I::Item, J::Item);

    fn seq_items(self) -> Vec<Self::Item> {
        self.left
            .seq_items()
            .into_iter()
            .zip(self.right.seq_items())
            .collect()
    }

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        consumer.consume(VecParIter::new(self.seq_items()))
    }
}

/// Zip adapter with Rayon-style equal-length value semantics.
pub struct ZipEq<I, J> {
    left: I,
    right: J,
}

impl<I, J> ZipEq<I, J> {
    pub(in crate::parallel) fn new(left: I, right: J) -> Self {
        Self { left, right }
    }
}

impl<I, J> ParallelIterator for ZipEq<I, J>
where
    I: ParallelIterator,
    J: ParallelIterator,
    I::Item: Sync + 'static,
    J::Item: Sync + 'static,
{
    type Item = (I::Item, J::Item);

    fn seq_items(self) -> Vec<Self::Item> {
        let left = self.left.seq_items();
        let right = self.right.seq_items();
        assert_eq!(
            left.len(),
            right.len(),
            "zip_eq requires equal input lengths"
        );
        left.into_iter().zip(right).collect()
    }

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        consumer.consume(VecParIter::new(self.seq_items()))
    }
}

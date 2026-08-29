use super::super::{Consumer, ParallelIterator, VecParIter};

fn interleave_all<T>(left: Vec<T>, right: Vec<T>) -> Vec<T> {
    let len = left.len().checked_add(right.len()).expect("overflow");
    let paired_len = left.len().min(right.len());
    let mut left = left.into_iter();
    let mut right = right.into_iter();
    let mut output = Vec::with_capacity(len);

    for _ in 0..paired_len {
        output.push(left.next().expect("paired length is bounded by left input"));
        output.push(
            right
                .next()
                .expect("paired length is bounded by right input"),
        );
    }
    output.extend(left);
    output.extend(right);
    output
}

fn interleave_shortest<T>(mut left: Vec<T>, mut right: Vec<T>) -> Vec<T> {
    let left_take = if left.len() <= right.len() {
        left.len()
    } else {
        right.len() + 1
    };
    let right_take = right.len().min(left.len());

    let output_len = left_take.checked_add(right_take).expect("overflow");
    left.truncate(left_take);
    right.truncate(right_take);

    let mut left = left.into_iter();
    let mut right = right.into_iter();
    let mut output = Vec::with_capacity(output_len);
    for _ in 0..right_take {
        output.push(left.next().expect("take count is bounded by left input"));
        output.push(right.next().expect("take count is bounded by right input"));
    }
    output.extend(left);
    output
}

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

    /// # Why this stays sequential (two sources, one split)
    ///
    /// Pairing needs both inputs split at the same logical positions. The
    /// consumer protocol splits one source and hands the halves one consumer
    /// each; it carries no way to divide a second, independently shaped
    /// iterator in lockstep. Splitting the left input alone and re-driving the
    /// right per shard would re-run the right input once per shard, which is a
    /// different program, not a parallelisation of this one. `zip_eq`,
    /// `interleave`, and `interleave_shortest` share this boundary; interleaving
    /// additionally depends on alternation parity carrying across the shard
    /// boundary.
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
    pub(in crate::parallel) left: I,
    pub(in crate::parallel) right: J,
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

    /// # Why this stays sequential
    ///
    /// Two sources split in lockstep, per [`Zip`].
    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        consumer.consume(VecParIter::new(self.seq_items()))
    }
}

/// Indexed interleave adapter with Rayon-style alternating value semantics.
pub struct Interleave<I, J> {
    pub(in crate::parallel) left: I,
    pub(in crate::parallel) right: J,
}

impl<I, J> Interleave<I, J> {
    pub(in crate::parallel) fn new(left: I, right: J) -> Self {
        Self { left, right }
    }
}

impl<I, J> ParallelIterator for Interleave<I, J>
where
    I: ParallelIterator,
    J: ParallelIterator<Item = I::Item>,
    I::Item: Sync + 'static,
{
    type Item = I::Item;

    fn seq_items(self) -> Vec<Self::Item> {
        interleave_all(self.left.seq_items(), self.right.seq_items())
    }

    /// # Why this stays sequential
    ///
    /// Two sources split in lockstep, per [`Zip`].
    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        consumer.consume(VecParIter::new(self.seq_items()))
    }
}

/// Indexed interleave adapter that truncates at the shorter input boundary.
pub struct InterleaveShortest<I, J> {
    pub(in crate::parallel) left: I,
    pub(in crate::parallel) right: J,
}

impl<I, J> InterleaveShortest<I, J> {
    pub(in crate::parallel) fn new(left: I, right: J) -> Self {
        Self { left, right }
    }
}

impl<I, J> ParallelIterator for InterleaveShortest<I, J>
where
    I: ParallelIterator,
    J: ParallelIterator<Item = I::Item>,
    I::Item: Sync + 'static,
{
    type Item = I::Item;

    fn seq_items(self) -> Vec<Self::Item> {
        interleave_shortest(self.left.seq_items(), self.right.seq_items())
    }

    /// # Why this stays sequential
    ///
    /// Two sources split in lockstep, per [`Zip`].
    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        consumer.consume(VecParIter::new(self.seq_items()))
    }
}

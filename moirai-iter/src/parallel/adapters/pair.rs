use super::super::{Consumer, ParallelIterator, VecParIter};

fn interleave_all<T>(left: Vec<T>, right: Vec<T>) -> Vec<T> {
    let len = left.len().checked_add(right.len()).expect("overflow");
    let mut output: Vec<T> = Vec::with_capacity(len);
    let output_ptr: *mut T = output.as_mut_ptr();
    let left = std::mem::ManuallyDrop::new(left);
    let right = std::mem::ManuallyDrop::new(right);
    let left_len = left.len();
    let right_len = right.len();
    let paired_len = left_len.min(right_len);
    let mut written = 0usize;

    // Safety: the source vectors are owned and wrapped in `ManuallyDrop`, so
    // every `read` moves an initialized element exactly once. `output` has
    // capacity for the total source length and each write uses a distinct slot.
    unsafe {
        let left_ptr = left.as_ptr();
        let right_ptr = right.as_ptr();

        for index in 0..paired_len {
            output_ptr.add(written).write(left_ptr.add(index).read());
            written += 1;
            output_ptr.add(written).write(right_ptr.add(index).read());
            written += 1;
        }

        if left_len > right_len {
            for index in paired_len..left_len {
                output_ptr.add(written).write(left_ptr.add(index).read());
                written += 1;
            }
        } else {
            for index in paired_len..right_len {
                output_ptr.add(written).write(right_ptr.add(index).read());
                written += 1;
            }
        }

        output.set_len(written);
    }
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

    let mut output: Vec<T> = Vec::with_capacity(output_len);
    let output_ptr: *mut T = output.as_mut_ptr();
    let left = std::mem::ManuallyDrop::new(left);
    let right = std::mem::ManuallyDrop::new(right);
    let mut written = 0usize;

    // Safety: the take counts are bounded by the source lengths. Each source
    // slot is read at most once and written into unique initialized output
    // capacity before setting the final length.
    unsafe {
        let left_ptr = left.as_ptr();
        let right_ptr = right.as_ptr();

        for index in 0..right_take {
            output_ptr.add(written).write(left_ptr.add(index).read());
            written += 1;
            output_ptr.add(written).write(right_ptr.add(index).read());
            written += 1;
        }

        if left_take > right_take {
            output_ptr
                .add(written)
                .write(left_ptr.add(right_take).read());
            written += 1;
        }

        output.set_len(written);
    }
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

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        consumer.consume(VecParIter::new(self.seq_items()))
    }
}

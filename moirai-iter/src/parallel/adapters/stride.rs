use super::super::{Consumer, ParallelIterator, VecParIter};

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct StepSize(usize);

impl StepSize {
    fn new(value: usize) -> Self {
        assert!(value != 0, "step size must be non-zero");
        Self(value)
    }

    const fn get(self) -> usize {
        self.0
    }
}

/// Indexed step-by adapter with exact-size source semantics.
pub struct StepBy<I> {
    pub(in crate::parallel) base: I,
    step: StepSize,
}

impl<I> StepBy<I> {
    pub(in crate::parallel) fn new(base: I, step: usize) -> Self {
        Self {
            base,
            step: StepSize::new(step),
        }
    }

    pub(in crate::parallel) fn step(&self) -> usize {
        self.step.get()
    }
}

impl<I> ParallelIterator for StepBy<I>
where
    I: ParallelIterator,
    I::Item: Sync + 'static,
{
    type Item = I::Item;

    fn seq_items(self) -> Vec<Self::Item> {
        self.base
            .seq_items()
            .into_iter()
            .step_by(self.step.get())
            .collect()
    }

    /// # Why this stays sequential
    ///
    /// Which of a shard's items survive depends on the shard's logical start
    /// index modulo `step`, the same absent offset documented on
    /// [`Enumerate`](super::ref_ops::Enumerate).
    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        consumer.consume(VecParIter::new(self.seq_items()))
    }
}

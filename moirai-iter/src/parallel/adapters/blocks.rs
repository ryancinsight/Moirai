use super::super::{Consumer, ParallelIterator, VecParIter};

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct ExponentialBlockPolicy;

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct UniformBlockPolicy;

#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct BlockSize(usize);

impl BlockSize {
    fn new(value: usize) -> Self {
        assert!(value != 0, "block size must be non-zero");
        Self(value)
    }
}

/// Indexed source adapter for exponential logical block scheduling.
pub struct ExponentialBlocks<I> {
    base: I,
    policy: ExponentialBlockPolicy,
}

impl<I> ExponentialBlocks<I> {
    pub(in crate::parallel) fn new(base: I) -> Self {
        Self {
            base,
            policy: ExponentialBlockPolicy,
        }
    }
}

impl<I> ParallelIterator for ExponentialBlocks<I>
where
    I: ParallelIterator,
    I::Item: Sync + 'static,
{
    type Item = I::Item;

    fn seq_items(self) -> Vec<Self::Item> {
        let _policy = self.policy;
        self.base.seq_items()
    }

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        consumer.consume(VecParIter::new(self.seq_items()))
    }
}

/// Indexed source adapter for uniform logical block scheduling.
pub struct UniformBlocks<I> {
    base: I,
    block_size: BlockSize,
    policy: UniformBlockPolicy,
}

impl<I> UniformBlocks<I> {
    pub(in crate::parallel) fn new(base: I, block_size: usize) -> Self {
        Self {
            base,
            block_size: BlockSize::new(block_size),
            policy: UniformBlockPolicy,
        }
    }
}

impl<I> ParallelIterator for UniformBlocks<I>
where
    I: ParallelIterator,
    I::Item: Sync + 'static,
{
    type Item = I::Item;

    fn seq_items(self) -> Vec<Self::Item> {
        let _block_size = self.block_size;
        let _policy = self.policy;
        self.base.seq_items()
    }

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        consumer.consume(VecParIter::new(self.seq_items()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_policy_markers_are_zero_sized() {
        assert_eq!(std::mem::size_of::<ExponentialBlockPolicy>(), 0);
        assert_eq!(std::mem::size_of::<UniformBlockPolicy>(), 0);
    }
}

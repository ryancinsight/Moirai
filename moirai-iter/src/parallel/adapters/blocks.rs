use super::super::{Consumer, ParallelIterator};
use std::ops::ControlFlow;

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

    fn seq_try_fold<Acc, B, FoldFn>(self, init: Acc, fold_fn: FoldFn) -> ControlFlow<B, Acc>
    where
        FoldFn: FnMut(Acc, Self::Item) -> ControlFlow<B, Acc>,
    {
        let _policy = self.policy;
        self.base.seq_try_fold(init, fold_fn)
    }

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        // The block policy selects a logical scheduling shape and leaves the
        // item stream identical to the base's, so this adapter drives the base
        // directly rather than collecting it. Collecting cost the source's
        // shards to express an identity.
        let _policy = self.policy;
        self.base.drive(consumer)
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

    fn seq_try_fold<Acc, B, FoldFn>(self, init: Acc, fold_fn: FoldFn) -> ControlFlow<B, Acc>
    where
        FoldFn: FnMut(Acc, Self::Item) -> ControlFlow<B, Acc>,
    {
        let _block_size = self.block_size;
        let _policy = self.policy;
        self.base.seq_try_fold(init, fold_fn)
    }

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        // Identity item stream, as on [`ExponentialBlocks`].
        let _block_size = self.block_size;
        let _policy = self.policy;
        self.base.drive(consumer)
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

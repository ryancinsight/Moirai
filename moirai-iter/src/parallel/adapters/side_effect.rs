use super::super::{Consumer, InspectConsumer, ParallelIterator};

/// Inspect adapter that observes items by shared reference without changing them.
pub struct Inspect<I, F> {
    base: I,
    inspect_fn: F,
}

impl<I, F> Inspect<I, F> {
    pub(in crate::parallel) fn new(base: I, inspect_fn: F) -> Self {
        Self { base, inspect_fn }
    }
}

impl<I, F> ParallelIterator for Inspect<I, F>
where
    I: ParallelIterator,
    F: Fn(&I::Item) + Send + Sync + Clone,
    I::Item: Sync + 'static,
{
    type Item = I::Item;

    fn seq_items(self) -> Vec<Self::Item> {
        let inspect_fn = self.inspect_fn;
        let items = self.base.seq_items();
        for item in &items {
            inspect_fn(item);
        }
        items
    }

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        self.base
            .drive(InspectConsumer::new(consumer, self.inspect_fn))
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct PanicFusePolicy;

/// Panic-fuse adapter for the non-indexed iterator subset.
pub struct PanicFuse<I> {
    base: I,
    _policy: PanicFusePolicy,
}

impl<I> PanicFuse<I> {
    pub(in crate::parallel) fn new(base: I) -> Self {
        Self {
            base,
            _policy: PanicFusePolicy,
        }
    }
}

impl<I> ParallelIterator for PanicFuse<I>
where
    I: ParallelIterator,
    I::Item: Sync,
{
    type Item = I::Item;

    fn seq_items(self) -> Vec<Self::Item> {
        self.base.seq_items()
    }

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        self.base.drive(consumer)
    }
}

#[cfg(test)]
mod tests {
    use super::PanicFusePolicy;

    #[test]
    fn panic_fuse_policy_is_zero_sized() {
        assert_eq!(std::mem::size_of::<PanicFusePolicy>(), 0);
    }
}

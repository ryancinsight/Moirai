use super::super::{Consumer, ParallelIterator, VecParIter};

/// Take adapter with prefix-bounded value semantics.
pub struct Take<I> {
    pub(super) base: I,
    pub(super) count: usize,
}

impl<I> Take<I> {
    pub(crate) fn new(base: I, count: usize) -> Self {
        Self { base, count }
    }
}

impl<I> ParallelIterator for Take<I>
where
    I: ParallelIterator,
    I::Item: Sync + 'static,
{
    type Item = I::Item;

    fn seq_items(self) -> Vec<Self::Item> {
        self.base.seq_items_window(0, Some(self.count))
    }

    fn seq_items_window(self, skip: usize, take: Option<usize>) -> Vec<Self::Item> {
        if skip >= self.count {
            return Vec::new();
        }

        let remaining = self.count - skip;
        let count = take.map_or(remaining, |count| count.min(remaining));
        self.base.seq_items_window(skip, Some(count))
    }

    fn seq_items_reversed(self) -> Vec<Self::Item> {
        self.base
            .seq_items_window(0, Some(self.count))
            .into_iter()
            .rev()
            .collect()
    }

    fn seq_items_reversed_prefix(self, count: usize) -> Vec<Self::Item> {
        let mut items = self.base.seq_items_window(0, Some(self.count));
        let keep = count.min(items.len());
        items.drain(..items.len().saturating_sub(keep));
        items.reverse();
        items
    }

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        consumer.consume(VecParIter::new(self.seq_items()))
    }
}

/// Skip adapter with prefix-discarding value semantics.
pub struct Skip<I> {
    pub(super) base: I,
    pub(super) count: usize,
}

impl<I> Skip<I> {
    pub(crate) fn new(base: I, count: usize) -> Self {
        Self { base, count }
    }
}

impl<I> ParallelIterator for Skip<I>
where
    I: ParallelIterator,
    I::Item: Sync + 'static,
{
    type Item = I::Item;

    fn seq_items(self) -> Vec<Self::Item> {
        self.base.seq_items_window(self.count, None)
    }

    fn seq_items_window(self, skip: usize, take: Option<usize>) -> Vec<Self::Item> {
        self.base
            .seq_items_window(self.count.saturating_add(skip), take)
    }

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        consumer.consume(VecParIter::new(self.seq_items()))
    }
}

/// Chain adapter with left-to-right concatenation semantics.
pub struct Chain<I, J> {
    pub(super) left: I,
    pub(super) right: J,
}

impl<I, J> Chain<I, J> {
    pub(crate) fn new(left: I, right: J) -> Self {
        Self { left, right }
    }
}

impl<I, J> ParallelIterator for Chain<I, J>
where
    I: ParallelIterator,
    J: ParallelIterator<Item = I::Item>,
    I::Item: Sync + 'static,
{
    type Item = I::Item;

    fn seq_items(self) -> Vec<Self::Item> {
        let mut left = self.left.seq_items();
        left.extend(self.right.seq_items());
        left
    }

    fn seq_items_reversed(self) -> Vec<Self::Item> {
        let mut items = self.right.seq_items_reversed();
        items.extend(self.left.seq_items_reversed());
        items
    }

    fn seq_items_reversed_prefix(self, count: usize) -> Vec<Self::Item> {
        let mut items = self.right.seq_items_reversed_prefix(count);
        if items.len() < count {
            items.extend(self.left.seq_items_reversed_prefix(count - items.len()));
        }
        items
    }

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        consumer.consume(VecParIter::new(self.seq_items()))
    }
}

/// Intersperse adapter with separator insertion between adjacent items.
pub struct Intersperse<I>
where
    I: ParallelIterator,
{
    pub(super) base: I,
    pub(super) separator: I::Item,
}

impl<I> Intersperse<I>
where
    I: ParallelIterator,
{
    pub(crate) fn new(base: I, separator: I::Item) -> Self {
        Self { base, separator }
    }
}

impl<I> ParallelIterator for Intersperse<I>
where
    I: ParallelIterator,
    I::Item: Clone + Sync + 'static,
{
    type Item = I::Item;

    fn seq_items(self) -> Vec<Self::Item> {
        let items = self.base.seq_items();
        if items.len() <= 1 {
            return items;
        }

        let mut output = Vec::with_capacity(items.len().saturating_mul(2).saturating_sub(1));
        let mut iter = items.into_iter();
        if let Some(first) = iter.next() {
            output.push(first);
        }
        for item in iter {
            output.push(self.separator.clone());
            output.push(item);
        }
        output
    }

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        consumer.consume(VecParIter::new(self.seq_items()))
    }
}

/// Reverse adapter with logical-order reversal semantics.
pub struct Rev<I> {
    pub(super) base: I,
}

impl<I> Rev<I> {
    pub(crate) fn new(base: I) -> Self {
        Self { base }
    }
}

impl<I> ParallelIterator for Rev<I>
where
    I: ParallelIterator,
    I::Item: Sync + 'static,
{
    type Item = I::Item;

    fn seq_items(self) -> Vec<Self::Item> {
        self.base.seq_items_reversed()
    }

    fn seq_items_window(self, skip: usize, take: Option<usize>) -> Vec<Self::Item> {
        let count = take.unwrap_or(usize::MAX);
        let prefix = skip.saturating_add(count);
        let mut items = self.base.seq_items_reversed_prefix(prefix);
        if skip >= items.len() {
            return Vec::new();
        }
        items.drain(..skip);
        if let Some(count) = take {
            items.truncate(count);
        }
        items
    }

    fn seq_items_reversed(self) -> Vec<Self::Item> {
        self.base.seq_items()
    }

    fn seq_items_reversed_prefix(self, count: usize) -> Vec<Self::Item> {
        self.base.seq_items_window(0, Some(count))
    }

    fn drive<C, R>(self, consumer: C) -> R
    where
        C: Consumer<Self::Item, Result = R> + Send + Sync,
        R: Send,
    {
        consumer.consume(VecParIter::new(self.seq_items()))
    }
}

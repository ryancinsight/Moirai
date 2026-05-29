use std::ops::ControlFlow;

use super::ParallelIterator;

mod private {
    pub trait Sealed {}

    impl<T> Sealed for Option<T> {}
    impl<T, E> Sealed for Result<T, E> {}
}

/// Sealed fallible stream item contract for `try_reduce_with`.
pub trait TryStreamItem: private::Sealed + Send {
    type Output: Send;

    fn branch(self) -> ControlFlow<Self, Self::Output>
    where
        Self: Sized;

    fn from_output(output: Self::Output) -> Self;
}

impl<T> TryStreamItem for Option<T>
where
    T: Send,
{
    type Output = T;

    fn branch(self) -> ControlFlow<Self, Self::Output> {
        match self {
            Some(value) => ControlFlow::Continue(value),
            None => ControlFlow::Break(None),
        }
    }

    fn from_output(output: Self::Output) -> Self {
        Some(output)
    }
}

impl<T, E> TryStreamItem for Result<T, E>
where
    T: Send,
    E: Send,
{
    type Output = T;

    fn branch(self) -> ControlFlow<Self, Self::Output> {
        match self {
            Ok(value) => ControlFlow::Continue(value),
            Err(error) => ControlFlow::Break(Err(error)),
        }
    }

    fn from_output(output: Self::Output) -> Self {
        Ok(output)
    }
}

pub(in crate::parallel) fn try_reduce_with<I, F>(iterator: I, reduce_fn: F) -> Option<I::Item>
where
    I: ParallelIterator,
    I::Item: TryStreamItem,
    F: Fn(<I::Item as TryStreamItem>::Output, <I::Item as TryStreamItem>::Output) -> I::Item
        + Send
        + Sync
        + Clone,
{
    try_reduce_with_items(iterator.seq_items(), reduce_fn)
}

pub(in crate::parallel) fn try_reduce_with_items<Items, Item, F>(
    items: Items,
    reduce_fn: F,
) -> Option<Item>
where
    Items: IntoIterator<Item = Item>,
    Item: TryStreamItem,
    F: Fn(<Item as TryStreamItem>::Output, <Item as TryStreamItem>::Output) -> Item
        + Send
        + Sync
        + Clone,
{
    let mut items = items.into_iter();
    let first = items.next()?;
    let mut accumulator = match first.branch() {
        ControlFlow::Continue(value) => value,
        ControlFlow::Break(residual) => return Some(residual),
    };

    for item in items {
        let value = match item.branch() {
            ControlFlow::Continue(value) => value,
            ControlFlow::Break(residual) => return Some(residual),
        };
        accumulator = match reduce_fn(accumulator, value).branch() {
            ControlFlow::Continue(value) => value,
            ControlFlow::Break(residual) => return Some(residual),
        };
    }

    Some(Item::from_output(accumulator))
}

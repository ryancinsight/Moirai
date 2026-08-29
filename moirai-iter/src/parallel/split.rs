use super::{FoldConsumer, ParallelIterator};

/// Sum type used by `ParallelIterator::partition_map`.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Either<L, R> {
    /// Route the value into the left output collection.
    Left(L),
    /// Route the value into the right output collection.
    Right(R),
}

pub(in crate::parallel) fn partition_map<I, A, B, P, L, R>(iterator: I, predicate: P) -> (A, B)
where
    I: ParallelIterator,
    A: Default + Extend<L> + Send,
    B: Default + Extend<R> + Send,
    P: Fn(I::Item) -> Either<L, R> + Send + Sync + Clone,
    L: Send,
    R: Send,
{
    // `A`/`B` are only `Extend`, so two partial outputs cannot be merged in
    // their own representation. Shards route items through vectors, which do
    // merge in shard order, and the outputs are extended once from the result.
    let (left_values, right_values) = iterator
        .drive(FoldConsumer::new(
            || (Vec::new(), Vec::new()),
            move |(mut left, mut right): (Vec<L>, Vec<R>), item| {
                match predicate(item) {
                    Either::Left(value) => left.push(value),
                    Either::Right(value) => right.push(value),
                }
                (left, right)
            },
            |(mut left, mut right): (Vec<L>, Vec<R>),
             (mut later_left, mut later_right): (Vec<L>, Vec<R>)| {
                left.append(&mut later_left);
                right.append(&mut later_right);
                (left, right)
            },
        ))
        .into_value();

    let mut left = A::default();
    let mut right = B::default();
    left.extend(left_values);
    right.extend(right_values);

    (left, right)
}

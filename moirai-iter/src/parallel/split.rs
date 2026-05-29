use super::ParallelIterator;

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
    let mut left = A::default();
    let mut right = B::default();

    for item in iterator.seq_items() {
        match predicate(item) {
            Either::Left(value) => left.extend(std::iter::once(value)),
            Either::Right(value) => right.extend(std::iter::once(value)),
        }
    }

    (left, right)
}

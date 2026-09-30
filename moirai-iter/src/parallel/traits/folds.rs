use super::super::{FoldConsumer, ParallelIterator};
use std::ops::ControlFlow;

#[inline]
pub(super) fn seq_mutate_state<I, Init, T, F>(iter: I, init: Init, mut op: F) -> T
where
    I: ParallelIterator,
    Init: FnOnce() -> T,
    F: FnMut(&mut T, I::Item),
{
    iter.seq_fold(init(), |mut state, item| {
        op(&mut state, item);
        state
    })
}

#[inline]
pub(super) fn seq_try_mutate_state<I, Init, T, F, E>(
    iter: I,
    init: Init,
    mut op: F,
) -> Result<(), E>
where
    I: ParallelIterator,
    Init: FnOnce() -> T,
    F: FnMut(&mut T, I::Item) -> Result<(), E>,
{
    let folded = iter.seq_try_fold((init(), Ok(())), |(mut state, _), item| {
        match op(&mut state, item) {
            Ok(()) => ControlFlow::Continue((state, Ok(()))),
            Err(error) => ControlFlow::Break((state, Err(error))),
        }
    });
    let (_, outcome) = match folded {
        ControlFlow::Continue(state) | ControlFlow::Break(state) => state,
    };

    outcome
}

#[inline]
pub(super) fn reassociated_fold<I, O, Empty, Single, Combine>(
    iter: I,
    empty: Empty,
    single: Single,
    combine: Combine,
) -> O
where
    I: ParallelIterator,
    O: Send,
    Empty: Fn() -> O + Send + Sync + Clone,
    Single: Fn(I::Item) -> O + Send + Sync + Clone,
    Combine: Fn(O, O) -> O + Send + Sync + Clone,
{
    iter.drive(FoldConsumer::new(
        empty,
        {
            let single = single.clone();
            let combine_step = combine.clone();
            move |accumulator: O, item: I::Item| combine_step(accumulator, single(item))
        },
        combine,
    ))
    .into_value()
}

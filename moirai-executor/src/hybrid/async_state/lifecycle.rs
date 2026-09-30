//! Lifecycle token cell contents of an async task.

use crate::registry::{
    OwnedStateLease, RunningTaskToken, SchedulerStateLease, StateLease, TaskLifecycleToken,
};

pub(super) enum AsyncLifecycle<L: StateLease> {
    Registered(TaskLifecycleToken<L>),
    Running(RunningTaskToken<L>),
    Completed,
}

// The lifecycle cell is shared under `AsyncFutureState`'s manual `Send`/`Sync`
// impls below, which state `L: StateLease` as their only lease bound. That
// bound suffices only while every variant is `Send` for every lease.
const _: [fn(); 2] = {
    const fn assert_send<T: Send>() {}
    fn lifecycle_is_send<L: StateLease>() {
        assert_send::<AsyncLifecycle<L>>();
    }
    [
        lifecycle_is_send::<OwnedStateLease>,
        lifecycle_is_send::<SchedulerStateLease>,
    ]
};

//! Register task ownership and wakeup in the executor lifecycle table.

use super::{
    future_state::AsyncFutureState,
    tasks::{AsyncTasks, Registration},
};
use crate::{registry::StateLease, schedule::WorkSubmit};
use moirai_core::error::ExecutorResult;
use std::{future::Future, sync::Arc, task::Waker};

impl<S, F, L> AsyncFutureState<S, F, L>
where
    S: WorkSubmit,
    F: Future + Send + 'static,
    F::Output: Send + 'static,
    L: StateLease,
{
    pub(crate) fn track(self: &Arc<Self>, tasks: &Arc<AsyncTasks>, id: u64) -> ExecutorResult<()> {
        let registration = Registration {
            tasks: Arc::downgrade(tasks),
            id,
        };
        assert!(
            self.registration.set(registration).is_ok(),
            "invariant: each future is registered once"
        );
        tasks.register(id, Waker::from(Arc::clone(self)))
    }
}

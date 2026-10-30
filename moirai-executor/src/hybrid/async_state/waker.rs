//! `AsyncFutureState` as its own `Waker`.

use std::{future::Future, sync::Arc, task::Wake};

use super::{future_state::AsyncFutureState, tasks::with_registration_context};
use crate::{registry::StateLease, schedule::WorkSubmit};

impl<S, F, L> Wake for AsyncFutureState<S, F, L>
where
    S: WorkSubmit,
    F: Future + Send + 'static,
    F::Output: Send + 'static,
    L: StateLease,
{
    fn wake(self: Arc<Self>) {
        with_registration_context(self.registration(), || self.schedule_wake());
    }

    fn wake_by_ref(self: &Arc<Self>) {
        with_registration_context(self.registration(), || self.schedule_wake());
    }
}

use std::sync::Arc;

use moirai_core::{
    Priority,
    error::ExecutorResult,
    task::{TaskHandle, TaskId},
};

use super::super::AsyncFutureState;
use crate::metrics::ExecutorMetrics;
use crate::registry::TaskRegistry;
use crate::schedule::{WorkClass, WorkSubmit};

impl WorkSubmit for Arc<TaskRegistry> {
    fn schedule<C, F>(
        &self,
        _priority: Priority,
        _locality_hint: Option<usize>,
        task: F,
    ) -> ExecutorResult<()>
    where
        C: WorkClass,
        F: FnOnce(usize) + Send + 'static,
    {
        task(0);
        Ok(())
    }
}

#[test]
fn scheduled_lifecycle_retires_before_its_registry_owner() {
    let registry = Arc::new(TaskRegistry::new());
    // SAFETY: the async state owns the only remaining registry Arc
    // through its scheduler field, which is declared after lifecycle.
    let (task_id, lifecycle) = unsafe { registry.register_next_scheduled_task() };
    let registry_owner = Arc::downgrade(&registry);
    let (_handle, result_sender) = TaskHandle::<()>::new_pending(TaskId(task_id));
    let state = AsyncFutureState::new(
        Arc::clone(&registry),
        std::future::pending::<()>(),
        lifecycle,
        result_sender,
        Arc::new(ExecutorMetrics::new()),
    );

    drop(registry);
    assert!(registry_owner.upgrade().is_some());
    drop(state);
    assert!(registry_owner.upgrade().is_none());
}

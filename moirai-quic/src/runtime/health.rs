//! Retain spawner rejection across Quinn's infallible Runtime::spawn ABI.

use crate::{QuicError, lock_mutex};
use moirai_core::error::ExecutorError;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};

#[derive(Default)]
pub(crate) struct RuntimeHealth {
    failed: AtomicBool,
    error: Mutex<Option<Arc<ExecutorError>>>,
}

impl RuntimeHealth {
    pub(crate) fn fail(&self, error: ExecutorError) {
        let mut retained = lock_mutex(&self.error);
        if retained.is_none() {
            *retained = Some(Arc::new(error));
            self.failed.store(true, Ordering::Release);
        }
    }

    pub(crate) fn check(&self) -> Result<(), QuicError> {
        if self.failed.load(Ordering::Acquire) {
            let retained = lock_mutex(&self.error);
            if let Some(error) = retained.as_ref() {
                return Err(QuicError::Executor(Arc::clone(error)));
            }
        }
        Ok(())
    }
}

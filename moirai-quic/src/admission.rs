//! Ownership bounds live application stream handles and pending admissions.

use crate::QuicError;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

pub(crate) struct StreamAdmission {
    active: AtomicU32,
    limit: u32,
}

impl StreamAdmission {
    pub(crate) fn new(limit: u32) -> Arc<Self> {
        Arc::new(Self {
            active: AtomicU32::new(0),
            limit,
        })
    }

    pub(crate) fn reserve(this: &Arc<Self>) -> Result<StreamPermit, QuicError> {
        // The counter publishes no payload. Its atomic modification order alone
        // proves 0 <= active <= limit, even when admission/drop race.
        this.active
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |active| {
                active.checked_add(1).filter(|next| *next <= this.limit)
            })
            .map_err(|_| QuicError::StreamCapacity)?;
        Ok(StreamPermit {
            admission: Arc::clone(this),
        })
    }
}

pub(crate) struct StreamPermit {
    admission: Arc<StreamAdmission>,
}

impl Drop for StreamPermit {
    fn drop(&mut self) {
        // Only a successful reservation constructs this owner; one Drop
        // therefore consumes exactly one positive count. No aliases own it.
        self.admission.active.fetch_sub(1, Ordering::Relaxed);
    }
}

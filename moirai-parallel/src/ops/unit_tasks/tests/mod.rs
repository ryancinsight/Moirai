use crate::policy::ExecutionPolicy;
use std::sync::Mutex;

/// Last `(len, chunks, bytes)` the operator reported to a policy.
pub(super) static REPORTED: Mutex<Option<(usize, usize, usize)>> = Mutex::new(None);

pub(super) struct Reporting;

impl ExecutionPolicy for Reporting {
    fn parallelize(_len: usize) -> bool {
        false
    }

    fn parallelize_work(len: usize, chunks: usize, bytes: usize) -> bool {
        *REPORTED.lock().expect("the reporting policy never panics") = Some((len, chunks, bytes));
        false
    }
}

mod many;
mod pair;
mod range;
mod single;
mod triple;

/// Bytes per unit that makes a task carry exactly four units.
pub(super) const FOUR_UNITS: usize = super::UNIT_TASK_BYTES / 4;

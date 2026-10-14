//! Atomic, bounded encoding into the diagnostic record quota.

use crate::webview_limits::{MAX_WEBVIEW_EVENTS, MAX_WEBVIEW_MESSAGE_BYTES};
use serde_json::Value;
use std::io::{self, Write};

/// An entire encoded record, or a counted omission after sticky exhaustion.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum RecordAdmission {
    Admitted(String),
    Exhausted { first: bool },
}

/// Exact retained record accounting; encoded bytes exclude the outer layer.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct BudgetSnapshot {
    pub(crate) records: usize,
    pub(crate) bytes: usize,
    pub(crate) omitted: u64,
    pub(crate) exhausted: bool,
}

/// One aggregate quota, with no selective admission after exhaustion.
#[derive(Debug, Default)]
pub(crate) struct TraceBudget {
    accounting: BudgetSnapshot,
}

impl TraceBudget {
    pub(crate) fn record(&mut self, value: &Value) -> io::Result<RecordAdmission> {
        if self.accounting.exhausted || self.accounting.records == MAX_WEBVIEW_EVENTS {
            return self.omit();
        }
        let mut writer = BoundedRecord::new(MAX_WEBVIEW_MESSAGE_BYTES - self.accounting.bytes);
        if let Err(error) = serde_json::to_writer(&mut writer, value) {
            if writer.overflow {
                return self.omit();
            }
            return Err(io::Error::new(io::ErrorKind::InvalidData, error));
        }
        // Discard spare capacity before transferring ownership to the caller;
        // retaining many admitted strings cannot retain each attempt's quota.
        let encoded = String::from_utf8(writer.bytes.into_boxed_slice().into_vec())
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?;
        self.accounting.records += 1;
        self.accounting.bytes += encoded.len();
        Ok(RecordAdmission::Admitted(encoded))
    }

    pub(crate) const fn snapshot(&self) -> BudgetSnapshot {
        self.accounting
    }

    fn omit(&mut self) -> io::Result<RecordAdmission> {
        let omitted = self
            .accounting
            .omitted
            .checked_add(1)
            .ok_or_else(|| io::Error::other("Startup omission count is exhausted"))?;
        let first = !self.accounting.exhausted;
        self.accounting.omitted = omitted;
        self.accounting.exhausted = true;
        Ok(RecordAdmission::Exhausted { first })
    }
}

struct BoundedRecord {
    bytes: Vec<u8>,
    limit: usize,
    overflow: bool,
}

impl BoundedRecord {
    fn new(limit: usize) -> Self {
        Self {
            // Reserving the admitted remainder once prevents geometric Vec
            // growth from exceeding the encoded-byte storage bound.
            bytes: Vec::with_capacity(limit),
            limit,
            overflow: false,
        }
    }
}

impl Write for BoundedRecord {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if bytes.len() > self.limit - self.bytes.len() {
            self.overflow = true;
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "Startup record exceeds the remaining encoded-byte quota",
            ));
        }
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        // Every byte already belongs to the in-memory record; no downstream
        // stream or deferred write exists to flush.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn omitted_counter_exhaustion_preserves_the_last_exact_snapshot() {
        let _trace = crate::webview_startup::tests::trace();
        let _test = tracing::info_span!("startup_omission_counter").entered();
        let mut budget = tracing::info_span!("arrange").in_scope(|| TraceBudget {
            accounting: BudgetSnapshot {
                records: MAX_WEBVIEW_EVENTS,
                bytes: MAX_WEBVIEW_EVENTS * 4,
                omitted: u64::MAX - 1,
                exhausted: true,
            },
        });
        let last = tracing::info_span!("act")
            .in_scope(|| budget.record(&Value::Null).expect("last exact omission"));
        let before = budget.snapshot();
        let error = tracing::info_span!("act")
            .in_scope(|| budget.record(&Value::Null).expect_err("exhausted count"));
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(last, RecordAdmission::Exhausted { first: false });
            assert_eq!(before.omitted, u64::MAX);
            assert_eq!(budget.snapshot(), before);
            assert_eq!(error.kind(), io::ErrorKind::Other);
            assert_eq!(error.to_string(), "Startup omission count is exhausted");
        });
    }
}

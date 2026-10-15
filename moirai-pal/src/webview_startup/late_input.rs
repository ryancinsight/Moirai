//! Default-off suffix retention from already-copied controller observations.
//!
//! See `late_input/design.md`. Arming is monotonic; no filter runs after the
//! actual anchor. Checkpoints spend the same quota as observations.

use super::{RecordAdmission, TraceBudget};
use serde_json::{Value, json};
use std::{fmt, io, num::NonZeroU8};

#[derive(Debug)]
pub(crate) enum LateInputFailure {
    InvalidDecimal,
    SharedOmission,
    SharedContention,
    Budget(io::Error),
}

impl LateInputFailure {
    fn kind(&self) -> &'static str {
        match self {
            Self::InvalidDecimal => "invalid_canonical_decimal",
            Self::SharedOmission => "shared_omission_count_exhausted",
            Self::SharedContention => "shared_contention_count_exhausted",
            Self::Budget(_) => "trace_budget_error",
        }
    }
}

impl fmt::Display for LateInputFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.kind())
    }
}

impl std::error::Error for LateInputFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        // Type erasure is required by the standard error-source API.
        match self {
            Self::Budget(error) => Some(error),
            _ => None,
        }
    }
}

type Result<T> = std::result::Result<T, LateInputFailure>;

pub(crate) const LATE_INPUT_TARGET: &str = "moirai_pal::windows::webview::late_input";
pub(crate) const LATE_INPUT_EVENT: &str = "webview_late_input_observation";
/// Longest kind/failure names, maximal counter widths and longer false
/// booleans bound terminals. Initial/progress checkpoints spend the quota.
pub(crate) const LATE_INPUT_TERMINAL_BYTES: usize = r#"{"kind":"late_input_quota_exhausted","after_virtual_key":255,"armed":false,"anchor_sequence":"18446744073709551615","anchor_elapsed_ns":"340282366920938463463374607431768211455","drained_through_sequence":"18446744073709551615","elapsed_ns":"340282366920938463463374607431768211455","records":256,"bytes":65536,"omitted":"18446744073709551615","shared_capacity_omitted":"18446744073709551615","shared_contended":"18446744073709551615","failed":false,"error":"shared_contention_count_exhausted","cleanup_failed":false,"complete":false}"#.len();
const _: () = assert!(crate::webview_limits::MAX_WEBVIEW_EVENTS == 256);
const _: () = assert!(crate::webview_limits::MAX_WEBVIEW_MESSAGE_BYTES == 65_536);

pub(crate) struct LateInput {
    after_key: NonZeroU8,
    anchor: Option<(u64, u128)>,
    budget: TraceBudget,
    drained: (u64, Option<u128>),
    shared_omitted: u64,
    shared_contended: u64,
    failed: bool,
    error: Option<LateInputFailure>,
    pending: bool,
    closed: bool,
}

impl LateInput {
    pub(crate) fn new(after_key: NonZeroU8) -> Self {
        Self {
            after_key,
            anchor: None,
            budget: TraceBudget::default(),
            drained: (0, None),
            shared_omitted: 0,
            shared_contended: 0,
            failed: false,
            error: None,
            pending: false,
            closed: false,
        }
    }

    pub(crate) fn observe(&mut self, value: &Value) -> Result<Option<String>> {
        if self.closed {
            return Ok(None);
        }
        let event = &value["event"];
        if self.anchor.is_none()
            && value["kind"] == "controller_input"
            && event["name"] == "AcceleratorKeyPressed"
            && event["virtual_key"].as_u64() == Some(u64::from(self.after_key.get()))
            // The SDK kinds are KEY_DOWN=0 and SYSTEM_KEY_DOWN=2.
            && matches!(event["event_kind"].as_i64(), Some(0 | 2))
        {
            let sequence = decimal::<u64>(&value["sequence"])?;
            let elapsed = decimal::<u128>(&value["elapsed_ns"])?;
            self.anchor = Some((sequence, elapsed));
        }
        let read_error = event["name"] == "read_error";
        self.failed |= read_error;
        if self.anchor.is_none() && !read_error {
            return Ok(None);
        }
        self.pending = true;
        self.record(value)
    }

    pub(crate) fn shared_loss(&mut self, omitted: u64, contended: u64) -> Result<()> {
        let omitted = self
            .shared_omitted
            .checked_add(omitted)
            .ok_or(LateInputFailure::SharedOmission)?;
        let contended = self
            .shared_contended
            .checked_add(contended)
            .ok_or(LateInputFailure::SharedContention)?;
        self.pending |= omitted != self.shared_omitted || contended != self.shared_contended;
        self.shared_omitted = omitted;
        self.shared_contended = contended;
        self.failed |= omitted != 0 || contended != 0;
        Ok(())
    }

    pub(crate) fn checkpoint(
        &mut self,
        sequence: u64,
        elapsed: Option<u128>,
        initial: bool,
    ) -> Result<Option<String>> {
        if self.closed || (!initial && !self.pending) {
            return Ok(None);
        }
        self.drained = (sequence, elapsed);
        self.failed |= sequence == u64::MAX;
        self.pending = false;
        self.record(&self.metadata("late_input_checkpoint", false))
    }

    pub(crate) fn close(&mut self, cleanup_failed: bool) -> Result<Option<String>> {
        if self.closed {
            return Ok(None);
        }
        self.closed = true;
        self.failed |= cleanup_failed;
        Ok(Some(self.terminal("late_input_closed", cleanup_failed)))
    }

    pub(crate) fn fail(&mut self, error: LateInputFailure) {
        self.failed = true;
        self.pending = true;
        self.error.get_or_insert(error);
    }

    fn record(&mut self, value: &Value) -> Result<Option<String>> {
        match self
            .budget
            .record(value)
            .map_err(LateInputFailure::Budget)?
        {
            RecordAdmission::Admitted(encoded) => Ok(Some(encoded)),
            RecordAdmission::Exhausted { first: true } => {
                Ok(Some(self.terminal("late_input_quota_exhausted", false)))
            }
            RecordAdmission::Exhausted { first: false } => Ok(None),
        }
    }

    fn terminal(&self, kind: &str, cleanup_failed: bool) -> String {
        let encoded = self.metadata(kind, cleanup_failed).to_string();
        assert!(
            encoded.len() <= LATE_INPUT_TERMINAL_BYTES,
            "late input terminal field bound"
        );
        encoded
    }

    fn metadata(&self, kind: &str, cleanup_failed: bool) -> Value {
        let budget = self.budget.snapshot();
        json!({"kind":kind,"after_virtual_key":self.after_key.get(),
            "armed":self.anchor.is_some(),
            "anchor_sequence":self.anchor.map(|a|a.0.to_string()),
            "anchor_elapsed_ns":self.anchor.map(|a|a.1.to_string()),
            "drained_through_sequence":self.drained.0.to_string(),
            "elapsed_ns":self.drained.1.map(|elapsed|elapsed.to_string()),
            "records":budget.records,"bytes":budget.bytes,"omitted":budget.omitted.to_string(),
            "shared_capacity_omitted":self.shared_omitted.to_string(),
            "shared_contended":self.shared_contended.to_string(),
            "failed":self.failed,"error":self.error.as_ref().map(LateInputFailure::kind),"cleanup_failed":cleanup_failed,
            "complete":self.closed && self.anchor.is_some() && !self.failed && !budget.exhausted})
    }
}

fn decimal<T: std::str::FromStr + fmt::Display>(value: &Value) -> Result<T> {
    let text = value.as_str().ok_or(LateInputFailure::InvalidDecimal)?;
    let parsed = text
        .parse::<T>()
        .map_err(|_| LateInputFailure::InvalidDecimal)?;
    if parsed.to_string() != text {
        return Err(LateInputFailure::InvalidDecimal);
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests;

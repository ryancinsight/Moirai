//! Private, explicitly selected native startup observations.
//!
//! Selection admits exactly one configured URI and a finite set of selectors;
//! it grants no navigation, input or page-lifecycle authority. Configuration
//! and retained records share the existing native/browser representation law.
//!
//! A trace record is admitted atomically: for encoded lengths `b_i`,
//! `sum(b_i) <= MAX_WEBVIEW_MESSAGE_BYTES` and the admitted record count is at
//! most `MAX_WEBVIEW_EVENTS`. Exhaustion is sticky and omitted attempts remain
//! counted. No prefix of an oversized record becomes evidence.
//!
//! The selected diagnostic JSON layer has no timestamp, spans or current-span
//! fields. Its empty observation envelope has fixed length `E`. Compact JSON
//! contains no raw control characters, so putting an encoded record inside a
//! second JSON string adds at most one byte per existing byte. Thus ordinary
//! records occupy at most `2*B + E*N` bytes. Two separately bounded terminal
//! metadata records add at most `4*B + 2*E`, giving `6*B + E*(N+2)`, below the
//! existing 1 MiB Metis consumer limit for the tested exact envelope. This
//! bounds retained Rust bytes; it makes no claim about Chromium's allocations
//! before a native callback receives a response.
//!
//! One request slot covers the entire asynchronous DOM chain. Invalidation
//! changes its generation without releasing the in-flight owner; only that
//! request's terminal callback clears the slot. Close makes every reply stale
//! and refuses new work. A late or unrelated reply cannot clear another owner.
//! These are point observations under the existing native deadline, never a
//! manufactured Ready event or a guarantee that a browser callback completes.

mod components;
mod config;
#[cfg(any(all(windows, feature = "webview2"), test))]
mod deferred;
#[cfg(any(all(windows, feature = "webview2"), test))]
mod late_input;
mod quota;
mod state;

#[cfg(all(windows, feature = "webview2", test))]
pub(crate) use components::Components;
#[cfg(all(windows, feature = "webview2"))]
pub(crate) use config::Selection;
#[cfg(all(windows, feature = "webview2"))]
pub(crate) use deferred::Deferred;
#[cfg(all(windows, feature = "webview2", test))]
pub(crate) use late_input::LATE_INPUT_TERMINAL_BYTES;
#[cfg(all(windows, feature = "webview2"))]
pub(crate) use late_input::{LATE_INPUT_EVENT, LATE_INPUT_TARGET, LateInput, LateInputFailure};
#[cfg(test)]
pub(crate) use quota::BudgetSnapshot;
pub(crate) use quota::{RecordAdmission, TraceBudget};
#[cfg(all(windows, feature = "webview2"))]
pub(crate) use state::RequestId;
#[cfg(test)]
pub(crate) use state::{Finish, RequestSnapshot};
pub(crate) use state::{RequestSlot, Start};

/// The dedicated diagnostic layer's exact tracing target.
pub(crate) const TRACE_TARGET: &str = "moirai_pal::windows::webview::startup";
/// Explicit fixture selection; ordinary hosts do not enable observations.
pub(crate) const SELECTION_ENV: &str = "MOIRAI_WEBVIEW_STARTUP_OBSERVATION";

#[cfg(test)]
mod tests;

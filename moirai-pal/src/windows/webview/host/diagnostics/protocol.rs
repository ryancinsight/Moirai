//! Actual native status and complete admitted reply context, independent of state.

use crate::webview_startup::RequestId;
use serde_json::{Value, json};
use windows::core::{HRESULT, PCWSTR};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Stage {
    Document,
    Query { index: usize, root: i64 },
    Html { index: usize, root: i64 },
}

/// One bounded, not-yet-issued stage. Its original request identity survives
/// admission denial; no SDK callback exists for this command until it is issued.
/// Parameters contain either fixed document options, one bounded node ID, or
/// one selector from the already byte-bounded Selection. Only one command is
/// retained; completed response ownership transfers into this successor.
pub(super) struct Command {
    pub(super) id: RequestId,
    pub(super) stage: Stage,
    pub(super) method: &'static str,
    pub(super) parameters: Value,
}

#[derive(Debug)]
pub(super) struct Failure {
    pub(super) error: windows::core::Error,
    response: Option<String>,
    decoding_error: Option<windows::core::Error>,
}

pub(super) type Reply = Result<Value, Failure>;

impl Failure {
    pub(super) fn protocol(error: windows::core::Error, value: &Value) -> Self {
        Self {
            error,
            response: Some(value.to_string()),
            decoding_error: None,
        }
    }

    pub(super) fn value(&self, operation: &str) -> Value {
        json!({"kind":"native_error","operation":operation,"hresult":self.error.code().0,
            "message":self.error.to_string(),"response":self.response,
            "decoding_error":self.decoding_error.as_ref().map(|error|
                json!({"hresult":error.code().0,"message":error.to_string()}))})
    }
}

/// Reads only the SDK's borrowed terminated reply, with representation admission.
///
/// # Safety
/// A nonnull pointer denotes a complete SDK-owned terminated UTF16 allocation
/// valid through this call. No pointer or borrow survives conversion.
pub(super) unsafe fn read(status: HRESULT, result: &PCWSTR) -> Reply {
    if let Err(error) = status.ok() {
        let (response, decoding_error) = if result.is_null() {
            (None, None)
        } else {
            // SAFETY: the caller supplies the complete borrowed SDK allocation.
            match unsafe { super::super::text::read_borrowed_message(result) } {
                Ok(response) => (Some(response), None),
                Err(error) => (None, Some(error)),
            }
        };
        return Err(Failure {
            error,
            response,
            decoding_error,
        });
    }
    // SAFETY: the caller supplies the complete borrowed SDK allocation; null
    // is rejected by the canonical scanner without a pointer read.
    let encoded =
        unsafe { super::super::text::read_borrowed_message(result) }.map_err(|error| Failure {
            error,
            response: None,
            decoding_error: None,
        })?;
    let value: Value = serde_json::from_str(&encoded).map_err(|error| Failure {
        error: invalid(format!("WebView2 CDP JSON: {error}")),
        response: Some(encoded.clone()),
        decoding_error: None,
    })?;
    let error = if !value.is_object() {
        Some(invalid("WebView2 CDP response is not an object"))
    } else if value.get("error").is_some() {
        Some(invalid("WebView2 CDP error response"))
    } else {
        None
    };
    if let Some(error) = error {
        return Err(Failure {
            error,
            response: Some(encoded),
            decoding_error: None,
        });
    }
    Ok(value)
}

fn invalid(message: impl AsRef<str>) -> windows::core::Error {
    windows::core::Error::new(windows::Win32::Foundation::E_ABORT, message.as_ref())
}

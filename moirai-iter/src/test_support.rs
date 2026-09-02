//! Helpers shared by this crate's test modules.

use std::any::Any;

/// The message a panic carried, whichever string form `panic!` chose.
///
/// A literal panics with `&'static str`, a formatted message with `String`.
/// Asserting on the message rather than on `is_err()` pins *which* panic
/// unwound, so a sentinel test cannot pass on an unrelated failure.
pub(crate) fn panic_message(payload: &(dyn Any + Send)) -> &str {
    payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .expect("invariant: the panic payload is a string message")
}

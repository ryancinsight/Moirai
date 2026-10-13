//! ECMAScript decimal formatting at the actual WebIDL Number boundary.

use std::io;

use js_sys::Number;

use super::browser_error::javascript_error;

/// Browser-native number formatting for DOM values.
pub struct WebNumber;

impl WebNumber {
    /// Returns the actual browser's ECMAScript base10 number string.
    ///
    /// Preserves ECMAScript exponent thresholds, shortest round-trip digits,
    /// subnormal values, negative-zero spelling and nonfinite number labels.
    /// `f64` is the WebIDL/ECMAScript Number representation; no precision
    /// conversion or application JavaScript formatting takes place.
    ///
    /// # Errors
    /// Returns the actual JavaScript formatting exception with context.
    pub fn decimal(value: f64) -> io::Result<String> {
        Number::from(value)
            .to_string_with_radix(10)
            .map(String::from)
            .map_err(|error| {
                javascript_error("Number.toString radix10", io::ErrorKind::Other, error)
            })
    }
}

#[cfg(test)]
mod tests;

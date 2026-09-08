//! Bounds for browser text metadata before it reaches an application.

use std::io;

/// Maximum UTF-8 bytes retained for one browser text value or event payload.
pub(crate) const MAX_TEXT_BYTES: usize = 1_048_576;
/// Maximum UTF-8 bytes retained for one browser input-operation name.
pub(crate) const MAX_INPUT_TYPE_BYTES: usize = 128;
/// Maximum UTF-8 bytes retained for a composition locale.
pub(crate) const MAX_LOCALE_BYTES: usize = 64;

pub(crate) fn text_value(value: String) -> io::Result<String> {
    bounded(
        value,
        MAX_TEXT_BYTES,
        "Browser text value exceeds the bounded size",
    )
}

pub(crate) fn event_data(data: Option<String>) -> io::Result<Option<String>> {
    data.map(|value| {
        bounded(
            value,
            MAX_TEXT_BYTES,
            "Browser composition data exceeds the bounded size",
        )
    })
    .transpose()
}

pub(crate) fn input_type(value: String) -> io::Result<String> {
    bounded(
        value,
        MAX_INPUT_TYPE_BYTES,
        "Browser input type exceeds the bounded size",
    )
}

pub(crate) fn locale(value: String) -> io::Result<String> {
    bounded(
        value,
        MAX_LOCALE_BYTES,
        "Browser composition locale exceeds the bounded size",
    )
}

fn bounded(value: String, maximum: usize, message: &'static str) -> io::Result<String> {
    if value.len() > maximum {
        return Err(io::Error::new(io::ErrorKind::InvalidInput, message));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::{
        event_data, input_type, locale, text_value, MAX_INPUT_TYPE_BYTES, MAX_LOCALE_BYTES,
        MAX_TEXT_BYTES,
    };

    #[test]
    fn text_payload_bounds_are_measured_in_utf8_bytes() {
        let accepted = "é".repeat(MAX_TEXT_BYTES / "é".len());
        assert_eq!(
            text_value(accepted.clone()).expect("bounded text"),
            accepted
        );
        assert!(text_value(format!("{accepted}é")).is_err());
    }

    #[test]
    fn event_data_preserves_absence_and_rejects_oversize_values() {
        assert_eq!(event_data(None).expect("absent event data"), None);
        assert_eq!(
            event_data(Some("preedit".to_owned())).expect("bounded event data"),
            Some("preedit".to_owned())
        );
        assert!(event_data(Some("x".repeat(MAX_TEXT_BYTES + 1))).is_err());
    }

    #[test]
    fn input_type_and_locale_have_independent_bounds() {
        assert!(input_type("x".repeat(MAX_INPUT_TYPE_BYTES)).is_ok());
        assert!(input_type("x".repeat(MAX_INPUT_TYPE_BYTES + 1)).is_err());
        assert!(locale("x".repeat(MAX_LOCALE_BYTES)).is_ok());
        assert!(locale("x".repeat(MAX_LOCALE_BYTES + 1)).is_err());
    }
}

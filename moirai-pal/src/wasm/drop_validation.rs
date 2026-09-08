//! Pure validation for browser file-drop metadata.

use std::io;

// A FileList reports a u32 length; 64 entries cap metadata allocation before
// an application can apply its own format policy.
#[cfg(target_arch = "wasm32")]
pub(crate) const MAX_FILE_COUNT: u32 = 64;
// Names and media types are copied into owned Rust strings at the trust
// boundary; these limits bound one event's metadata footprint.
const MAX_FILE_NAME_BYTES: usize = 4_096;
const MAX_MEDIA_TYPE_BYTES: usize = 256;

pub(crate) fn file_name(value: String) -> io::Result<String> {
    bounded_text(value, MAX_FILE_NAME_BYTES, false, "file name")
}

pub(crate) fn media_type(value: String) -> io::Result<String> {
    bounded_text(value, MAX_MEDIA_TYPE_BYTES, true, "media type")
}

pub(crate) fn parse_size(value: f64) -> io::Result<u64> {
    if !value.is_finite() || value.is_sign_negative() || value.fract() != 0.0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Browser file size is not a finite non-negative integer",
        ));
    }
    value.to_string().parse::<u64>().map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "Browser file size cannot be represented",
        )
    })
}

fn bounded_text(
    value: String,
    maximum: usize,
    allow_empty: bool,
    label: &str,
) -> io::Result<String> {
    if (!allow_empty && value.is_empty()) || value.len() > maximum || value.contains('\0') {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("Browser {label} metadata is invalid or exceeds its bound"),
        ));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::{file_name, media_type, parse_size};
    use std::io::ErrorKind;

    #[test]
    fn file_name_validation_preserves_bounded_display_metadata() {
        assert_eq!(
            file_name("scan.dcm".to_owned()).expect("valid name"),
            "scan.dcm"
        );
        assert_eq!(
            file_name("".to_owned())
                .expect_err("empty file names must be rejected")
                .kind(),
            ErrorKind::InvalidInput
        );
        assert_eq!(
            file_name("bad\0name".to_owned())
                .expect_err("NUL in file names must be rejected")
                .kind(),
            ErrorKind::InvalidInput
        );
        assert_eq!(
            file_name("x".repeat(4_097))
                .expect_err("oversized file names must be rejected")
                .kind(),
            ErrorKind::InvalidInput
        );
    }

    #[test]
    fn media_type_allows_unknown_type_but_rejects_unbounded_metadata() {
        assert_eq!(
            media_type(String::new()).expect("unknown type is valid"),
            ""
        );
        assert_eq!(
            media_type("application/dicom".to_owned()).expect("valid type"),
            "application/dicom"
        );
        assert_eq!(
            media_type("x".repeat(257))
                .expect_err("oversized media types must be rejected")
                .kind(),
            ErrorKind::InvalidInput
        );
    }

    #[test]
    fn file_size_requires_a_finite_integer_representable_by_u64() {
        assert_eq!(parse_size(0.0).expect("zero size"), 0);
        assert_eq!(parse_size(1024.0).expect("integer size"), 1024);
        for value in [f64::NAN, f64::INFINITY, -1.0, 1.5, f64::MAX] {
            assert_eq!(
                parse_size(value)
                    .expect_err(
                        "non-finite, negative, fractional, or overflowing sizes must be rejected"
                    )
                    .kind(),
                ErrorKind::InvalidInput
            );
        }
    }
}

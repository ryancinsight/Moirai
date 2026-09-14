//! Bounds for browser keyboard names shared by native validation tests and WASM.

use std::io;

/// Maximum UTF-8 bytes retained for one browser key or code name.
pub(crate) const MAX_KEY_NAME_BYTES: usize = 64;

pub(crate) fn bounded_name(value: String) -> io::Result<String> {
    if value.len() > MAX_KEY_NAME_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "browser keyboard name exceeds the provider bound",
        ));
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::{MAX_KEY_NAME_BYTES, bounded_name};

    #[test]
    fn keyboard_names_are_bounded() {
        assert_eq!(
            bounded_name("ArrowDown".to_owned()).expect("short key"),
            "ArrowDown"
        );
        assert!(bounded_name("x".repeat(MAX_KEY_NAME_BYTES + 1)).is_err());
    }
}

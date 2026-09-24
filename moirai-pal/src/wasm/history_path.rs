//! Validation of the paths an application may put in the browser history.

use std::io;

/// Longest path accepted for a history entry, in bytes.
pub const MAX_HISTORY_PATH_BYTES: usize = 2048;

/// Admits a same-origin path: it starts with one `/`, is printable ASCII
/// without `\`, and fits [`MAX_HISTORY_PATH_BYTES`]. A leading `//` would
/// name another host, and a `\` is read as `/` by browsers, so both are
/// refused; history entries can never leave the page's origin.
pub(crate) fn validate(path: &str) -> io::Result<()> {
    let admitted = path.len() <= MAX_HISTORY_PATH_BYTES
        && path.starts_with('/')
        && !path.starts_with("//")
        && path
            .bytes()
            .all(|byte| byte.is_ascii_graphic() && byte != b'\\');
    if admitted {
        Ok(())
    } else {
        Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "history path must be a printable same-origin path",
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::{MAX_HISTORY_PATH_BYTES, validate};

    #[test]
    fn only_same_origin_paths_are_admitted() {
        for path in ["/", "/study/7", "/study/a%20b?tab=info#top"] {
            assert!(validate(path).is_ok(), "{path}");
        }
        let long = format!("/{}", "a".repeat(MAX_HISTORY_PATH_BYTES));
        for path in [
            "",
            "study",
            "//evil.example/x",
            "/\\evil.example",
            "https://evil.example/",
            "/a b",
            "/tab\t",
            "/é",
            &long,
        ] {
            assert!(validate(path).is_err(), "{path:?}");
        }
    }
}

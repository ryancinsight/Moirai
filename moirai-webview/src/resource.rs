//! Same-origin packaged resource law, independent of native framework handles.

use std::{
    fs, io,
    path::{Component, Path, PathBuf},
};

/// Largest single packaged resource, matching a WebAssembly module's bounded
/// loading contract rather than allowing unbounded allocation from a URL.
pub const MAX_RESOURCE_BYTES: u64 = 64 * 1024 * 1024;
pub(crate) const ORIGIN: &str = "metis://app";

/// Canonical, bounded resource directory for one application.
#[derive(Debug, Clone)]
pub struct ResourceRoot(PathBuf);

impl ResourceRoot {
    /// Validates a real resource directory and its HTML entry point.
    ///
    /// # Errors
    /// Returns the filesystem error or `InvalidInput` for a non-directory root.
    pub fn new(path: impl AsRef<Path>) -> io::Result<Self> {
        let root = path.as_ref().canonicalize()?;
        if !root.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "resource root is not a directory",
            ));
        }
        let result = Self(root);
        result.read("metis://app/index.html")?;
        Ok(result)
    }

    /// Reads one same-origin regular file with a fixed allocation bound.
    ///
    /// # Errors
    /// Returns `PermissionDenied` for another origin or an escaping path,
    /// `InvalidInput` for malformed URL encoding, and the concrete I/O failure.
    pub fn read(&self, uri: &str) -> io::Result<(Vec<u8>, &'static str)> {
        use io::Read;
        if !Self::permits(uri) {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "resource URL has another origin",
            ));
        }
        // permits proves the ASCII origin and slash prefix, including the
        // length and UTF-8 boundary required by this slice.
        let encoded = &uri[ORIGIN.len() + 1..];
        let encoded = encoded.split(['?', '#']).next().unwrap_or(encoded);
        let decoded = decode_path(encoded)?;
        let relative = Path::new(&decoded);
        if decoded.contains('\\')
            || relative
                .components()
                .any(|part| !matches!(part, Component::Normal(_)))
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "resource path escapes its root",
            ));
        }
        let path = self.0.join(relative).canonicalize()?;
        if !path.starts_with(&self.0) {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "resource symlink escapes its root",
            ));
        }
        validate_file(&fs::metadata(&path)?)?;
        let mut options = fs::File::options();
        options.read(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            // Nonblocking open prevents a replaced FIFO from waiting for a
            // writer; no-follow refuses replacement of the canonical leaf
            // by a link. Validate the opened descriptor before reading it.
            options.custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW);
        }
        let file = options.open(&path)?;
        validate_file(&file.metadata()?)?;
        let mut bytes = Vec::new();
        file.take(MAX_RESOURCE_BYTES + 1).read_to_end(&mut bytes)?;
        if u64::try_from(bytes.len()).map_err(io::Error::other)? > MAX_RESOURCE_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "resource grew beyond its byte bound",
            ));
        }
        Ok((bytes, mime(&path)))
    }

    pub(crate) fn permits(uri: &str) -> bool {
        uri.strip_prefix(ORIGIN)
            .is_some_and(|path| path.starts_with('/'))
    }
}

fn validate_file(metadata: &fs::Metadata) -> io::Result<()> {
    if !metadata.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "resource is not a regular file",
        ));
    }
    if metadata.len() > MAX_RESOURCE_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "resource exceeds its byte bound",
        ));
    }
    Ok(())
}

fn decode_path(encoded: &str) -> io::Result<String> {
    let mut bytes = Vec::with_capacity(encoded.len());
    let mut input = encoded.bytes();
    while let Some(byte) = input.next() {
        let byte = if byte == b'%' {
            let high = input.next().and_then(hex);
            let low = input.next().and_then(hex);
            match (high, low) {
                (Some(high), Some(low)) => (high << 4) | low,
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "invalid resource percent encoding",
                    ));
                }
            }
        } else {
            byte
        };
        if byte == 0 || byte.is_ascii_control() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "resource path contains a control character",
            ));
        }
        bytes.push(byte);
    }
    String::from_utf8(bytes).map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))
}

const fn hex(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

fn mime(path: &Path) -> &'static str {
    match path.extension().and_then(|value| value.to_str()) {
        Some("html") => "text/html",
        Some("js" | "mjs") => "text/javascript",
        Some("wasm") => "application/wasm",
        Some("css") => "text/css",
        Some("svg") => "image/svg+xml",
        Some("png") => "image/png",
        Some("jpg" | "jpeg") => "image/jpeg",
        Some("json" | "map") => "application/json",
        Some("woff") => "font/woff",
        Some("woff2") => "font/woff2",
        _ => "application/octet-stream",
    }
}

#[cfg(test)]
mod tests;

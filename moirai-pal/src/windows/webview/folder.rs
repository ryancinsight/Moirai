//! A local folder served to WebView2 under a reserved `https` host name.

use std::{
    io,
    path::{Path, PathBuf},
};

/// Top-level domains reserved from the public DNS (RFC 2606 §2, RFC 6761 §6),
/// so a mapped host can never shadow a real site.
const RESERVED_TLDS: [&str; 4] = ["example", "invalid", "localhost", "test"];
/// RFC 1035 §2.3.4 bounds.
const LABEL_LIMIT: usize = 63;
const NAME_LIMIT: usize = 253;

/// A validated host-to-folder mapping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct FolderMapping {
    host: String,
    folder: PathBuf,
}

impl FolderMapping {
    pub(super) fn new(host: &str, folder: &Path) -> io::Result<Self> {
        validate_host(host)?;
        if !folder.is_absolute() {
            return Err(invalid(
                "WebView2 folder mapping requires an absolute folder",
            ));
        }
        if !std::fs::metadata(folder)?.is_dir() {
            return Err(invalid("WebView2 folder mapping target is not a directory"));
        }
        Ok(Self {
            host: host.to_owned(),
            folder: folder.to_owned(),
        })
    }

    pub(super) fn host(&self) -> &str {
        &self.host
    }

    pub(super) fn folder(&self) -> &Path {
        &self.folder
    }
}

/// Lowercase LDH labels (RFC 1123 §2.1) ending in a reserved TLD.
fn validate_host(host: &str) -> io::Result<()> {
    if host.is_empty() || host.len() > NAME_LIMIT {
        return Err(invalid("WebView2 mapped host name length is out of bounds"));
    }
    let labels: Vec<&str> = host.split('.').collect();
    let valid_labels = labels.iter().all(|label| {
        !label.is_empty()
            && label.len() <= LABEL_LIMIT
            && !label.starts_with('-')
            && !label.ends_with('-')
            && label
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
    });
    if !valid_labels {
        return Err(invalid(
            "WebView2 mapped host must be lowercase letters, digits and inner hyphens",
        ));
    }
    match labels.as_slice() {
        [_, .., tld] if RESERVED_TLDS.contains(tld) => Ok(()),
        _ => Err(invalid(
            "WebView2 mapped host must be a subdomain of .example, .invalid, .localhost or .test",
        )),
    }
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

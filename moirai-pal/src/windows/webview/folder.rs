//! A local folder served to WebView2 under a reserved `https` host name.

use std::{
    ffi::OsString,
    io,
    path::{Component, Path, PathBuf, Prefix},
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
            folder: plain(folder)?,
        })
    }

    pub(super) fn host(&self) -> &str {
        &self.host
    }

    pub(super) fn folder(&self) -> &Path {
        &self.folder
    }
}

/// The non-verbatim spelling of an absolute folder.
///
/// WebView2 appends each request path to the mapped folder with `/`
/// separators, and a verbatim (`\\?\`) path is passed through without
/// normalization, so only files at the top of a canonicalized folder would
/// resolve. A verbatim disk or UNC path becomes its plain equivalent; a
/// verbatim path with no plain form (a volume GUID or device namespace) is
/// refused, as is a `.` or `..` segment, which the verbatim form does not
/// interpret but the plain form would.
fn plain(folder: &Path) -> io::Result<PathBuf> {
    let mut components = folder.components();
    let Some(Component::Prefix(prefix)) = components.next() else {
        return Err(invalid(
            "WebView2 folder mapping requires a drive or UNC folder",
        ));
    };
    let mut plain = match prefix.kind() {
        Prefix::Disk(_) | Prefix::UNC(..) => return Ok(folder.to_owned()),
        Prefix::VerbatimDisk(letter) => PathBuf::from(format!("{}:\\", char::from(letter))),
        Prefix::VerbatimUNC(server, share) => {
            let mut root = OsString::from(r"\\");
            root.push(server);
            root.push(r"\");
            root.push(share);
            root.push(r"\");
            PathBuf::from(root)
        }
        Prefix::Verbatim(_) | Prefix::DeviceNS(_) => {
            return Err(invalid(
                "WebView2 folder mapping requires a drive or UNC folder",
            ));
        }
    };
    for component in components {
        match component {
            Component::RootDir => {}
            Component::Normal(part) if part != "." && part != ".." => plain.push(part),
            _ => {
                return Err(invalid(
                    "WebView2 folder mapping path contains a relative segment",
                ));
            }
        }
    }
    Ok(plain)
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

#[cfg(test)]
mod tests {
    use super::plain;
    use std::path::{Path, PathBuf};

    #[test]
    fn verbatim_folders_map_to_their_plain_spelling() {
        for (verbatim, expected) in [
            (r"\\?\C:\apps\dist\app", r"C:\apps\dist\app"),
            (r"\\?\d:\x", r"d:\x"),
            (r"\\?\UNC\server\share\dist", r"\\server\share\dist"),
            (r"C:\already\plain", r"C:\already\plain"),
            (r"\\server\share\plain", r"\\server\share\plain"),
        ] {
            assert_eq!(
                plain(Path::new(verbatim)).expect("representable folder"),
                PathBuf::from(expected),
                "{verbatim}"
            );
        }
        for refused in [
            r"\\?\Volume{0b1f5c3e-0000-0000-0000-100000000000}\dist",
            r"\\.\PhysicalDrive0",
            r"\\?\C:\apps\..\secret",
            r"relative\dist",
        ] {
            assert!(plain(Path::new(refused)).is_err(), "accepted {refused}");
        }
    }
}

//! Host-owned external destinations and the packaged navigation contract.
//!
//! External authority is an exact destination allowlist, not a prefix or
//! origin grant. The full URI, including its query and fragment, remains
//! unchanged. Link activation identifies a navigation kind; it does not
//! prove a trusted user gesture. External handoff therefore requires a
//! new-window request, where WebKit's automatic popup prohibition also
//! applies, and never replaces an existing packaged frame.

#[cfg(any(target_os = "macos", target_os = "ios", test))]
use crate::ResourceRoot;
use std::io;

/// Maximum host-configured external destinations in one application.
///
/// This allocation bound matches Metis's existing 64-origin opener budget.
pub const MAX_EXTERNAL_LINKS: usize = 64;

/// Maximum UTF-8 bytes in one host-configured external destination.
///
/// This admission bound matches Metis's existing 2048-byte opener budget.
pub const MAX_EXTERNAL_LINK_BYTES: usize = 2048;

/// Exact HTTPS destinations authorized by the native application host.
///
/// An empty policy denies all external handoffs. Destinations are retained
/// verbatim and matched byte for byte: admitting one path does not admit
/// another path, fragment, query, port, spelling or origin.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ExternalLinks {
    destinations: Box<[Box<str>]>,
}

impl ExternalLinks {
    /// Validates and owns a bounded set of exact HTTPS destinations.
    ///
    /// # Errors
    /// Returns `InvalidInput` if the iterator contains more than
    /// [`MAX_EXTERNAL_LINKS`] entries, or a destination exceeds
    /// [`MAX_EXTERNAL_LINK_BYTES`], lacks an absolute HTTPS authority/host,
    /// contains credentials, contains characters outside the RFC 3986 URI
    /// alphabet, has incomplete percent escapes, or has an invalid port.
    pub fn new<I, U>(destinations: I) -> io::Result<Self>
    where
        I: IntoIterator<Item = U>,
        U: AsRef<str>,
    {
        let mut owned: Vec<Box<str>> = Vec::new();
        for (index, destination) in destinations.into_iter().enumerate() {
            if index == MAX_EXTERNAL_LINKS {
                return Err(invalid("external destination count exceeds the host bound"));
            }
            let destination = destination.as_ref();
            validate(destination)?;
            if !owned.iter().any(|value| &**value == destination) {
                owned.push(Box::<str>::from(destination));
            }
        }
        Ok(Self {
            destinations: owned.into_boxed_slice(),
        })
    }

    #[cfg(any(target_os = "macos", target_os = "ios", test))]
    pub(crate) fn decide(
        &self,
        destination: &str,
        source: Option<&str>,
        main_source: bool,
        link_activated: bool,
        download: bool,
        target: Frame,
    ) -> Decision {
        if ResourceRoot::permits(destination) && target != Frame::NewWindow {
            return Decision::Allow;
        }
        if target == Frame::NewWindow
            && main_source
            && link_activated
            && !download
            && source.is_some_and(ResourceRoot::permits)
            && self
                .destinations
                .iter()
                .any(|value| &**value == destination)
        {
            Decision::ExternalPopup
        } else {
            Decision::Deny
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(any(target_os = "macos", target_os = "ios", test))]
pub(crate) enum Frame {
    Main,
    Subframe,
    NewWindow,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg(any(target_os = "macos", target_os = "ios", test))]
pub(crate) enum Decision {
    Allow,
    ExternalPopup,
    Deny,
}

fn validate(destination: &str) -> io::Result<()> {
    if destination.is_empty()
        || destination.len() > MAX_EXTERNAL_LINK_BYTES
        || !destination.bytes().all(is_uri_byte)
    {
        return Err(invalid(
            "external destination violates the URI byte contract",
        ));
    }
    for (index, byte) in destination.bytes().enumerate() {
        if byte == b'%'
            && !destination
                .as_bytes()
                .get(index + 1..index + 3)
                .is_some_and(|pair| pair.iter().all(u8::is_ascii_hexdigit))
        {
            return Err(invalid("external destination has invalid percent encoding"));
        }
    }
    // http::Uri represents a request target and discards its fragment. Parse
    // only the transport part, validate the full URI alphabet above, and keep
    // the original string as the sole identity used by the allowlist.
    let transport = if let Some((transport, fragment)) = destination.split_once('#') {
        if fragment.contains('#') {
            return Err(invalid("external destination has an invalid fragment"));
        }
        transport
    } else {
        destination
    };
    let uri = transport
        .parse::<http::Uri>()
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
    if uri.scheme_str() != Some("https") {
        return Err(invalid("external destination must use absolute HTTPS"));
    }
    let authority = uri
        .authority()
        .ok_or_else(|| invalid("external destination has no authority"))?;
    if authority.as_str().contains('@') {
        return Err(invalid("external destination contains credentials"));
    }
    let host = uri
        .host()
        .filter(|host| !host.is_empty())
        .ok_or_else(|| invalid("external destination has no host"))?;
    let suffix = authority
        .as_str()
        .strip_prefix(host)
        .ok_or_else(|| invalid("external destination has an invalid authority"))?;
    if !suffix.is_empty()
        && !suffix.strip_prefix(':').is_some_and(|port| {
            !port.is_empty()
                && port.bytes().all(|byte| byte.is_ascii_digit())
                && authority.port_u16().is_some()
        })
    {
        return Err(invalid("external destination has an invalid port"));
    }
    Ok(())
}

const fn is_uri_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric()
        || matches!(
            byte,
            b'-' | b'.'
                | b'_'
                | b'~'
                | b':'
                | b'/'
                | b'?'
                | b'#'
                | b'['
                | b']'
                | b'@'
                | b'!'
                | b'$'
                | b'&'
                | b'\''
                | b'('
                | b')'
                | b'*'
                | b'+'
                | b','
                | b';'
                | b'='
                | b'%'
        )
}

fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

#[cfg(test)]
mod tests;

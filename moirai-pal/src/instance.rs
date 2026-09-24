//! Single-instance coordination between processes of one application.
//!
//! The first process to [`claim`] an [`InstanceName`] becomes the
//! [`PrimaryInstance`]; a later process becomes a [`SecondaryInstance`] and
//! forwards one bounded message, typically its command-line arguments or the
//! deep link it was started with, before exiting. This is the service of
//! Tauri's single-instance plugin.
//!
//! On Unix the primary holds an advisory lock and listens on a socket, both
//! in a directory only the current user can open. On Windows it owns the
//! first instance of a named pipe scoped to the login session that refuses
//! remote clients. The operating system releases either claim when the
//! primary exits, however it exits, so a crash never leaves the name taken.

use std::io::{self, Read, Write};
use std::time::Duration;

#[cfg(unix)]
mod unix;
#[cfg(unix)]
use unix as native;

#[cfg(windows)]
mod windows;
#[cfg(windows)]
use windows as native;

/// Maximum bytes in an instance name.
pub const MAX_INSTANCE_NAME_BYTES: usize = 64;
/// Maximum bytes one secondary may forward.
pub const MAX_INSTANCE_MESSAGE_BYTES: usize = 16 * 1024;
/// Longest a sender or receiver waits on the other side of one message.
pub const INSTANCE_IO_TIMEOUT: Duration = Duration::from_secs(2);

/// A validated application instance name.
///
/// Names are 1 to [`MAX_INSTANCE_NAME_BYTES`] bytes of lowercase ASCII
/// letters, digits, `.` and `-`, starting with a letter or digit, such as a
/// reverse-DNS application identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct InstanceName(String);

impl InstanceName {
    /// Validates an instance name.
    ///
    /// # Errors
    /// Returns `InvalidInput` for an empty, over-long or non-conforming name.
    pub fn new(name: &str) -> io::Result<Self> {
        let bytes = name.as_bytes();
        let valid = !bytes.is_empty()
            && bytes.len() <= MAX_INSTANCE_NAME_BYTES
            && bytes[0].is_ascii_alphanumeric()
            && bytes.iter().all(|byte| {
                byte.is_ascii_lowercase() || byte.is_ascii_digit() || matches!(byte, b'.' | b'-')
            });
        if !valid {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "instance name must be 1 to 64 lowercase letters, digits, '.' or '-'",
            ));
        }
        Ok(Self(name.to_owned()))
    }

    /// The name.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// The outcome of [`claim`].
#[derive(Debug)]
pub enum InstanceRole {
    /// This process holds the name and receives forwarded messages.
    Primary(PrimaryInstance),
    /// Another process holds the name; forward a message to it.
    Secondary(SecondaryInstance),
}

/// The process that holds an instance name.
#[derive(Debug)]
pub struct PrimaryInstance {
    inner: native::Primary,
}

impl PrimaryInstance {
    /// Receives the next forwarded message without blocking when none is
    /// waiting.
    ///
    /// A secondary that connects but does not deliver a complete message
    /// within [`INSTANCE_IO_TIMEOUT`] is dropped with an error, and the next
    /// call serves the next secondary.
    ///
    /// # Errors
    /// Returns the I/O error, `TimedOut` for a stalled sender, or
    /// `InvalidData` for a message over [`MAX_INSTANCE_MESSAGE_BYTES`].
    pub fn try_receive(&mut self) -> io::Result<Option<Vec<u8>>> {
        self.inner.try_receive()
    }
}

/// A process that found the instance name held by another process.
#[derive(Debug)]
pub struct SecondaryInstance {
    inner: native::Secondary,
}

impl SecondaryInstance {
    /// Forwards one message to the primary.
    ///
    /// # Errors
    /// Returns `InvalidInput` for a message over
    /// [`MAX_INSTANCE_MESSAGE_BYTES`], or the I/O error, including `TimedOut`
    /// when the primary does not accept the message in time.
    pub fn send(self, message: &[u8]) -> io::Result<()> {
        if message.len() > MAX_INSTANCE_MESSAGE_BYTES {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "instance message exceeds its bound",
            ));
        }
        self.inner.send(message)
    }
}

/// Claims `name` for this process, or connects to the process holding it.
///
/// # Errors
/// Returns the I/O error when the claim cannot be made or checked, for
/// example when the per-user directory is unsafe or the holder does not
/// accept a connection within [`INSTANCE_IO_TIMEOUT`].
pub fn claim(name: &InstanceName) -> io::Result<InstanceRole> {
    native::claim(name).map(|role| match role {
        native::Role::Primary(inner) => InstanceRole::Primary(PrimaryInstance { inner }),
        native::Role::Secondary(inner) => InstanceRole::Secondary(SecondaryInstance { inner }),
    })
}

/// Writes one length-prefixed message.
fn write_frame(writer: &mut impl Write, message: &[u8]) -> io::Result<()> {
    let length = u32::try_from(message.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "instance message too large"))?;
    writer.write_all(&length.to_le_bytes())?;
    writer.write_all(message)?;
    writer.flush()
}

/// Reads one length-prefixed message of at most the message bound.
fn read_frame(reader: &mut impl Read) -> io::Result<Vec<u8>> {
    let mut header = [0; 4];
    reader.read_exact(&mut header)?;
    let length = usize::try_from(u32::from_le_bytes(header)).unwrap_or(usize::MAX);
    if length > MAX_INSTANCE_MESSAGE_BYTES {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "instance message exceeds its bound",
        ));
    }
    let mut message = Vec::new();
    message
        .try_reserve_exact(length)
        .map_err(|_| io::Error::new(io::ErrorKind::OutOfMemory, "instance message allocation"))?;
    message.resize(length, 0);
    reader.read_exact(&mut message)?;
    Ok(message)
}

#[cfg(test)]
mod tests;

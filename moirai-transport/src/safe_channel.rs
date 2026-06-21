//! Zero-copy archive channel helpers for transport boundaries.
//!
//! The transport owns message bytes. Receivers validate those bytes and expose
//! typed archived views that borrow from the received buffer instead of
//! reconstructing owned values. This follows the same architectural rule as
//! rkyv-style archives: validate once, then read through a borrowed view.

use crate::{Address, TransportError, TransportManager, TransportResult};
use std::{marker::PhantomData, str, sync::Arc};

/// Writes a value into transport-owned archive bytes.
pub trait ArchiveSerialize: Send + 'static {
    /// Exact archive byte length when the representation is statically known
    /// from the value.
    fn archive_size_hint(&self) -> usize {
        0
    }

    /// Append the archived representation to `output`.
    fn encode_archive(&self, output: &mut Vec<u8>) -> TransportResult<()>;

    /// Create an owned byte buffer suitable for transport.
    fn archive_bytes(&self) -> TransportResult<Vec<u8>> {
        let mut output = Vec::with_capacity(self.archive_size_hint());
        self.encode_archive(&mut output)?;
        Ok(output)
    }
}

/// Validates archive bytes and returns a typed borrowed view.
pub trait ArchiveView: Send + 'static {
    /// Borrowed representation backed by the archive byte buffer.
    type Archived<'a>
    where
        Self: 'a;

    /// Validate and view an archived value without allocating an owned value.
    fn view_archive(bytes: &[u8]) -> TransportResult<Self::Archived<'_>>;
}

/// Transport message bytes plus a typed archive view contract.
pub struct ArchivedMessage<T: ArchiveView> {
    bytes: Vec<u8>,
    _phantom: PhantomData<T>,
}

impl<T: ArchiveView> ArchivedMessage<T> {
    /// Create a message from validated transport bytes.
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            _phantom: PhantomData,
        }
    }

    /// Borrow the typed archived value from the owned message bytes.
    pub fn get(&self) -> TransportResult<T::Archived<'_>> {
        T::view_archive(&self.bytes)
    }

    /// Raw archived bytes for diagnostics or forwarding.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }
}

/// Universal sender for archive-serializable values.
pub struct ArchivedUniversalSender<T: ArchiveSerialize + ?Sized> {
    transport: Arc<TransportManager>,
    target: Address,
    _phantom: PhantomData<T>,
}

impl<T: ArchiveSerialize + ?Sized> ArchivedUniversalSender<T> {
    /// Create a new archived sender.
    pub fn new(transport: Arc<TransportManager>, target: Address) -> Self {
        Self {
            transport,
            target,
            _phantom: PhantomData,
        }
    }

    /// Archive and send a value.
    pub fn send(&self, value: &T) -> TransportResult<()> {
        self.transport.send(&self.target, value.archive_bytes()?)
    }
}

/// Universal receiver for zero-copy archive views.
pub struct ArchivedUniversalReceiver<T: ArchiveView> {
    transport: Arc<TransportManager>,
    source: Address,
    _phantom: PhantomData<T>,
}

impl<T: ArchiveView> ArchivedUniversalReceiver<T> {
    /// Create a new archived receiver.
    pub fn new(transport: Arc<TransportManager>, source: Address) -> Self {
        Self {
            transport,
            source,
            _phantom: PhantomData,
        }
    }

    /// Receive bytes and keep them alive for borrowed archived views.
    pub fn recv(&self) -> TransportResult<ArchivedMessage<T>> {
        self.transport
            .recv(&self.source)
            .map(ArchivedMessage::from_bytes)
    }
}

impl ArchiveSerialize for i32 {
    fn archive_size_hint(&self) -> usize {
        core::mem::size_of::<Self>()
    }

    fn encode_archive(&self, output: &mut Vec<u8>) -> TransportResult<()> {
        output.extend_from_slice(&self.to_le_bytes());
        Ok(())
    }
}

impl ArchiveView for i32 {
    type Archived<'a> = i32;

    fn view_archive(bytes: &[u8]) -> TransportResult<Self::Archived<'_>> {
        let array: [u8; 4] = bytes.try_into().map_err(|_| TransportError::Closed)?;
        Ok(i32::from_le_bytes(array))
    }
}

impl ArchiveSerialize for String {
    fn archive_size_hint(&self) -> usize {
        self.as_str().archive_size_hint()
    }

    fn encode_archive(&self, output: &mut Vec<u8>) -> TransportResult<()> {
        self.as_str().encode_archive(output)
    }
}

impl ArchiveSerialize for str {
    fn archive_size_hint(&self) -> usize {
        core::mem::size_of::<u32>() + self.len()
    }

    fn encode_archive(&self, output: &mut Vec<u8>) -> TransportResult<()> {
        let bytes = self.as_bytes();
        let len = u32::try_from(bytes.len()).map_err(|_| TransportError::Closed)?;

        output.extend_from_slice(&len.to_le_bytes());
        output.extend_from_slice(bytes);
        Ok(())
    }
}

impl ArchiveView for String {
    type Archived<'a> = &'a str;

    fn view_archive(bytes: &[u8]) -> TransportResult<Self::Archived<'_>> {
        if bytes.len() < 4 {
            return Err(TransportError::Closed);
        }

        let len_bytes: [u8; 4] = bytes[0..4].try_into().map_err(|_| TransportError::Closed)?;
        let len = u32::from_le_bytes(len_bytes) as usize;
        let end = len.checked_add(4).ok_or(TransportError::Closed)?;
        let payload = bytes.get(4..end).ok_or(TransportError::Closed)?;
        if bytes.len() != end {
            return Err(TransportError::Closed);
        }

        str::from_utf8(payload).map_err(|_| TransportError::Closed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn archive_views_validate_value_semantics() {
        let value: i32 = 42;
        let archived = value.archive_bytes().unwrap();
        assert_eq!(archived.capacity(), core::mem::size_of::<i32>());
        let message = ArchivedMessage::<i32>::from_bytes(archived);

        assert_eq!(message.get().unwrap(), value);

        let value = String::from("Hello, Moirai!");
        let archived = value.archive_bytes().unwrap();
        assert_eq!(
            archived.capacity(),
            core::mem::size_of::<u32>() + value.len()
        );
        let message = ArchivedMessage::<String>::from_bytes(archived);

        assert_eq!(message.get().unwrap(), value.as_str());
    }

    #[test]
    fn string_archive_view_borrows_from_message_buffer() {
        let value = String::from("borrowed archive view");
        let message = ArchivedMessage::<String>::from_bytes(value.archive_bytes().unwrap());
        let view = message.get().unwrap();

        let buffer_start = message.as_bytes().as_ptr() as usize;
        let buffer_end = buffer_start + message.as_bytes().len();
        let view_start = view.as_ptr() as usize;
        let view_end = view_start + view.len();

        assert_eq!(view, value);
        assert!(view_start >= buffer_start);
        assert!(view_end <= buffer_end);
    }

    #[test]
    fn archived_channel_returns_borrowed_view() {
        let transport = Arc::new(TransportManager::new());
        let address = Address::Local("archive-test".to_string());
        let sender =
            ArchivedUniversalSender::<String>::new(Arc::clone(&transport), address.clone());
        let receiver = ArchivedUniversalReceiver::<String>::new(transport, address);
        let value = String::from("zero copy receive");

        sender.send(&value).unwrap();
        let message = receiver.recv().unwrap();

        assert_eq!(message.get().unwrap(), value.as_str());
    }

    #[test]
    fn archive_views_reject_malformed_bytes() {
        assert!(i32::view_archive(&[1, 2, 3]).is_err());
        assert!(String::view_archive(&[1, 0, 0]).is_err());

        let declared_length_exceeds_payload = [4, 0, 0, 0, b'a', b'b'];
        assert!(String::view_archive(&declared_length_exceeds_payload).is_err());

        let trailing_bytes_after_payload = [1, 0, 0, 0, b'a', b'b'];
        assert!(String::view_archive(&trailing_bytes_after_payload).is_err());

        let invalid_utf8 = [1, 0, 0, 0, 0xff];
        assert!(String::view_archive(&invalid_utf8).is_err());
    }
}

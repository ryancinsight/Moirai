//! Typed ownership regions for transport archive payload bytes.

use crate::{safe_channel::ArchiveSerialize, TransportResult};
use core::marker::PhantomData;

mod sealed {
    pub trait Sealed {}
}

/// Transport boundary that owns an archived payload buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PayloadBoundary {
    /// Payload remains inside the current process.
    Thread,
    /// Payload is handed to a child process route.
    Process,
    /// Payload is handed to a server route.
    Server,
}

/// Sealed payload ownership region.
pub trait PayloadRegion: sealed::Sealed + Copy + Default + Send + Sync + 'static {
    /// Boundary represented by this region.
    const BOUNDARY: PayloadBoundary;

    /// Whether raw pointer identity may be reused across this boundary.
    const POINTER_TRANSFER_ALLOWED: bool;
}

/// Current-process payload region.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct ThreadPayloadRegion;

impl sealed::Sealed for ThreadPayloadRegion {}

impl PayloadRegion for ThreadPayloadRegion {
    const BOUNDARY: PayloadBoundary = PayloadBoundary::Thread;
    const POINTER_TRANSFER_ALLOWED: bool = true;
}

/// Child-process payload region.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct ProcessPayloadRegion;

impl sealed::Sealed for ProcessPayloadRegion {}

impl PayloadRegion for ProcessPayloadRegion {
    const BOUNDARY: PayloadBoundary = PayloadBoundary::Process;
    const POINTER_TRANSFER_ALLOWED: bool = false;
}

/// Server payload region.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct ServerPayloadRegion;

impl sealed::Sealed for ServerPayloadRegion {}

impl PayloadRegion for ServerPayloadRegion {
    const BOUNDARY: PayloadBoundary = PayloadBoundary::Server;
    const POINTER_TRANSFER_ALLOWED: bool = false;
}

/// Owned archived payload bytes tagged with an allocation ownership region.
#[repr(transparent)]
#[derive(Debug, PartialEq, Eq)]
pub struct TransportPayload<R: PayloadRegion> {
    bytes: Vec<u8>,
    _region: PhantomData<R>,
}

impl<R: PayloadRegion> TransportPayload<R> {
    /// Construct a region-tagged payload from owned archive bytes.
    pub fn from_bytes(bytes: Vec<u8>) -> Self {
        Self {
            bytes,
            _region: PhantomData,
        }
    }

    /// Consume the payload and return owned archive bytes.
    pub fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }

    /// Borrow the archived payload bytes.
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Return the payload length in bytes.
    pub fn len(&self) -> usize {
        self.bytes.len()
    }

    /// Return whether the payload is empty.
    pub fn is_empty(&self) -> bool {
        self.bytes.is_empty()
    }

    /// Return the boundary represented by this payload region.
    pub const fn boundary() -> PayloadBoundary {
        R::BOUNDARY
    }

    /// Return whether raw pointer transfer is valid for this region.
    pub const fn pointer_transfer_allowed() -> bool {
        R::POINTER_TRANSFER_ALLOWED
    }

    /// Move the same owned bytes into another region marker.
    pub fn handoff<Target: PayloadRegion>(self) -> TransportPayload<Target> {
        TransportPayload {
            bytes: self.bytes,
            _region: PhantomData,
        }
    }
}

/// Archive a value into typed transport-owned payload bytes.
pub fn archive_transport_payload<R, T>(value: &T) -> TransportResult<TransportPayload<R>>
where
    R: PayloadRegion,
    T: ArchiveSerialize + ?Sized,
{
    value.archive_bytes().map(TransportPayload::from_bytes)
}

#[cfg(test)]
mod tests {
    use super::{
        archive_transport_payload, PayloadBoundary, PayloadRegion, ProcessPayloadRegion,
        ServerPayloadRegion, ThreadPayloadRegion, TransportPayload,
    };
    use core::mem::size_of;

    const _: () = {
        assert!(ThreadPayloadRegion::POINTER_TRANSFER_ALLOWED);
        assert!(!ProcessPayloadRegion::POINTER_TRANSFER_ALLOWED);
        assert!(!ServerPayloadRegion::POINTER_TRANSFER_ALLOWED);
    };

    #[test]
    fn payload_region_markers_are_zero_sized() {
        assert_eq!(size_of::<ThreadPayloadRegion>(), 0);
        assert_eq!(size_of::<ProcessPayloadRegion>(), 0);
        assert_eq!(size_of::<ServerPayloadRegion>(), 0);
    }

    #[test]
    fn payload_regions_encode_pointer_transfer_contract() {
        assert_eq!(ThreadPayloadRegion::BOUNDARY, PayloadBoundary::Thread);
        assert_eq!(ProcessPayloadRegion::BOUNDARY, PayloadBoundary::Process);
        assert_eq!(ServerPayloadRegion::BOUNDARY, PayloadBoundary::Server);
    }

    #[test]
    fn payload_handoff_moves_same_owned_buffer_between_regions() {
        let bytes = b"mnemosyne-owned archive payload".to_vec();
        let ptr = bytes.as_ptr();
        let thread_payload = TransportPayload::<ThreadPayloadRegion>::from_bytes(bytes);
        let process_payload = thread_payload.handoff::<ProcessPayloadRegion>();

        assert_eq!(
            process_payload.as_bytes(),
            b"mnemosyne-owned archive payload"
        );
        assert_eq!(process_payload.as_bytes().as_ptr(), ptr);
    }

    #[test]
    fn archive_transport_payload_preserves_value_bytes() {
        let payload =
            archive_transport_payload::<ServerPayloadRegion, _>(&"route payload".to_string())
                .unwrap();

        assert_eq!(payload.len(), 17);
        assert_eq!(payload.as_bytes()[..4], 13u32.to_le_bytes());
        assert_eq!(&payload.as_bytes()[4..], b"route payload");
    }
}

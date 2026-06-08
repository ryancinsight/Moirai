//! Static capability tokens for fixed-format remote task construction.

use super::RemoteTaskOperation;
use core::marker::PhantomData;

mod sealed {
    pub trait Sealed {}
}

/// Built-in remote operation kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RemoteTaskOperationKind {
    /// Echo a byte payload.
    EchoBytes,
    /// Sum `u64` values with wrapping arithmetic.
    SumU64,
}

/// Sealed capability for a fixed-format remote operation.
pub trait RemoteCapability: sealed::Sealed + Copy + Default + Send + Sync + 'static {
    /// Operation kind admitted by this capability.
    const OPERATION_KIND: RemoteTaskOperationKind;
}

/// Capability admitting byte echo tasks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct EchoBytesCapability;

impl sealed::Sealed for EchoBytesCapability {}

impl RemoteCapability for EchoBytesCapability {
    const OPERATION_KIND: RemoteTaskOperationKind = RemoteTaskOperationKind::EchoBytes;
}

/// Capability admitting wrapping `u64` sum tasks.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct SumU64Capability;

impl sealed::Sealed for SumU64Capability {}

impl RemoteCapability for SumU64Capability {
    const OPERATION_KIND: RemoteTaskOperationKind = RemoteTaskOperationKind::SumU64;
}

/// Zero-sized token proving that a caller selected an admitted remote capability.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub struct RemoteCapabilityToken<C: RemoteCapability> {
    _capability: PhantomData<C>,
}

impl<C: RemoteCapability> RemoteCapabilityToken<C> {
    /// Construct a zero-sized capability token.
    pub const fn new() -> Self {
        Self {
            _capability: PhantomData,
        }
    }

    /// Return the operation kind selected by the capability type.
    pub const fn operation_kind(self) -> RemoteTaskOperationKind {
        C::OPERATION_KIND
    }
}

/// Convert an admitted payload into a fixed-format remote operation.
pub trait IntoRemoteOperation<C: RemoteCapability> {
    /// Build the owned operation represented by the capability token.
    fn into_remote_operation(self, token: RemoteCapabilityToken<C>) -> RemoteTaskOperation;
}

impl IntoRemoteOperation<EchoBytesCapability> for Vec<u8> {
    fn into_remote_operation(
        self,
        _token: RemoteCapabilityToken<EchoBytesCapability>,
    ) -> RemoteTaskOperation {
        RemoteTaskOperation::EchoBytes(self)
    }
}

impl IntoRemoteOperation<SumU64Capability> for Vec<u64> {
    fn into_remote_operation(
        self,
        _token: RemoteCapabilityToken<SumU64Capability>,
    ) -> RemoteTaskOperation {
        RemoteTaskOperation::SumU64(self)
    }
}

/// Build a fixed-format remote operation from an admitted capability and payload.
pub fn build_remote_operation<C, P>(
    payload: P,
    token: RemoteCapabilityToken<C>,
) -> RemoteTaskOperation
where
    C: RemoteCapability,
    P: IntoRemoteOperation<C>,
{
    payload.into_remote_operation(token)
}

#[cfg(test)]
mod tests {
    use super::{
        build_remote_operation, EchoBytesCapability, RemoteCapabilityToken,
        RemoteTaskOperationKind, SumU64Capability,
    };
    use crate::remote_task::RemoteTaskOperation;
    use core::mem::size_of;

    #[test]
    fn remote_capability_tokens_are_zero_sized() {
        assert_eq!(size_of::<RemoteCapabilityToken<EchoBytesCapability>>(), 0);
        assert_eq!(size_of::<RemoteCapabilityToken<SumU64Capability>>(), 0);
    }

    #[test]
    fn remote_capability_tokens_report_operation_kind() {
        assert_eq!(
            RemoteCapabilityToken::<EchoBytesCapability>::new().operation_kind(),
            RemoteTaskOperationKind::EchoBytes
        );
        assert_eq!(
            RemoteCapabilityToken::<SumU64Capability>::new().operation_kind(),
            RemoteTaskOperationKind::SumU64
        );
    }

    #[test]
    fn remote_capabilities_build_only_fixed_format_operations() {
        let echo = build_remote_operation(
            b"capability bytes".to_vec(),
            RemoteCapabilityToken::<EchoBytesCapability>::new(),
        );
        let sum = build_remote_operation(
            vec![1u64, 2, u64::MAX],
            RemoteCapabilityToken::<SumU64Capability>::new(),
        );

        assert_eq!(
            echo,
            RemoteTaskOperation::EchoBytes(b"capability bytes".to_vec())
        );
        assert_eq!(sum, RemoteTaskOperation::SumU64(vec![1, 2, u64::MAX]));
    }
}

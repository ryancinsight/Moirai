//! Work class markers for compile-time scheduler routing.

mod sealed {
    pub trait Sealed {}
}

/// Compile-time routing contract for a scheduler work class.
///
/// Implementors are zero-sized marker types. The scheduler monomorphizes
/// dispatch per marker and uses the associated constants for deterministic
/// worker selection without storing a runtime policy object.
pub trait WorkClass: sealed::Sealed + Send + Sync + 'static {
    /// Offset mixed into round-robin worker selection for this work class.
    const AFFINITY_OFFSET: usize;

    /// Worker offset used for serial handoff when no queued backlog exists.
    ///
    /// Serial result-bearing submit/join loops benefit from a single stable
    /// worker cache footprint. The value is an associated constant so the
    /// scheduler monomorphizes the routing decision per work class.
    const SERIAL_AFFINITY_OFFSET: usize;

    /// Stable diagnostic name for this work class.
    const NAME: &'static str;

    /// Whether this work class is eligible for asynchronous lane routing.
    const USES_ASYNC_LANE: bool;

    /// Whether this work class uses the dedicated bounded blocking lane.
    const USES_BLOCKING_LANE: bool;
}

/// CPU-bound synchronous task marker.
#[derive(Debug, Clone, Copy, Default)]
pub struct SyncTask;

/// Future-driving task marker.
#[derive(Debug, Clone, Copy, Default)]
pub struct AsyncTask;

/// Potentially blocking task marker.
#[derive(Debug, Clone, Copy, Default)]
pub struct BlockingTask;

impl sealed::Sealed for SyncTask {}
impl sealed::Sealed for AsyncTask {}
impl sealed::Sealed for BlockingTask {}

impl WorkClass for SyncTask {
    const AFFINITY_OFFSET: usize = 0;
    const SERIAL_AFFINITY_OFFSET: usize = 0;
    const NAME: &'static str = "sync";
    const USES_ASYNC_LANE: bool = false;
    const USES_BLOCKING_LANE: bool = false;
}

impl WorkClass for AsyncTask {
    const AFFINITY_OFFSET: usize = 1;
    const SERIAL_AFFINITY_OFFSET: usize = Self::AFFINITY_OFFSET;
    const NAME: &'static str = "async";
    const USES_ASYNC_LANE: bool = true;
    const USES_BLOCKING_LANE: bool = false;
}

impl WorkClass for BlockingTask {
    const AFFINITY_OFFSET: usize = 2;
    const SERIAL_AFFINITY_OFFSET: usize = Self::AFFINITY_OFFSET;
    const NAME: &'static str = "blocking";
    const USES_ASYNC_LANE: bool = false;
    const USES_BLOCKING_LANE: bool = true;
}

#[cfg(test)]
mod tests {
    use super::{AsyncTask, BlockingTask, SyncTask};

    #[test]
    fn work_class_markers_are_zero_sized() {
        assert_eq!(core::mem::size_of::<SyncTask>(), 0);
        assert_eq!(core::mem::size_of::<AsyncTask>(), 0);
        assert_eq!(core::mem::size_of::<BlockingTask>(), 0);
    }
}

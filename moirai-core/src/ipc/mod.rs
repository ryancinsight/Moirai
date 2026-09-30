//! Same-machine inter-process communication over shared memory.

#[cfg(unix)]
mod backing_store;
mod error;
mod memory;
mod queue;

#[cfg(test)]
mod tests;

pub use error::IpcError;
pub use memory::SharedMemory;
pub use queue::{SendError, SharedQueue};

/// Exercise the pure shared-queue layout arithmetic.
///
/// This entry point exists only in cargo-fuzz builds; production builds do not
/// expose a test-only API surface.
#[cfg(fuzzing)]
#[doc(hidden)]
pub fn __fuzz_ipc_layout(elem_size: usize, capacity: usize) -> Result<usize, IpcError> {
    queue::layout_total(queue::QUEUE_META_SIZE, elem_size, 1, capacity)
}

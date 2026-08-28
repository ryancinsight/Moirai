//! Same-machine inter-process communication over shared memory.

mod error;
mod memory;
mod queue;

#[cfg(test)]
mod tests;

pub use error::IpcError;
pub use memory::SharedMemory;
pub use queue::SharedQueue;

/// Exercise the pure shared-queue header and layout validation boundaries.
///
/// This entry point exists only in cargo-fuzz builds; production builds do not
/// expose a test-only API surface.
#[cfg(fuzzing)]
#[doc(hidden)]
pub fn __fuzz_ipc_header(
    bytes: &[u8],
    elem_size: usize,
    capacity: usize,
) -> (Result<usize, IpcError>, Result<usize, IpcError>) {
    (
        queue::parse_header_capacity(bytes),
        queue::layout_total(queue::QUEUE_META_SIZE, elem_size, 1, capacity),
    )
}

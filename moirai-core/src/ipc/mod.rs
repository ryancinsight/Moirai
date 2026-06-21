//! Same-machine inter-process communication over shared memory.

mod error;
mod memory;
mod queue;

#[cfg(test)]
mod tests;

pub use error::IpcError;
pub use memory::SharedMemory;
pub use queue::SharedQueue;

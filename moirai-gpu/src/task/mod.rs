//! Typed GPU task contracts scheduled by Moirai.

mod configured;
mod function;

pub use configured::{ConfiguredGpuTask, GpuTaskBuilder};
pub use function::FunctionGpuTask;

use hephaestus_core::{ComputeDevice, Result};

/// A statically dispatched task executed by one Hephaestus device.
pub trait GpuTask: Send + 'static {
    /// Device type required by this task.
    type Device: ComputeDevice + Send + Sync + 'static;
    /// Value returned by the task.
    type Output: Send + 'static;

    /// Execute provider-owned GPU work.
    fn execute_gpu(self, device: &Self::Device) -> Result<Self::Output>;

    /// Estimated scheduler cost for this task.
    fn estimated_cost(&self) -> u32 {
        1
    }
}

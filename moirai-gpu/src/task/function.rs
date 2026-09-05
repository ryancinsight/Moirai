use core::marker::PhantomData;

use hephaestus_core::{ComputeDevice, Result};

use super::GpuTask;

/// A typed GPU task backed by one caller-provided function.
pub struct FunctionGpuTask<D, F, T> {
    function: F,
    _types: PhantomData<fn(&D) -> T>,
}

impl<D, F, T> FunctionGpuTask<D, F, T>
where
    D: ComputeDevice + Send + Sync + 'static,
    F: FnOnce(&D) -> Result<T> + Send + 'static,
    T: Send + 'static,
{
    /// Create a task whose device operation is performed when scheduled.
    #[must_use]
    pub fn new(function: F) -> Self {
        Self {
            function,
            _types: PhantomData,
        }
    }
}

impl<D, F, T> GpuTask for FunctionGpuTask<D, F, T>
where
    D: ComputeDevice + Send + Sync + 'static,
    F: FnOnce(&D) -> Result<T> + Send + 'static,
    T: Send + 'static,
{
    type Device = D;
    type Output = T;

    fn execute_gpu(self, device: &D) -> Result<T> {
        (self.function)(device)
    }
}

#[cfg(test)]
mod tests {
    use super::FunctionGpuTask;
    use crate::GpuTask;
    use hephaestus_core::{ComputeDevice, Result};
    use hephaestus_host::HostDevice;

    #[test]
    fn function_task_preserves_provider_and_output_types() -> Result<()> {
        let task = FunctionGpuTask::new(|device: &HostDevice| {
            let input = device.upload(&[2_u32, 3, 5, 7])?;
            let mut output = [0_u32; 4];
            device.download(&input, &mut output)?;
            Ok(output.into_iter().sum::<u32>())
        });

        assert_eq!(task.execute_gpu(&HostDevice::new())?, 17);
        Ok(())
    }
}

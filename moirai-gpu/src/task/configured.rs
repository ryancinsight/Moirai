use std::borrow::Cow;

use hephaestus_core::{ComputeDeviceCapabilities, DeviceFeature, Result};

use super::GpuTask;

/// Builder for scheduler metadata attached to a GPU task.
pub struct GpuTaskBuilder<T> {
    task: T,
    complexity: u32,
    required_features: Cow<'static, [DeviceFeature]>,
}

impl<T: GpuTask> GpuTaskBuilder<T> {
    /// Start configuring `task`.
    #[must_use]
    pub fn new(task: T) -> Self {
        Self {
            task,
            complexity: 1,
            required_features: Cow::Borrowed(&[]),
        }
    }

    /// Set the scheduler cost estimate.
    #[must_use]
    pub const fn with_complexity(mut self, complexity: u32) -> Self {
        self.complexity = complexity;
        self
    }

    /// Require provider-neutral device features.
    #[must_use]
    pub fn with_required_features(
        mut self,
        features: impl Into<Cow<'static, [DeviceFeature]>>,
    ) -> Self {
        self.required_features = features.into();
        self
    }

    /// Finish the task configuration.
    #[must_use]
    pub fn build(self) -> ConfiguredGpuTask<T> {
        ConfiguredGpuTask {
            task: self.task,
            complexity: self.complexity,
            required_features: self.required_features,
        }
    }
}

/// A GPU task with scheduler cost and device-feature metadata.
pub struct ConfiguredGpuTask<T> {
    task: T,
    complexity: u32,
    required_features: Cow<'static, [DeviceFeature]>,
}

impl<T> GpuTask for ConfiguredGpuTask<T>
where
    T: GpuTask,
    T::Device: ComputeDeviceCapabilities,
{
    type Device = T::Device;
    type Output = T::Output;

    fn execute_gpu(self, device: &Self::Device) -> Result<Self::Output> {
        for feature in self.required_features.iter().copied() {
            if !device.supports_device_feature(feature) {
                return Err(hephaestus_core::HephaestusError::InvalidConfiguration {
                    message: format!("required device feature {feature:?} is unavailable"),
                });
            }
        }
        self.task.execute_gpu(device)
    }

    fn estimated_cost(&self) -> u32 {
        self.complexity
    }
}

#[cfg(test)]
mod tests {
    use super::GpuTaskBuilder;
    use crate::{DeviceFeature, GpuTask};
    use hephaestus_core::{ComputeDevice, Result};
    use hephaestus_host::HostDevice;

    #[test]
    fn configured_task_executes_and_reports_scheduler_cost() -> Result<()> {
        let task =
            GpuTaskBuilder::new(super::super::FunctionGpuTask::new(|device: &HostDevice| {
                let input = device.upload(&[11_u32, 13])?;
                let mut output = [0_u32; 2];
                device.download(&input, &mut output)?;
                Ok(output.into_iter().product::<u32>())
            }))
            .with_complexity(7)
            .build();

        assert_eq!(task.estimated_cost(), 7);
        assert_eq!(task.execute_gpu(&HostDevice::new())?, 143);
        Ok(())
    }

    #[test]
    fn configured_task_rejects_unavailable_features_before_execution() {
        let task = GpuTaskBuilder::new(super::super::FunctionGpuTask::new(
            |_device: &HostDevice| Ok::<_, hephaestus_core::HephaestusError>(1_u32),
        ))
        .with_required_features(vec![DeviceFeature::ShaderF16])
        .build();

        let error = task
            .execute_gpu(&HostDevice::new())
            .expect_err("unsupported feature must reject the task");
        assert!(matches!(
            error,
            hephaestus_core::HephaestusError::InvalidConfiguration { .. }
        ));
    }
}

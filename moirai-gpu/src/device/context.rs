use std::sync::Arc;

use eunomia::Pod;
use hephaestus_core::{ComputeDevice, ComputeDeviceAcquisition, Result};

use super::DevicePreferences;

/// Shared handle to one acquired Hephaestus device.
pub struct GpuContext<D> {
    device: Arc<D>,
}

impl<D> Clone for GpuContext<D> {
    fn clone(&self) -> Self {
        Self {
            device: Arc::clone(&self.device),
        }
    }
}

impl<D: ComputeDevice> GpuContext<D> {
    /// Wrap an already acquired provider device.
    #[must_use]
    pub fn from_device(device: D) -> Self {
        Self {
            device: Arc::new(device),
        }
    }

    /// Borrow the provider-owned device.
    #[must_use]
    pub fn device(&self) -> &D {
        &self.device
    }

    /// Clone the shared device handle for a scheduled task.
    #[must_use]
    pub fn device_handle(&self) -> Arc<D> {
        Arc::clone(&self.device)
    }

    /// Return the provider's stable backend identifier.
    #[must_use]
    pub fn backend_name(&self) -> &'static str {
        self.device.backend_name()
    }

    /// Allocate zeroed typed device storage through Hephaestus.
    ///
    /// The `Pod` bound is Eunomia's host/device layout contract. No local byte
    /// cast or staging buffer is introduced by the adapter.
    pub fn alloc_zeroed<T: Pod>(&self, len: usize) -> Result<D::Buffer<T>> {
        self.device.alloc_zeroed(len)
    }

    /// Upload typed host data through the provider's transfer path.
    pub fn upload<T: Pod>(&self, host: &[T]) -> Result<D::Buffer<T>> {
        self.device.upload(host)
    }

    /// Download typed device data into caller-owned storage.
    pub fn download<T: Pod>(&self, buffer: &D::Buffer<T>, out: &mut [T]) -> Result<()> {
        self.device.download(buffer, out)
    }

    /// Return the provider's topology snapshot, if it reported one.
    #[must_use]
    pub fn topology(&self) -> Option<&themis::GpuTopology> {
        self.device.topology()
    }

    /// Wait for provider-visible work to complete.
    pub fn synchronize(&self) -> Result<()> {
        self.device.synchronize()
    }
}

impl<D> GpuContext<D>
where
    D: ComputeDeviceAcquisition + Send + Sync + 'static,
{
    /// Acquire one provider device using typed backend-neutral preferences.
    pub fn acquire(preferences: DevicePreferences) -> Result<Self> {
        let device = D::try_acquire_device(
            "moirai-gpu",
            preferences.device_preference,
            preferences.optional_features.as_ref(),
            preferences.required_limits,
        )?;
        Ok(Self::from_device(device))
    }
}

#[cfg(test)]
mod tests {
    use super::GpuContext;
    use hephaestus_host::HostDevice;

    #[test]
    fn context_round_trips_provider_owned_values() {
        let context = GpuContext::from_device(HostDevice::new());
        let input = [2_u32, 3, 5, 7];
        let buffer = context.upload(&input).expect("host upload succeeds");
        let mut output = [0_u32; 4];

        context
            .download(&buffer, &mut output)
            .expect("host download succeeds");

        assert_eq!(context.backend_name(), "host");
        assert_eq!(output, input);
    }
}

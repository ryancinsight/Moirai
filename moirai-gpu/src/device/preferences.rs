use std::borrow::Cow;

use hephaestus_core::{DeviceFeature, DeviceLimits, DevicePreference};

/// Backend-neutral device acquisition preferences.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DevicePreferences {
    /// Adapter selection preference.
    pub device_preference: DevicePreference,
    /// Features to enable when the selected provider supports them.
    pub optional_features: Cow<'static, [DeviceFeature]>,
    /// Minimum device limits required by the consumer.
    pub required_limits: DeviceLimits,
}

impl DevicePreferences {
    /// Create preferences with explicit provider-neutral required limits.
    #[must_use]
    pub const fn new(required_limits: DeviceLimits) -> Self {
        Self {
            device_preference: DevicePreference::HighPerformance,
            optional_features: Cow::Borrowed(&[]),
            required_limits,
        }
    }

    /// Select the adapter power/performance preference.
    #[must_use]
    pub const fn with_device_preference(mut self, preference: DevicePreference) -> Self {
        self.device_preference = preference;
        self
    }

    /// Enable optional provider-neutral features.
    #[must_use]
    pub fn with_optional_features(
        mut self,
        features: impl Into<Cow<'static, [DeviceFeature]>>,
    ) -> Self {
        self.optional_features = features.into();
        self
    }

    /// Use the Hephaestus WGPU downlevel acquisition limits.
    #[cfg(feature = "wgpu-backend")]
    #[must_use]
    pub fn wgpu() -> Self {
        Self::new(hephaestus_wgpu::WgpuDevice::downlevel_device_limits())
    }

    /// Use no provider-specific minimum limits for a CUDA acquisition.
    ///
    /// CUDA validates the provider's real device limits after acquisition. A
    /// consumer that has a minimum requirement should construct preferences
    /// with [`Self::new`] and pass that requirement explicitly.
    #[cfg(feature = "cuda-backend")]
    #[must_use]
    pub const fn cuda() -> Self {
        Self::new(DeviceLimits {
            max_buffer_size: 0,
            max_compute_workgroup_size_x: 0,
            max_compute_workgroup_size_y: 0,
            max_compute_workgroup_size_z: 0,
            max_compute_invocations_per_workgroup: 0,
            max_compute_workgroup_storage_size: 0,
            max_storage_buffers_per_shader_stage: None,
            max_buffers_and_acceleration_structures_per_shader_stage: None,
            max_immediate_size: 0,
        })
    }
}

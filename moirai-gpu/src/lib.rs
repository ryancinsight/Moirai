//! Hephaestus-backed GPU scheduling for Moirai.
//!
//! `moirai-gpu` owns the scheduler adapter only. Device acquisition, typed
//! buffers, transfers, kernel dispatch, and synchronization remain in
//! Hephaestus providers. The generic [`GpuContext`] and [`GpuTask`] seams
//! monomorphize over the selected provider; no vendor API or dynamic future
//! crosses this crate's public boundary.

#![deny(missing_docs)]

pub mod device;
pub mod occupancy;
pub mod task;

pub use device::{DevicePreferences, GpuContext};
pub use hephaestus_core::{
    ComputeDevice, ComputeDeviceAcquisition, ComputeDeviceCapabilities, DeviceBuffer,
    DeviceFeature, DeviceLimits, DevicePreference, DispatchGrid, HephaestusError as GpuError,
    Result as GpuResult,
};
pub use mnemosyne_core::KernelResourceBudget;
pub use occupancy::{plan_launch, plan_persistent_launch, resident_blocks, LaunchShape};
pub use task::{ConfiguredGpuTask, FunctionGpuTask, GpuTask, GpuTaskBuilder};

/// The concrete Hephaestus WGPU context exposed by the default provider.
#[cfg(feature = "wgpu-backend")]
pub type WgpuContext = GpuContext<hephaestus_wgpu::WgpuDevice>;

/// The concrete Hephaestus CUDA context.
#[cfg(feature = "cuda-backend")]
pub type CudaContext = GpuContext<hephaestus_cuda::CudaDevice>;

/// Convenient imports for Moirai GPU consumers.
pub mod prelude {
    #[cfg(feature = "cuda-backend")]
    pub use crate::CudaContext;
    #[cfg(feature = "wgpu-backend")]
    pub use crate::WgpuContext;
    pub use crate::{
        plan_launch, plan_persistent_launch, resident_blocks, ConfiguredGpuTask, DevicePreferences,
        FunctionGpuTask, GpuContext, GpuError, GpuResult, GpuTask, GpuTaskBuilder,
        KernelResourceBudget, LaunchShape,
    };
}

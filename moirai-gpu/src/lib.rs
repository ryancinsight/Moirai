//! # Moirai GPU Compute
//!
//! Cross-platform GPU compute integration for the Moirai concurrency library.
//!
//! This crate provides backend-independent GPU launch planning by default and,
//! with the `wgpu-backend` feature, GPU acceleration using wgpu-rs.
//!
//! ## Features
//!
//! - Backend-independent launch planning through [`occupancy`]
//! - Optional cross-platform GPU compute using wgpu-rs
//! - Zero-copy buffer management with automatic memory mapping
//! - Async compute pipeline integration
//! - SIMD-fallback for unsupported hardware
//! - Buffer pooling and resource management
//! - Automatic device selection and capability detection
//!
//! ## Architecture
//!
//! The GPU integration follows these principles:
//! - **Composable**: GPU tasks integrate seamlessly with CPU tasks
//! - **Unix Philosophy**: Single responsibility for GPU compute
//! - **Predictable**: Consistent performance characteristics
//! - **Idiomatic**: Follows Rust and wgpu best practices
//! - **Domain-centric**: Designed specifically for concurrent GPU compute

#![deny(missing_docs)]
#![warn(missing_docs)]
#![allow(clippy::module_name_repetitions)]
// GPU module development allows - per ADR GPU infrastructure is not production critical yet
#![allow(clippy::unnecessary_cast)]
#![allow(dead_code)]
#![allow(unused_imports)]

#[cfg(feature = "wgpu-backend")]
use std::sync::Arc;
#[cfg(feature = "wgpu-backend")]
use std::sync::{Mutex, MutexGuard, PoisonError};

#[cfg(feature = "wgpu-backend")]
pub(crate) fn lock_mutex<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

#[cfg(feature = "wgpu-backend")]
pub mod buffer;
#[cfg(feature = "wgpu-backend")]
pub mod compute;
#[cfg(feature = "wgpu-backend")]
pub mod device;
#[cfg(feature = "wgpu-backend")]
pub mod error;
pub mod occupancy;
#[cfg(feature = "wgpu-backend")]
pub mod pipeline;
#[cfg(feature = "wgpu-backend")]
pub mod task;

// Core exports
#[cfg(feature = "wgpu-backend")]
pub use buffer::{BufferUsage, GpuBuffer, GpuBufferPool};
#[cfg(feature = "wgpu-backend")]
pub use compute::{ComputeKernel, ComputeShader, KernelDispatch};
#[cfg(feature = "wgpu-backend")]
pub use device::{DeviceCapabilities, GpuDevice, GpuDeviceManager};
#[cfg(feature = "wgpu-backend")]
pub use error::{GpuError, GpuResult};
pub use occupancy::{plan_launch, plan_persistent_launch, resident_blocks, LaunchShape};
#[cfg(feature = "wgpu-backend")]
pub use pipeline::{ComputePipeline, PipelineBuilder};
#[cfg(feature = "wgpu-backend")]
pub use task::{GpuTask, GpuTaskBuilder, GpuTaskFuture};

/// GPU compute context that manages devices and resources
#[cfg(feature = "wgpu-backend")]
#[derive(Clone)]
pub struct GpuContext {
    device_manager: Arc<GpuDeviceManager>,
    buffer_pool: Arc<GpuBufferPool>,
}

#[cfg(feature = "wgpu-backend")]
impl GpuContext {
    /// Create a new GPU context with automatic device detection
    pub async fn new() -> GpuResult<Self> {
        let device_manager = Arc::new(GpuDeviceManager::new().await?);
        let buffer_pool = Arc::new(GpuBufferPool::new());

        Ok(Self {
            device_manager,
            buffer_pool,
        })
    }

    /// Create a GPU context with specific device preferences
    pub async fn with_preferences(preferences: DevicePreferences) -> GpuResult<Self> {
        let device_manager = Arc::new(GpuDeviceManager::with_preferences(preferences).await?);
        let buffer_pool = Arc::new(GpuBufferPool::new());

        Ok(Self {
            device_manager,
            buffer_pool,
        })
    }

    /// Get the primary GPU device
    pub fn device(&self) -> &GpuDevice {
        self.device_manager.primary_device()
    }

    /// Get the buffer pool for memory management
    pub fn buffer_pool(&self) -> &GpuBufferPool {
        &self.buffer_pool
    }

    /// Create a new compute pipeline
    pub fn create_pipeline(&self, shader_source: &str) -> PipelineBuilder {
        PipelineBuilder::new(self.device().clone(), shader_source)
    }

    /// Spawn a GPU task
    pub fn spawn_gpu_task<T>(&self, task: T) -> GpuTaskFuture<T::Output>
    where
        T: GpuTask + Send + 'static,
        T::Output: Send + 'static,
    {
        GpuTaskFuture::new(task, self.device().clone())
    }
}

/// Device selection preferences
#[cfg(feature = "wgpu-backend")]
#[derive(Debug, Clone)]
pub struct DevicePreferences {
    /// Prefer discrete GPU over integrated
    pub prefer_discrete: bool,
    /// Minimum required memory (in bytes)
    pub min_memory: u64,
    /// Required features
    pub required_features: wgpu::Features,
    /// Required limits
    pub required_limits: wgpu::Limits,
}

#[cfg(feature = "wgpu-backend")]
impl Default for DevicePreferences {
    fn default() -> Self {
        Self {
            prefer_discrete: false,
            min_memory: 0,
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        }
    }
}

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{plan_launch, plan_persistent_launch, resident_blocks, LaunchShape};
    #[cfg(feature = "wgpu-backend")]
    pub use crate::{
        BufferUsage, ComputeKernel, ComputePipeline, DeviceCapabilities, GpuBuffer, GpuBufferPool,
        GpuContext, GpuDevice, GpuError, GpuResult, GpuTask, GpuTaskBuilder, GpuTaskFuture,
        KernelDispatch, PipelineBuilder,
    };
}

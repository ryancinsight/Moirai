//! # Moirai GPU Compute
//!
//! Cross-platform GPU compute integration for the Moirai concurrency library.
//! 
//! This crate provides GPU acceleration using wgpu-rs for maximum compatibility
//! across platforms while maintaining zero-copy principles and following SOLID/CUPID
//! design patterns.
//!
//! ## Features
//!
//! - Cross-platform GPU compute using wgpu-rs
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

#![warn(missing_docs)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::must_use_candidate)]
// GPU module development allows - per ADR GPU infrastructure is not production critical yet
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![allow(clippy::return_self_not_must_use)]
#![allow(clippy::cast_lossless)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::unnecessary_cast)]
#![allow(clippy::needless_pass_by_value)]
#![allow(clippy::uninlined_format_args)]
#![allow(clippy::unused_async)]
#![allow(clippy::no_effect_underscore_binding)]
#![allow(dead_code)]
#![allow(unused_imports)]

use std::sync::Arc;

pub mod buffer;
pub mod compute;
pub mod device;
pub mod error;
pub mod pipeline;
pub mod task;

// Core exports
pub use buffer::{GpuBuffer, GpuBufferPool, BufferUsage};
pub use compute::{ComputeKernel, ComputeShader, KernelDispatch};
pub use device::{GpuDevice, GpuDeviceManager, DeviceCapabilities};
pub use error::{GpuError, GpuResult};
pub use pipeline::{ComputePipeline, PipelineBuilder};
pub use task::{GpuTask, GpuTaskBuilder, GpuTaskFuture};

/// GPU compute context that manages devices and resources
#[derive(Clone)]
pub struct GpuContext {
    device_manager: Arc<GpuDeviceManager>,
    buffer_pool: Arc<GpuBufferPool>,
}

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

impl Default for DevicePreferences {
    fn default() -> Self {
        Self {
            prefer_discrete: true,
            min_memory: 256 * 1024 * 1024, // 256MB minimum
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
        }
    }
}

/// Prelude module for convenient imports
pub mod prelude {
    pub use crate::{
        BufferUsage, ComputeKernel, ComputePipeline, DeviceCapabilities, GpuBuffer,
        GpuBufferPool, GpuContext, GpuDevice, GpuError, GpuResult, GpuTask,
        GpuTaskBuilder, GpuTaskFuture, KernelDispatch, PipelineBuilder,
    };
}
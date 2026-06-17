//! GPU-specific error types

use std::fmt;

/// GPU computation result type
pub type GpuResult<T> = Result<T, GpuError>;

/// GPU-specific errors
#[derive(Debug, Clone)]
pub enum GpuError {
    /// Device initialization failed
    DeviceInitFailed(String),
    /// No suitable GPU device found
    NoSuitableDevice,
    /// Buffer allocation failed
    BufferAllocationFailed {
        /// Requested size in bytes
        size: u64,
        /// Error message
        message: String,
    },
    /// Shader compilation failed
    ShaderCompilationFailed(String),
    /// Pipeline creation failed
    PipelineCreationFailed(String),
    /// Compute dispatch failed
    ComputeDispatchFailed(String),
    /// Buffer mapping failed
    BufferMappingFailed(String),
    /// Resource limit exceeded
    ResourceLimitExceeded {
        /// Resource type
        resource: String,
        /// Current usage
        current: u64,
        /// Maximum allowed
        limit: u64,
    },
    /// Unsupported operation
    UnsupportedOperation(String),
    /// Validation error
    ValidationError(String),
    /// Timeout waiting for GPU operation
    Timeout,
    /// Internal wgpu error
    WgpuError(String),
}

impl fmt::Display for GpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GpuError::DeviceInitFailed(msg) => {
                write!(f, "GPU device initialization failed: {}", msg)
            }
            GpuError::NoSuitableDevice => write!(f, "No suitable GPU device found"),
            GpuError::BufferAllocationFailed { size, message } => {
                write!(
                    f,
                    "Buffer allocation failed for {} bytes: {}",
                    size, message
                )
            }
            GpuError::ShaderCompilationFailed(msg) => {
                write!(f, "Shader compilation failed: {}", msg)
            }
            GpuError::PipelineCreationFailed(msg) => write!(f, "Pipeline creation failed: {}", msg),
            GpuError::ComputeDispatchFailed(msg) => write!(f, "Compute dispatch failed: {}", msg),
            GpuError::BufferMappingFailed(msg) => write!(f, "Buffer mapping failed: {}", msg),
            GpuError::ResourceLimitExceeded {
                resource,
                current,
                limit,
            } => {
                write!(
                    f,
                    "Resource limit exceeded for {}: {} > {}",
                    resource, current, limit
                )
            }
            GpuError::UnsupportedOperation(msg) => write!(f, "Unsupported operation: {}", msg),
            GpuError::ValidationError(msg) => write!(f, "Validation error: {}", msg),
            GpuError::Timeout => write!(f, "GPU operation timed out"),
            GpuError::WgpuError(msg) => write!(f, "wgpu error: {}", msg),
        }
    }
}

impl std::error::Error for GpuError {}

#[cfg(feature = "wgpu-backend")]
impl From<wgpu::CreateSurfaceError> for GpuError {
    fn from(err: wgpu::CreateSurfaceError) -> Self {
        GpuError::WgpuError(err.to_string())
    }
}

#[cfg(feature = "wgpu-backend")]
impl From<wgpu::RequestDeviceError> for GpuError {
    fn from(err: wgpu::RequestDeviceError) -> Self {
        GpuError::DeviceInitFailed(err.to_string())
    }
}

#[cfg(feature = "wgpu-backend")]
impl From<wgpu::BufferAsyncError> for GpuError {
    fn from(err: wgpu::BufferAsyncError) -> Self {
        GpuError::BufferMappingFailed(err.to_string())
    }
}

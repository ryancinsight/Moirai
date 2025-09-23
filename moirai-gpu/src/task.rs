//! GPU compute tasks integration with Moirai task system

use crate::{error::GpuResult, GpuDevice, GpuError};
use moirai_core::{Task, TaskContext};
use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

/// Trait for GPU-accelerated tasks
pub trait GpuTask: Send + 'static {
    /// Output type of the GPU task
    type Output: Send + 'static;
    
    /// Execute the task on the GPU
    fn execute_gpu(self, device: &GpuDevice) -> impl Future<Output = GpuResult<Self::Output>> + Send;
    
    /// Get estimated GPU memory requirements in bytes
    fn estimated_memory_usage(&self) -> u64 {
        0
    }
    
    /// Get estimated computational complexity (for scheduling)
    fn estimated_complexity(&self) -> u32 {
        1
    }
    
    /// Check if this task can run on the given device
    fn can_run_on_device(&self, device: &GpuDevice) -> bool {
        let _ = device;
        true
    }
}

/// Builder for GPU tasks with configuration
pub struct GpuTaskBuilder<T> {
    task: T,
    memory_hint: Option<u64>,
    complexity_hint: Option<u32>,
    device_requirements: Option<wgpu::Features>,
}

impl<T> GpuTaskBuilder<T>
where
    T: GpuTask,
{
    /// Create a new GPU task builder
    pub fn new(task: T) -> Self {
        Self {
            task,
            memory_hint: None,
            complexity_hint: None,
            device_requirements: None,
        }
    }
    
    /// Set memory usage hint
    pub fn with_memory_hint(mut self, bytes: u64) -> Self {
        self.memory_hint = Some(bytes);
        self
    }
    
    /// Set computational complexity hint
    pub fn with_complexity_hint(mut self, complexity: u32) -> Self {
        self.complexity_hint = Some(complexity);
        self
    }
    
    /// Set required device features
    pub fn with_device_requirements(mut self, features: wgpu::Features) -> Self {
        self.device_requirements = Some(features);
        self
    }
    
    /// Build the configured GPU task
    pub fn build(self) -> ConfiguredGpuTask<T> {
        ConfiguredGpuTask {
            task: self.task,
            memory_hint: self.memory_hint,
            complexity_hint: self.complexity_hint,
            device_requirements: self.device_requirements,
        }
    }
}

/// GPU task with configuration
pub struct ConfiguredGpuTask<T> {
    task: T,
    memory_hint: Option<u64>,
    complexity_hint: Option<u32>,
    device_requirements: Option<wgpu::Features>,
}

impl<T> GpuTask for ConfiguredGpuTask<T>
where
    T: GpuTask,
{
    type Output = T::Output;
    
    async fn execute_gpu(self, device: &GpuDevice) -> GpuResult<Self::Output> {
        // Check device requirements
        if let Some(required_features) = self.device_requirements {
            if !device.supports_features(required_features) {
                return Err(GpuError::UnsupportedOperation(
                    "Device does not support required features".to_string(),
                ));
            }
        }
        
        self.task.execute_gpu(device).await
    }
    
    fn estimated_memory_usage(&self) -> u64 {
        self.memory_hint.unwrap_or_else(|| self.task.estimated_memory_usage())
    }
    
    fn estimated_complexity(&self) -> u32 {
        self.complexity_hint.unwrap_or_else(|| self.task.estimated_complexity())
    }
    
    fn can_run_on_device(&self, device: &GpuDevice) -> bool {
        if let Some(required_features) = self.device_requirements {
            if !device.supports_features(required_features) {
                return false;
            }
        }
        self.task.can_run_on_device(device)
    }
}

/// Future representing a GPU task execution
pub struct GpuTaskFuture<T> {
    future: Pin<Box<dyn Future<Output = GpuResult<T>> + Send>>,
}

impl<T> GpuTaskFuture<T>
where
    T: Send + 'static,
{
    /// Create a new GPU task future
    pub fn new<G>(task: G, device: GpuDevice) -> Self
    where
        G: GpuTask<Output = T> + Send + 'static,
    {
        let future = Box::pin(async move {
            task.execute_gpu(&device).await
        });
        
        Self { future }
    }
}

impl<T> Future for GpuTaskFuture<T> {
    type Output = GpuResult<T>;
    
    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.future.as_mut().poll(cx)
    }
}

/// Adapter to make GPU tasks compatible with Moirai's Task trait
pub struct GpuTaskAdapter<T> {
    gpu_task: T,
    device: GpuDevice,
    context: TaskContext,
}

impl<T> GpuTaskAdapter<T>
where
    T: GpuTask,
{
    /// Create a new GPU task adapter
    pub fn new(gpu_task: T, device: GpuDevice, context: TaskContext) -> Self {
        Self {
            gpu_task,
            device,
            context,
        }
    }
}

impl<T> Task for GpuTaskAdapter<T>
where
    T: GpuTask,
    T::Output: Send + 'static,
{
    type Output = Result<T::Output, GpuError>;
    
    fn execute(self) -> Self::Output {
        // Use pollster to block on the async GPU task
        pollster::block_on(self.gpu_task.execute_gpu(&self.device))
    }
    
    fn context(&self) -> &TaskContext {
        &self.context
    }
    
    fn estimated_cost(&self) -> u32 {
        self.gpu_task.estimated_complexity()
    }
}

/// Simple GPU task for function-based operations
pub struct FunctionGpuTask<F, T> {
    func: F,
    _phantom: std::marker::PhantomData<T>,
}

impl<F, T> FunctionGpuTask<F, T>
where
    F: FnOnce(&GpuDevice) -> GpuResult<T> + Send + 'static,
    T: Send + 'static,
{
    /// Create a new function-based GPU task
    pub fn new(func: F) -> Self {
        Self {
            func,
            _phantom: std::marker::PhantomData,
        }
    }
}

impl<F, T> GpuTask for FunctionGpuTask<F, T>
where
    F: FnOnce(&GpuDevice) -> GpuResult<T> + Send + 'static,
    T: Send + 'static,
{
    type Output = T;
    
    async fn execute_gpu(self, device: &GpuDevice) -> GpuResult<Self::Output> {
        (self.func)(device)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_gpu_task_execution() {
        // Simplified test - testing task creation
        let task = FunctionGpuTask::new(|_device| Ok(42u32));
        // Test that the task was created successfully (no public ID field in this struct)
        let _result = task; // Just verify it compiles
    }
}
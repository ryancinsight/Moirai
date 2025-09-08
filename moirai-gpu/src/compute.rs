//! GPU compute shader and kernel management

use crate::{error::GpuResult, GpuDevice, GpuBuffer};
use wgpu::{ComputePipeline, ComputePipelineDescriptor, PipelineLayoutDescriptor, BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindGroupDescriptor, BindGroupEntry, BindGroup};

/// Compute shader wrapper
pub struct ComputeShader {
    shader_module: wgpu::ShaderModule,
    entry_point: String,
}

impl ComputeShader {
    /// Create a new compute shader from WGSL source
    pub fn from_wgsl(device: &GpuDevice, source: &str, entry_point: &str) -> GpuResult<Self> {
        let shader_module = device.device().create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Moirai Compute Shader"),
            source: wgpu::ShaderSource::Wgsl(source.into()),
        });
        
        Ok(Self {
            shader_module,
            entry_point: entry_point.to_string(),
        })
    }
    
    /// Get the shader module
    pub fn module(&self) -> &wgpu::ShaderModule {
        &self.shader_module
    }
    
    /// Get the entry point name
    pub fn entry_point(&self) -> &str {
        &self.entry_point
    }
}

/// Kernel dispatch configuration
#[derive(Debug, Clone)]
pub struct KernelDispatch {
    /// Workgroup dimensions (x, y, z)
    pub workgroups: (u32, u32, u32),
    /// Workgroup size (threads per workgroup)
    pub workgroup_size: Option<(u32, u32, u32)>,
}

impl KernelDispatch {
    /// Create a 1D dispatch
    pub fn new_1d(workgroups_x: u32) -> Self {
        Self {
            workgroups: (workgroups_x, 1, 1),
            workgroup_size: None,
        }
    }
    
    /// Create a 2D dispatch
    pub fn new_2d(workgroups_x: u32, workgroups_y: u32) -> Self {
        Self {
            workgroups: (workgroups_x, workgroups_y, 1),
            workgroup_size: None,
        }
    }
    
    /// Create a 3D dispatch
    pub fn new_3d(workgroups_x: u32, workgroups_y: u32, workgroups_z: u32) -> Self {
        Self {
            workgroups: (workgroups_x, workgroups_y, workgroups_z),
            workgroup_size: None,
        }
    }
    
    /// Set the workgroup size
    pub fn with_workgroup_size(mut self, x: u32, y: u32, z: u32) -> Self {
        self.workgroup_size = Some((x, y, z));
        self
    }
    
    /// Calculate total number of threads
    pub fn total_threads(&self) -> u64 {
        let (x, y, z) = self.workgroups;
        (x as u64) * (y as u64) * (z as u64) * 
        if let Some((wx, wy, wz)) = self.workgroup_size {
            (wx as u64) * (wy as u64) * (wz as u64)
        } else {
            1
        }
    }
}

/// Compute kernel for GPU execution
pub struct ComputeKernel {
    pipeline: ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    device: GpuDevice,
}

impl ComputeKernel {
    /// Create a new compute kernel
    pub fn new(device: GpuDevice, shader: ComputeShader, bind_group_entries: &[BindGroupLayoutEntry]) -> GpuResult<Self> {
        let bind_group_layout = device.device().create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("Moirai Compute Bind Group Layout"),
            entries: bind_group_entries,
        });
        
        let pipeline_layout = device.device().create_pipeline_layout(&PipelineLayoutDescriptor {
            label: Some("Moirai Compute Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        
        let pipeline = device.device().create_compute_pipeline(&ComputePipelineDescriptor {
            label: Some("Moirai Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: shader.module(),
            entry_point: shader.entry_point(),
        });
        
        Ok(Self {
            pipeline,
            bind_group_layout,
            device,
        })
    }
    
    /// Create a bind group for this kernel
    pub fn create_bind_group(&self, buffers: &[&GpuBuffer]) -> GpuResult<BindGroup> {
        let entries: Vec<BindGroupEntry> = buffers
            .iter()
            .enumerate()
            .map(|(i, buffer)| BindGroupEntry {
                binding: i as u32,
                resource: buffer.buffer().as_entire_binding(),
            })
            .collect();
        
        let bind_group = self.device.device().create_bind_group(&BindGroupDescriptor {
            label: Some("Moirai Compute Bind Group"),
            layout: &self.bind_group_layout,
            entries: &entries,
        });
        
        Ok(bind_group)
    }
    
    /// Execute the kernel with given dispatch configuration
    pub fn execute(&self, bind_group: &BindGroup, dispatch: &KernelDispatch) -> GpuResult<()> {
        let mut encoder = self.device.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Moirai Compute Encoder"),
        });
        
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Moirai Compute Pass"),
                timestamp_writes: None,
            });
            
            compute_pass.set_pipeline(&self.pipeline);
            compute_pass.set_bind_group(0, bind_group, &[]);
            compute_pass.dispatch_workgroups(dispatch.workgroups.0, dispatch.workgroups.1, dispatch.workgroups.2);
        }
        
        self.device.queue().submit(std::iter::once(encoder.finish()));
        Ok(())
    }
    
    /// Execute the kernel asynchronously and wait for completion
    pub async fn execute_async(&self, bind_group: &BindGroup, dispatch: &KernelDispatch) -> GpuResult<()> {
        self.execute(bind_group, dispatch)?;
        
        // Poll device to ensure completion
        self.device.device().poll(wgpu::Maintain::Wait);
        
        Ok(())
    }
    
    /// Get the compute pipeline
    pub fn pipeline(&self) -> &ComputePipeline {
        &self.pipeline
    }
}

/// Convenience function to create a storage buffer bind group layout entry
pub fn storage_buffer_entry(binding: u32, read_only: bool) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

/// Convenience function to create a uniform buffer bind group layout entry
pub fn uniform_buffer_entry(binding: u32) -> BindGroupLayoutEntry {
    BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{BufferUsage, DevicePreferences, GpuContext};
    
    const SIMPLE_COMPUTE_SHADER: &str = r#"
        @group(0) @binding(0) var<storage, read_write> data: array<f32>;
        
        @compute @workgroup_size(64)
        fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
            let index = global_id.x;
            if (index >= arrayLength(&data)) {
                return;
            }
            data[index] = data[index] * 2.0;
        }
    "#;
    
    #[test]
    fn test_compute_kernel_creation() {
        // Simplified test - in a full implementation, this would use Moirai's async runtime
        // For now, just test that the GPU types are properly defined
        let preferences = DevicePreferences::default();
        assert_eq!(preferences.preferred_backend, PreferredBackend::Auto);
    }
    
    #[test] 
    fn test_kernel_dispatch() {
        // Simplified test - testing GPU dispatch structures
        let dispatch = KernelDispatch::new_1d(16);
        assert_eq!(dispatch.groups, (16, 1, 1));
        
        let dispatch_3d = KernelDispatch::new_3d(4, 4, 4);
        assert_eq!(dispatch_3d.groups, (4, 4, 4));
    }
}
//! GPU device management and capabilities detection

use crate::{error::GpuResult, DevicePreferences, GpuError};
use std::sync::Arc;
use wgpu::{Device, Queue, Adapter, Features, Limits};

/// GPU device wrapper with enhanced capabilities
#[derive(Clone)]
pub struct GpuDevice {
    device: Arc<Device>,
    queue: Arc<Queue>,
    adapter: Arc<Adapter>,
    capabilities: DeviceCapabilities,
}

impl GpuDevice {
    /// Create a new GPU device from wgpu components
    pub fn new(device: Device, queue: Queue, adapter: Adapter) -> Self {
        let capabilities = DeviceCapabilities::from_adapter(&adapter);
        
        Self {
            device: Arc::new(device),
            queue: Arc::new(queue),
            adapter: Arc::new(adapter),
            capabilities,
        }
    }
    
    /// Get the underlying wgpu device
    pub fn device(&self) -> &Device {
        &self.device
    }
    
    /// Get the command queue
    pub fn queue(&self) -> &Queue {
        &self.queue
    }
    
    /// Get the adapter information
    pub fn adapter(&self) -> &Adapter {
        &self.adapter
    }
    
    /// Get device capabilities
    pub fn capabilities(&self) -> &DeviceCapabilities {
        &self.capabilities
    }
    
    /// Check if device supports specific features
    pub fn supports_features(&self, features: Features) -> bool {
        self.device.features().contains(features)
    }
    
    /// Check if device meets limit requirements
    pub fn meets_limits(&self, limits: &Limits) -> bool {
        // Simple comparison - in practice you'd want more sophisticated checking
        self.device.limits().max_buffer_size >= limits.max_buffer_size
    }
}

/// GPU device capabilities and characteristics
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    /// Device name
    pub name: String,
    /// Device type (discrete, integrated, etc.)
    pub device_type: wgpu::DeviceType,
    /// Supported features
    pub features: Features,
    /// Device limits
    pub limits: Limits,
    /// Memory information
    pub memory_info: MemoryInfo,
    /// Vendor ID
    pub vendor_id: u32,
    /// Device ID
    pub device_id: u32,
}

impl DeviceCapabilities {
    /// Extract capabilities from a wgpu adapter
    pub fn from_adapter(adapter: &Adapter) -> Self {
        let info = adapter.get_info();
        let features = adapter.features();
        let limits = adapter.limits();
        let limits_clone = limits.clone();
        
        Self {
            name: info.name,
            device_type: info.device_type,
            features,
            limits,
            memory_info: MemoryInfo::estimate_from_limits(&limits_clone),
            vendor_id: info.vendor,
            device_id: info.device,
        }
    }
    
    /// Check if this device is suitable for given preferences
    pub fn meets_preferences(&self, preferences: &DevicePreferences) -> bool {
        // Check memory requirements
        if self.memory_info.total_memory < preferences.min_memory {
            return false;
        }
        
        // Check required features
        if !self.features.contains(preferences.required_features) {
            return false;
        }
        
        // Check limits (this is approximate since Limits doesn't implement PartialOrd)
        if !self.meets_limits_approx(&preferences.required_limits) {
            return false;
        }
        
        true
    }
    
    /// Approximate limits checking
    fn meets_limits_approx(&self, required: &Limits) -> bool {
        self.limits.max_texture_dimension_1d >= required.max_texture_dimension_1d
            && self.limits.max_texture_dimension_2d >= required.max_texture_dimension_2d
            && self.limits.max_texture_dimension_3d >= required.max_texture_dimension_3d
            && self.limits.max_buffer_size >= required.max_buffer_size
            && self.limits.max_storage_buffers_per_shader_stage >= required.max_storage_buffers_per_shader_stage
    }
    
    /// Get a quality score for device selection
    pub fn quality_score(&self, preferences: &DevicePreferences) -> u32 {
        let mut score = 0;
        
        // Prefer discrete GPUs
        if preferences.prefer_discrete && self.device_type == wgpu::DeviceType::DiscreteGpu {
            score += 100;
        }
        
        // Memory score (more memory = better)
        score += (self.memory_info.total_memory / (1024 * 1024)) as u32; // Score in MB
        
        // Feature score
        score += self.features.bits().count_ones() * 10;
        
        score
    }
}

/// Memory information for the GPU
#[derive(Debug, Clone)]
pub struct MemoryInfo {
    /// Total GPU memory in bytes (estimated)
    pub total_memory: u64,
    /// Available memory in bytes (estimated)
    pub available_memory: u64,
}

impl MemoryInfo {
    /// Estimate memory from wgpu limits
    pub fn estimate_from_limits(limits: &Limits) -> Self {
        // Rough estimation based on buffer limits
        let estimated_total = (limits.max_buffer_size as u64).min(4 * 1024 * 1024 * 1024); // Cap at 4GB
        
        Self {
            total_memory: estimated_total,
            available_memory: estimated_total, // Assume all available initially
        }
    }
}

/// GPU device manager for device selection and management
pub struct GpuDeviceManager {
    instance: wgpu::Instance,
    primary_device: GpuDevice,
    available_devices: Vec<GpuDevice>,
}

impl GpuDeviceManager {
    /// Create a new device manager with automatic device selection
    pub async fn new() -> GpuResult<Self> {
        Self::with_preferences(DevicePreferences::default()).await
    }
    
    /// Create a device manager with specific preferences
    pub async fn with_preferences(preferences: DevicePreferences) -> GpuResult<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        
        // Find the best adapter
        let mut best_adapter = None;
        let mut best_score = 0;
        
        for adapter in instance.enumerate_adapters(wgpu::Backends::all()) {
            let capabilities = DeviceCapabilities::from_adapter(&adapter);
            if capabilities.meets_preferences(&preferences) {
                let score = capabilities.quality_score(&preferences);
                if score > best_score {
                    best_score = score;
                    best_adapter = Some(adapter);
                }
            }
        }
        
        let adapter = best_adapter.ok_or(GpuError::NoSuitableDevice)?;
        
        // Request device and queue
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    label: Some("Moirai GPU Device"),
                    required_features: preferences.required_features,
                    required_limits: preferences.required_limits.clone(),
                },
                None,
            )
            .await?;
        
        let primary_device = GpuDevice::new(device, queue, adapter);
        
        Ok(Self {
            instance,
            primary_device,
            available_devices: vec![], // Simplified for now
        })
    }
    
    /// Get the primary GPU device
    pub fn primary_device(&self) -> &GpuDevice {
        &self.primary_device
    }
    
    /// Get all available devices
    pub fn available_devices(&self) -> &[GpuDevice] {
        &self.available_devices
    }
    
    /// Get the wgpu instance
    pub fn instance(&self) -> &wgpu::Instance {
        &self.instance
    }
}
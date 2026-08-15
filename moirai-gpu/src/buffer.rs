//! GPU buffer management with zero-copy principles

#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]
#![expect(
    clippy::unwrap_used,
    reason = "ratchet MOIRAI-UNWRAP-1: pre-existing debt"
)]

use crate::{error::GpuResult, GpuDevice, GpuError};
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Mutex;
use wgpu::{util::DeviceExt, Buffer, BufferDescriptor, BufferUsages};

fn checked_buffer_range(
    buffer_size: u64,
    offset: u64,
    requested_size: Option<u64>,
) -> GpuResult<Range<u64>> {
    if offset > buffer_size {
        return Err(GpuError::ValidationError(format!(
            "Buffer offset {offset} exceeds buffer size {buffer_size}"
        )));
    }

    let size = requested_size.unwrap_or(buffer_size - offset);
    let end = offset.checked_add(size).ok_or_else(|| {
        GpuError::ValidationError(format!(
            "Buffer range {offset}..{size} overflows the address space"
        ))
    })?;

    if end > buffer_size {
        return Err(GpuError::ValidationError(format!(
            "Buffer range {offset}..{end} exceeds buffer size {buffer_size}"
        )));
    }

    Ok(offset..end)
}

/// Buffer usage patterns for optimization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BufferUsage {
    /// Storage buffer for compute shaders (read/write)
    Storage,
    /// Uniform buffer for constants
    Uniform,
    /// Vertex buffer for rendering
    Vertex,
    /// Index buffer for rendering
    Index,
    /// Staging buffer for CPU-GPU transfers
    Staging,
    /// Custom usage combination
    Custom(BufferUsages),
}

impl BufferUsage {
    /// Convert to wgpu buffer usages
    pub fn to_wgpu_usage(self) -> BufferUsages {
        match self {
            BufferUsage::Storage => {
                BufferUsages::STORAGE | BufferUsages::COPY_DST | BufferUsages::COPY_SRC
            }
            BufferUsage::Uniform => BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            BufferUsage::Vertex => BufferUsages::VERTEX | BufferUsages::COPY_DST,
            BufferUsage::Index => BufferUsages::INDEX | BufferUsages::COPY_DST,
            BufferUsage::Staging => {
                BufferUsages::MAP_READ
                    | BufferUsages::MAP_WRITE
                    | BufferUsages::COPY_SRC
                    | BufferUsages::COPY_DST
            }
            BufferUsage::Custom(usage) => usage,
        }
    }
}

/// GPU buffer wrapper with enhanced functionality
pub struct GpuBuffer {
    buffer: Buffer,
    size: u64,
    usage: BufferUsage,
    device: GpuDevice,
}

impl GpuBuffer {
    /// Create a new GPU buffer
    pub fn new(device: GpuDevice, size: u64, usage: BufferUsage) -> GpuResult<Self> {
        let buffer = device.device().create_buffer(&BufferDescriptor {
            label: Some("Moirai GPU Buffer"),
            size,
            usage: usage.to_wgpu_usage(),
            mapped_at_creation: false,
        });

        Ok(Self {
            buffer,
            size,
            usage,
            device,
        })
    }

    /// Create a buffer with initial data
    pub fn with_data<T: bytemuck::Pod>(
        device: GpuDevice,
        data: &[T],
        usage: BufferUsage,
    ) -> GpuResult<Self> {
        let bytes = bytemuck::cast_slice(data);
        let buffer = device
            .device()
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Moirai GPU Buffer"),
                contents: bytes,
                usage: usage.to_wgpu_usage(),
            });

        Ok(Self {
            buffer,
            size: bytes.len() as u64,
            usage,
            device,
        })
    }

    /// Get the underlying wgpu buffer
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// Get buffer size in bytes
    pub fn size(&self) -> u64 {
        self.size
    }

    /// Get buffer usage
    pub fn usage(&self) -> BufferUsage {
        self.usage
    }

    /// Write data to the buffer
    pub fn write<T: bytemuck::Pod>(&self, data: &[T], offset: u64) -> GpuResult<()> {
        let bytes = bytemuck::cast_slice(data);
        let byte_count = u64::try_from(bytes.len()).map_err(|_| {
            GpuError::ValidationError("Buffer write length exceeds u64 address space".to_string())
        })?;
        checked_buffer_range(self.size, offset, Some(byte_count))?;

        self.device
            .queue()
            .write_buffer(&self.buffer, offset, bytes);
        Ok(())
    }

    /// Map buffer for reading (async)
    pub async fn map_read(&self, offset: u64, size: Option<u64>) -> GpuResult<()> {
        let buffer_slice = self
            .buffer
            .slice(checked_buffer_range(self.size, offset, size)?);
        let (sender, receiver) = futures::channel::oneshot::channel();

        buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });

        // Poll the device to process the mapping
        self.device.device().poll(wgpu::Maintain::Wait);

        receiver
            .await
            .map_err(|_| GpuError::BufferMappingFailed("Channel closed".to_string()))?
            .map_err(GpuError::from)?;

        Ok(())
    }

    /// Map buffer for writing (async)
    pub async fn map_write(&self, offset: u64, size: Option<u64>) -> GpuResult<()> {
        let buffer_slice = self
            .buffer
            .slice(checked_buffer_range(self.size, offset, size)?);
        let (sender, receiver) = futures::channel::oneshot::channel();

        buffer_slice.map_async(wgpu::MapMode::Write, move |result| {
            let _ = sender.send(result);
        });

        // Poll the device to process the mapping
        self.device.device().poll(wgpu::Maintain::Wait);

        receiver
            .await
            .map_err(|_| GpuError::BufferMappingFailed("Channel closed".to_string()))?
            .map_err(GpuError::from)?;

        Ok(())
    }

    /// Copy data from another buffer
    pub fn copy_from(&self, src: &GpuBuffer, encoder: &mut wgpu::CommandEncoder) -> GpuResult<()> {
        if self.size != src.size {
            return Err(GpuError::ValidationError(
                "Buffer sizes must match for copy".to_string(),
            ));
        }

        encoder.copy_buffer_to_buffer(&src.buffer, 0, &self.buffer, 0, self.size);
        Ok(())
    }
}

#[cfg(test)]
mod range_tests {
    use super::{checked_buffer_range, GpuError};

    #[test]
    fn buffer_range_accepts_empty_end_boundary() {
        assert!(matches!(
            checked_buffer_range(16, 16, None),
            Ok(range) if range == (16..16)
        ));
    }

    #[test]
    fn buffer_range_rejects_offset_past_buffer() {
        let Err(GpuError::ValidationError(message)) = checked_buffer_range(16, 17, None) else {
            panic!("an out-of-bounds offset must be rejected");
        };
        assert!(message.contains("exceeds buffer size"));
    }

    #[test]
    fn buffer_range_rejects_requested_end_past_buffer() {
        let Err(GpuError::ValidationError(message)) = checked_buffer_range(16, 12, Some(5)) else {
            panic!("an out-of-bounds range must be rejected");
        };
        assert!(message.contains("exceeds buffer size"));
    }

    #[test]
    fn buffer_range_rejects_end_overflow() {
        let Err(GpuError::ValidationError(message)) =
            checked_buffer_range(u64::MAX, u64::MAX - 1, Some(2))
        else {
            panic!("an overflowing range must be rejected");
        };
        assert!(message.contains("overflows"));
    }
}

/// Buffer pool for efficient memory management
pub struct GpuBufferPool {
    pools: Mutex<HashMap<(u64, BufferUsage), Vec<GpuBuffer>>>,
    stats: Mutex<PoolStats>,
}

#[derive(Debug, Default)]
struct PoolStats {
    total_allocated: u64,
    total_reused: u64,
    peak_memory_usage: u64,
    current_memory_usage: u64,
}

impl GpuBufferPool {
    /// Create a new buffer pool
    pub fn new() -> Self {
        Self {
            pools: Mutex::new(HashMap::new()),
            stats: Mutex::new(PoolStats::default()),
        }
    }

    /// Acquire a buffer from the pool or create a new one
    pub fn acquire(
        &self,
        device: GpuDevice,
        size: u64,
        usage: BufferUsage,
    ) -> GpuResult<GpuBuffer> {
        let mut pools = self.pools.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        let key = (size, usage);

        if let Some(pool) = pools.get_mut(&key) {
            if let Some(buffer) = pool.pop() {
                stats.total_reused += 1;
                return Ok(buffer);
            }
        }

        // Create new buffer
        let buffer = GpuBuffer::new(device, size, usage)?;
        stats.total_allocated += 1;
        stats.current_memory_usage += size;
        stats.peak_memory_usage = stats.peak_memory_usage.max(stats.current_memory_usage);

        Ok(buffer)
    }

    /// Return a buffer to the pool
    pub fn release(&self, buffer: GpuBuffer) {
        let mut pools = self.pools.lock().unwrap();
        let key = (buffer.size(), buffer.usage());

        pools.entry(key).or_default().push(buffer);
    }

    /// Get pool statistics
    pub fn stats(&self) -> (u64, u64, u64, u64) {
        let stats = self.stats.lock().unwrap();
        (
            stats.total_allocated,
            stats.total_reused,
            stats.peak_memory_usage,
            stats.current_memory_usage,
        )
    }

    /// Clear the pool and free all buffers
    pub fn clear(&self) {
        let mut pools = self.pools.lock().unwrap();
        let mut stats = self.stats.lock().unwrap();

        pools.clear();
        stats.current_memory_usage = 0;
    }
}

impl Default for GpuBufferPool {
    fn default() -> Self {
        Self::new()
    }
}

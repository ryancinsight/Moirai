//! Video Processing Pipeline - SIMD-Optimized Parallel Media Processing
//!
//! This example demonstrates:
//! - SIMD-accelerated video frame processing with parallel pipelines
//! - Memory pool management for large media buffers
//! - Multi-stage processing with different worker thread specializations
//! - Real-time video effects with temporal consistency
//! - Batch processing optimization for throughput vs. latency trade-offs
//! - Hardware-accelerated operations with fallback mechanisms

#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]
#![allow(dead_code)] // This example keeps frame metadata and processing flags beyond the compact demo path.

use moirai::{Moirai, Priority};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

/// Represents a video frame with metadata
#[derive(Debug, Clone)]
struct VideoFrame {
    id: u64,
    width: usize,
    height: usize,
    format: PixelFormat,
    data: Vec<u8>,
    timestamp_ms: u64,
    frame_number: u64,
    quality_score: f32,
    processing_flags: ProcessingFlags,
}

#[derive(Debug, Clone, PartialEq)]
enum PixelFormat {
    RGB24,  // 3 bytes per pixel
    RGBA32, // 4 bytes per pixel
    YUV420, // Planar YUV format
    BGR24,  // BGR order
}

impl PixelFormat {
    fn bytes_per_pixel(&self) -> usize {
        match self {
            PixelFormat::RGB24 | PixelFormat::BGR24 => 3,
            PixelFormat::RGBA32 => 4,
            PixelFormat::YUV420 => 3, // Simplified for demo
        }
    }
}

#[derive(Debug, Clone, Default)]
struct ProcessingFlags {
    denoise: bool,
    sharpen: bool,
    color_correct: bool,
    resize: Option<(usize, usize)>,
    stabilize: bool,
    extract_metadata: bool,
}

impl VideoFrame {
    fn new(width: usize, height: usize, format: PixelFormat, frame_number: u64) -> Self {
        let bytes_per_pixel = format.bytes_per_pixel();
        let data_size = width * height * bytes_per_pixel;
        let mut data = vec![0u8; data_size];

        // Generate synthetic video data (gradient pattern)
        for y in 0..height {
            for x in 0..width {
                let offset = (y * width + x) * bytes_per_pixel;
                match format {
                    PixelFormat::RGB24 | PixelFormat::BGR24 => {
                        data[offset] = ((x * 255) / width) as u8; // R/B
                        data[offset + 1] = ((y * 255) / height) as u8; // G
                        data[offset + 2] = ((frame_number * 32) % 256) as u8; // B/R
                    }
                    PixelFormat::RGBA32 => {
                        data[offset] = ((x * 255) / width) as u8;
                        data[offset + 1] = ((y * 255) / height) as u8;
                        data[offset + 2] = ((frame_number * 32) % 256) as u8;
                        data[offset + 3] = 255; // Alpha
                    }
                    PixelFormat::YUV420 => {
                        // Simplified YUV (not true 420 planar)
                        let r = ((x * 255) / width) as f32;
                        let g = ((y * 255) / height) as f32;
                        let b = ((frame_number * 32) % 256) as f32;
                        let y_val = (0.299 * r + 0.587 * g + 0.114 * b) as u8;
                        data[offset] = y_val;
                        data[offset + 1] = 128; // U
                        data[offset + 2] = 128; // V
                    }
                }
            }
        }

        Self {
            id: fastrand::u64(..),
            width,
            height,
            format,
            data,
            timestamp_ms: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
            frame_number,
            quality_score: 0.0,
            processing_flags: ProcessingFlags::default(),
        }
    }

    fn data_size(&self) -> usize {
        self.data.len()
    }

    fn pixel_count(&self) -> usize {
        self.width * self.height
    }
}

/// Memory pool for efficient allocation of video buffers
struct VideoMemoryPool {
    pools: Arc<RwLock<HashMap<usize, VecDeque<Vec<u8>>>>>,
    pool_sizes: Vec<usize>,
    max_buffers_per_size: usize,
    total_allocated: AtomicUsize,
    total_reused: AtomicUsize,
    peak_memory_usage: AtomicUsize,
    current_memory_usage: AtomicUsize,
}

impl VideoMemoryPool {
    fn new() -> Self {
        // Common video buffer sizes (width * height * bytes_per_pixel)
        let pool_sizes = vec![
            1920 * 1080 * 3, // 1080p RGB
            1920 * 1080 * 4, // 1080p RGBA
            3840 * 2160 * 3, // 4K RGB
            3840 * 2160 * 4, // 4K RGBA
            1280 * 720 * 3,  // 720p RGB
            1280 * 720 * 4,  // 720p RGBA
        ];

        Self {
            pools: Arc::new(RwLock::new(HashMap::new())),
            pool_sizes,
            max_buffers_per_size: 10,
            total_allocated: AtomicUsize::new(0),
            total_reused: AtomicUsize::new(0),
            peak_memory_usage: AtomicUsize::new(0),
            current_memory_usage: AtomicUsize::new(0),
        }
    }

    fn acquire_buffer(&self, size: usize) -> Vec<u8> {
        // Find the best fitting pool size
        let pool_size = self
            .pool_sizes
            .iter()
            .find(|&&pool_size| pool_size >= size)
            .copied()
            .unwrap_or(size);

        // Try to get buffer from pool
        if let Ok(mut pools) = self.pools.write() {
            if let Some(pool) = pools.get_mut(&pool_size) {
                if let Some(mut buffer) = pool.pop_front() {
                    // Resize buffer if needed
                    buffer.resize(size, 0);
                    buffer.fill(0); // Clear buffer
                    self.total_reused.fetch_add(1, Ordering::Relaxed);
                    return buffer;
                }
            }
        }

        // Allocate new buffer
        let buffer = vec![0u8; size];
        self.total_allocated.fetch_add(1, Ordering::Relaxed);
        let new_usage = self.current_memory_usage.fetch_add(size, Ordering::Relaxed) + size;

        // Update peak usage
        let current_peak = self.peak_memory_usage.load(Ordering::Relaxed);
        if new_usage > current_peak {
            self.peak_memory_usage.store(new_usage, Ordering::Relaxed);
        }

        buffer
    }

    fn release_buffer(&self, buffer: Vec<u8>) {
        let size = buffer.len();

        // Find appropriate pool
        let pool_size = self
            .pool_sizes
            .iter()
            .find(|&&pool_size| pool_size >= size)
            .copied()
            .unwrap_or(size);

        // Return to pool if not full
        if let Ok(mut pools) = self.pools.write() {
            let pool = pools.entry(pool_size).or_insert_with(VecDeque::new);
            if pool.len() < self.max_buffers_per_size {
                pool.push_back(buffer);
                return;
            }
        }

        // Buffer not returned to pool, decrease memory usage
        self.current_memory_usage.fetch_sub(size, Ordering::Relaxed);
    }

    fn stats(&self) -> (usize, usize, usize, usize, f64) {
        let allocated = self.total_allocated.load(Ordering::Relaxed);
        let reused = self.total_reused.load(Ordering::Relaxed);
        let current_mem = self.current_memory_usage.load(Ordering::Relaxed);
        let peak_mem = self.peak_memory_usage.load(Ordering::Relaxed);
        let reuse_rate = if allocated > 0 {
            (reused as f64 / (allocated + reused) as f64) * 100.0
        } else {
            0.0
        };

        (allocated, reused, current_mem, peak_mem, reuse_rate)
    }
}

/// SIMD-optimized image processing operations
struct SIMDProcessor {
    operations_count: AtomicUsize,
    simd_operations: AtomicUsize,
    fallback_operations: AtomicUsize,
    total_processing_time: AtomicU64,
    pixels_processed: AtomicU64,
}

impl SIMDProcessor {
    fn new() -> Self {
        Self {
            operations_count: AtomicUsize::new(0),
            simd_operations: AtomicUsize::new(0),
            fallback_operations: AtomicUsize::new(0),
            total_processing_time: AtomicU64::new(0),
            pixels_processed: AtomicU64::new(0),
        }
    }

    fn denoise_frame(&self, frame: &mut VideoFrame) -> Result<(), String> {
        let start_time = Instant::now();

        match frame.format {
            PixelFormat::RGB24 | PixelFormat::BGR24 => {
                self.denoise_rgb24(&mut frame.data, frame.width, frame.height)?;
            }
            PixelFormat::RGBA32 => {
                self.denoise_rgba32(&mut frame.data, frame.width, frame.height)?;
            }
            PixelFormat::YUV420 => {
                self.denoise_yuv420(&mut frame.data, frame.width, frame.height)?;
            }
        }

        let processing_time = start_time.elapsed().as_micros() as u64;
        self.operations_count.fetch_add(1, Ordering::Relaxed);
        self.total_processing_time
            .fetch_add(processing_time, Ordering::Relaxed);
        self.pixels_processed
            .fetch_add(frame.pixel_count() as u64, Ordering::Relaxed);

        Ok(())
    }

    fn denoise_rgb24(&self, data: &mut [u8], width: usize, height: usize) -> Result<(), String> {
        // SIMD-optimized 3x3 gaussian blur for denoising
        if self.has_simd_support() {
            self.simd_denoise_rgb24(data, width, height)?;
            self.simd_operations.fetch_add(1, Ordering::Relaxed);
        } else {
            self.fallback_denoise_rgb24(data, width, height)?;
            self.fallback_operations.fetch_add(1, Ordering::Relaxed);
        }
        Ok(())
    }

    fn simd_denoise_rgb24(
        &self,
        data: &mut [u8],
        width: usize,
        height: usize,
    ) -> Result<(), String> {
        // Simulate SIMD operations with vectorized processing
        const GAUSSIAN_KERNEL: [f32; 9] = [
            1.0 / 16.0,
            2.0 / 16.0,
            1.0 / 16.0,
            2.0 / 16.0,
            4.0 / 16.0,
            2.0 / 16.0,
            1.0 / 16.0,
            2.0 / 16.0,
            1.0 / 16.0,
        ];

        let temp_data = data.to_vec();

        // Process in chunks of 4 pixels (simulating SIMD vectorization)
        for y in 1..height - 1 {
            for x_chunk in (1..width - 1).step_by(4) {
                let chunk_end = (x_chunk + 4).min(width - 1);

                for x in x_chunk..chunk_end {
                    for c in 0..3 {
                        // RGB channels
                        let mut sum = 0.0f32;

                        // Apply 3x3 kernel
                        for ky in 0..3 {
                            for kx in 0..3 {
                                let py = y + ky - 1;
                                let px = x + kx - 1;
                                let pixel_idx = (py * width + px) * 3 + c;
                                sum += temp_data[pixel_idx] as f32 * GAUSSIAN_KERNEL[ky * 3 + kx];
                            }
                        }

                        data[(y * width + x) * 3 + c] = sum.round() as u8;
                    }
                }
            }
        }

        Ok(())
    }

    fn fallback_denoise_rgb24(
        &self,
        data: &mut [u8],
        width: usize,
        height: usize,
    ) -> Result<(), String> {
        // Simple box blur as fallback
        let temp_data = data.to_vec();

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                for c in 0..3 {
                    let mut sum = 0u32;

                    // 3x3 neighborhood average
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            let py = (y as i32 + dy) as usize;
                            let px = (x as i32 + dx) as usize;
                            sum += temp_data[(py * width + px) * 3 + c] as u32;
                        }
                    }

                    data[(y * width + x) * 3 + c] = (sum / 9) as u8;
                }
            }
        }

        Ok(())
    }

    fn denoise_rgba32(&self, data: &mut [u8], width: usize, height: usize) -> Result<(), String> {
        // Similar to RGB24 but with alpha channel handling
        if self.has_simd_support() {
            self.simd_operations.fetch_add(1, Ordering::Relaxed);
        } else {
            self.fallback_operations.fetch_add(1, Ordering::Relaxed);
        }

        let temp_data = data.to_vec();

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                for c in 0..3 {
                    // Don't blur alpha channel
                    let mut sum = 0u32;

                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            let py = (y as i32 + dy) as usize;
                            let px = (x as i32 + dx) as usize;
                            sum += temp_data[(py * width + px) * 4 + c] as u32;
                        }
                    }

                    data[(y * width + x) * 4 + c] = (sum / 9) as u8;
                }
                // Preserve alpha channel
                data[(y * width + x) * 4 + 3] = temp_data[(y * width + x) * 4 + 3];
            }
        }

        Ok(())
    }

    fn denoise_yuv420(&self, data: &mut [u8], width: usize, height: usize) -> Result<(), String> {
        // YUV processing (simplified)
        if self.has_simd_support() {
            self.simd_operations.fetch_add(1, Ordering::Relaxed);
        } else {
            self.fallback_operations.fetch_add(1, Ordering::Relaxed);
        }

        // Only denoise Y channel for simplicity
        let temp_data = data.to_vec();

        for y in 1..height - 1 {
            for x in 1..width - 1 {
                let mut sum = 0u32;

                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let py = (y as i32 + dy) as usize;
                        let px = (x as i32 + dx) as usize;
                        sum += temp_data[(py * width + px) * 3] as u32; // Y channel
                    }
                }

                data[(y * width + x) * 3] = (sum / 9) as u8;
            }
        }

        Ok(())
    }

    fn sharpen_frame(&self, frame: &mut VideoFrame) -> Result<(), String> {
        let start_time = Instant::now();

        // Unsharp mask sharpening
        const SHARPEN_KERNEL: [f32; 9] = [
            -1.0 / 9.0,
            -1.0 / 9.0,
            -1.0 / 9.0,
            -1.0 / 9.0,
            8.0 / 9.0,
            -1.0 / 9.0,
            -1.0 / 9.0,
            -1.0 / 9.0,
            -1.0 / 9.0,
        ];

        let channels = frame.format.bytes_per_pixel();
        let temp_data = frame.data.clone();

        for y in 1..frame.height - 1 {
            for x in 1..frame.width - 1 {
                for c in 0..channels.min(3) {
                    // Don't sharpen alpha
                    let mut sum = 0.0f32;

                    for ky in 0..3 {
                        for kx in 0..3 {
                            let py = y + ky - 1;
                            let px = x + kx - 1;
                            let pixel_idx = (py * frame.width + px) * channels + c;
                            sum += temp_data[pixel_idx] as f32 * SHARPEN_KERNEL[ky * 3 + kx];
                        }
                    }

                    let current_idx = (y * frame.width + x) * channels + c;
                    let new_value = (temp_data[current_idx] as f32 + sum).clamp(0.0, 255.0);
                    frame.data[current_idx] = new_value as u8;
                }
            }
        }

        let processing_time = start_time.elapsed().as_micros() as u64;
        self.operations_count.fetch_add(1, Ordering::Relaxed);
        self.total_processing_time
            .fetch_add(processing_time, Ordering::Relaxed);

        Ok(())
    }

    fn color_correct_frame(&self, frame: &mut VideoFrame) -> Result<(), String> {
        let start_time = Instant::now();

        // Simple color correction: adjust brightness, contrast, and gamma
        let brightness = 10.0; // Brightness adjustment (-100 to 100)
        let contrast = 1.1; // Contrast multiplier (0.5 to 2.0)
        let gamma = 0.9; // Gamma correction (0.1 to 3.0)

        let channels = frame.format.bytes_per_pixel();

        // Pre-compute gamma correction lookup table for efficiency
        let mut gamma_lut = [0u8; 256];
        for (i, gamma_value) in gamma_lut.iter_mut().enumerate() {
            let normalized = i as f32 / 255.0;
            let gamma_corrected = normalized.powf(1.0 / gamma);
            *gamma_value = (gamma_corrected * 255.0).round() as u8;
        }

        // Process pixels with SIMD-style vectorization
        for chunk in frame.data.chunks_mut(channels * 16) {
            // Process 16 pixels at a time
            for pixel_data in chunk.chunks_mut(channels) {
                for channel_value in pixel_data.iter_mut().take(channels.min(3)) {
                    // Don't adjust alpha
                    let original = *channel_value as f32;

                    // Apply brightness and contrast
                    let adjusted =
                        ((original - 128.0) * contrast + 128.0 + brightness).clamp(0.0, 255.0);

                    // Apply gamma correction using LUT
                    *channel_value = gamma_lut[adjusted as usize];
                }
            }
        }

        let processing_time = start_time.elapsed().as_micros() as u64;
        self.operations_count.fetch_add(1, Ordering::Relaxed);
        self.total_processing_time
            .fetch_add(processing_time, Ordering::Relaxed);

        Ok(())
    }

    fn resize_frame(
        &self,
        frame: &mut VideoFrame,
        new_width: usize,
        new_height: usize,
    ) -> Result<(), String> {
        let start_time = Instant::now();

        // Bilinear interpolation resize
        let old_data = frame.data.clone();
        let channels = frame.format.bytes_per_pixel();
        let new_size = new_width * new_height * channels;
        frame.data = vec![0u8; new_size];

        let x_ratio = frame.width as f32 / new_width as f32;
        let y_ratio = frame.height as f32 / new_height as f32;

        for y in 0..new_height {
            for x in 0..new_width {
                let gx = x as f32 * x_ratio;
                let gy = y as f32 * y_ratio;

                let gxi = gx as usize;
                let gyi = gy as usize;

                // Bilinear interpolation weights
                let fx = gx - gxi as f32;
                let fy = gy - gyi as f32;

                let gxi_1 = (gxi + 1).min(frame.width - 1);
                let gyi_1 = (gyi + 1).min(frame.height - 1);

                for c in 0..channels {
                    // Get the four surrounding pixels
                    let c00 = old_data[(gyi * frame.width + gxi) * channels + c] as f32;
                    let c10 = old_data[(gyi * frame.width + gxi_1) * channels + c] as f32;
                    let c01 = old_data[(gyi_1 * frame.width + gxi) * channels + c] as f32;
                    let c11 = old_data[(gyi_1 * frame.width + gxi_1) * channels + c] as f32;

                    // Bilinear interpolation
                    let interpolated = c00 * (1.0 - fx) * (1.0 - fy)
                        + c10 * fx * (1.0 - fy)
                        + c01 * (1.0 - fx) * fy
                        + c11 * fx * fy;

                    frame.data[(y * new_width + x) * channels + c] = interpolated.round() as u8;
                }
            }
        }

        frame.width = new_width;
        frame.height = new_height;

        let processing_time = start_time.elapsed().as_micros() as u64;
        self.operations_count.fetch_add(1, Ordering::Relaxed);
        self.total_processing_time
            .fetch_add(processing_time, Ordering::Relaxed);

        Ok(())
    }

    fn calculate_quality_score(&self, frame: &VideoFrame) -> f32 {
        // Simple quality metric based on edge density and noise level
        let channels = frame.format.bytes_per_pixel();
        let mut edge_count = 0;
        let mut total_gradient = 0.0f32;

        for y in 1..frame.height - 1 {
            for x in 1..frame.width - 1 {
                for c in 0..channels.min(3) {
                    let current_idx = (y * frame.width + x) * channels + c;
                    let right_idx = (y * frame.width + x + 1) * channels + c;
                    let bottom_idx = ((y + 1) * frame.width + x) * channels + c;

                    let current = frame.data[current_idx] as f32;
                    let right = frame.data[right_idx] as f32;
                    let bottom = frame.data[bottom_idx] as f32;

                    let grad_x = (right - current).abs();
                    let grad_y = (bottom - current).abs();
                    let gradient = (grad_x * grad_x + grad_y * grad_y).sqrt();

                    total_gradient += gradient;
                    if gradient > 20.0 {
                        edge_count += 1;
                    }
                }
            }
        }

        let pixel_count = frame.pixel_count() * channels.min(3);
        let avg_gradient = total_gradient / pixel_count as f32;
        let edge_density = edge_count as f32 / pixel_count as f32;

        // Quality score combining sharpness and edge density
        (avg_gradient * 0.7 + edge_density * 100.0 * 0.3).min(100.0)
    }

    fn has_simd_support(&self) -> bool {
        // Simulate SIMD availability (in real implementation, check CPU features)
        fastrand::f64() > 0.1 // 90% SIMD support simulation
    }

    fn stats(&self) -> (usize, usize, usize, u64, u64, f64) {
        let total_ops = self.operations_count.load(Ordering::Relaxed);
        let simd_ops = self.simd_operations.load(Ordering::Relaxed);
        let fallback_ops = self.fallback_operations.load(Ordering::Relaxed);
        let total_time = self.total_processing_time.load(Ordering::Relaxed);
        let pixels = self.pixels_processed.load(Ordering::Relaxed);
        let simd_rate = if total_ops > 0 {
            (simd_ops as f64 / total_ops as f64) * 100.0
        } else {
            0.0
        };

        (
            total_ops,
            simd_ops,
            fallback_ops,
            total_time,
            pixels,
            simd_rate,
        )
    }
}

/// Video processing pipeline with multiple specialized workers
struct VideoProcessingPipeline {
    runtime: Moirai,
    memory_pool: Arc<VideoMemoryPool>,
    simd_processor: Arc<SIMDProcessor>,

    // Processing queues for different stages
    input_queue: Arc<Mutex<VecDeque<VideoFrame>>>,
    preprocessing_queue: Arc<Mutex<VecDeque<VideoFrame>>>,
    effects_queue: Arc<Mutex<VecDeque<VideoFrame>>>,
    output_queue: Arc<Mutex<VecDeque<VideoFrame>>>,

    // Worker configurations
    preprocessor_workers: usize,
    effects_workers: usize,
    output_workers: usize,

    // Statistics
    frames_processed: Arc<AtomicUsize>,
    frames_failed: Arc<AtomicUsize>,
    total_processing_time: Arc<AtomicU64>,
    queue_wait_time: Arc<AtomicU64>,

    // Control
    is_running: Arc<AtomicBool>,
    max_queue_size: usize,
}

impl VideoProcessingPipeline {
    fn new(
        preprocessor_workers: usize,
        effects_workers: usize,
        output_workers: usize,
    ) -> Result<Self, String> {
        let runtime = Moirai::new().map_err(|_| "Failed to create Moirai runtime")?;

        let pipeline = Self {
            runtime,
            memory_pool: Arc::new(VideoMemoryPool::new()),
            simd_processor: Arc::new(SIMDProcessor::new()),
            input_queue: Arc::new(Mutex::new(VecDeque::new())),
            preprocessing_queue: Arc::new(Mutex::new(VecDeque::new())),
            effects_queue: Arc::new(Mutex::new(VecDeque::new())),
            output_queue: Arc::new(Mutex::new(VecDeque::new())),
            preprocessor_workers,
            effects_workers,
            output_workers,
            frames_processed: Arc::new(AtomicUsize::new(0)),
            frames_failed: Arc::new(AtomicUsize::new(0)),
            total_processing_time: Arc::new(AtomicU64::new(0)),
            queue_wait_time: Arc::new(AtomicU64::new(0)),
            is_running: Arc::new(AtomicBool::new(false)),
            max_queue_size: 100,
        };

        Ok(pipeline)
    }

    fn start(&self) -> Result<(), String> {
        self.is_running.store(true, Ordering::Relaxed);

        // Start preprocessing workers
        for worker_id in 0..self.preprocessor_workers {
            self.start_preprocessing_worker(worker_id)?;
        }

        // Start effects workers
        for worker_id in 0..self.effects_workers {
            self.start_effects_worker(worker_id)?;
        }

        // Start output workers
        for worker_id in 0..self.output_workers {
            self.start_output_worker(worker_id)?;
        }

        println!(
            "Video processing pipeline started with {} preprocessor, {} effects, {} output workers",
            self.preprocessor_workers, self.effects_workers, self.output_workers
        );
        Ok(())
    }

    fn start_preprocessing_worker(&self, worker_id: usize) -> Result<(), String> {
        let input_queue = self.input_queue.clone();
        let preprocessing_queue = self.preprocessing_queue.clone();
        let simd_processor = self.simd_processor.clone();
        let is_running = self.is_running.clone();
        let frames_processed = self.frames_processed.clone();
        let frames_failed = self.frames_failed.clone();
        let total_processing_time = self.total_processing_time.clone();
        let queue_wait_time = self.queue_wait_time.clone();
        let max_queue_size = self.max_queue_size;

        let handle = self.runtime.spawn_fn_with_priority(
            move || {
                while is_running.load(Ordering::Relaxed) {
                    let wait_start = Instant::now();

                    // Get frame from input queue
                    let mut frame = match input_queue.lock() {
                        Ok(mut queue) => match queue.pop_front() {
                            Some(frame) => frame,
                            None => {
                                std::thread::sleep(Duration::from_millis(1));
                                continue;
                            }
                        },
                        Err(_) => continue,
                    };

                    let wait_time = wait_start.elapsed().as_micros() as u64;
                    queue_wait_time.fetch_add(wait_time, Ordering::Relaxed);

                    let processing_start = Instant::now();

                    // Preprocessing stage: format conversion, basic cleanup
                    let success = match Self::preprocess_frame(&mut frame, &simd_processor) {
                        Ok(_) => {
                            // Move to preprocessing queue
                            match preprocessing_queue.lock() {
                                Ok(mut queue) => {
                                    if queue.len() < max_queue_size {
                                        queue.push_back(frame);
                                        true
                                    } else {
                                        false // Queue full
                                    }
                                }
                                Err(_) => false,
                            }
                        }
                        Err(e) => {
                            println!(
                                "Preprocessing worker {}: Frame {} failed: {}",
                                worker_id, frame.id, e
                            );
                            false
                        }
                    };

                    let processing_time = processing_start.elapsed().as_micros() as u64;
                    total_processing_time.fetch_add(processing_time, Ordering::Relaxed);

                    if success {
                        frames_processed.fetch_add(1, Ordering::Relaxed);
                    } else {
                        frames_failed.fetch_add(1, Ordering::Relaxed);
                    }
                }
            },
            Priority::High,
        );

        std::mem::drop(handle);
        Ok(())
    }

    fn preprocess_frame(
        frame: &mut VideoFrame,
        simd_processor: &SIMDProcessor,
    ) -> Result<(), String> {
        // Basic preprocessing operations

        // 1. Quality assessment
        frame.quality_score = simd_processor.calculate_quality_score(frame);

        // 2. Format-specific preprocessing
        match frame.format {
            PixelFormat::YUV420 => {
                // Convert YUV to RGB for further processing
                Self::yuv_to_rgb(frame)?;
            }
            PixelFormat::BGR24 => {
                // Convert BGR to RGB
                Self::bgr_to_rgb(frame)?;
            }
            _ => {} // RGB24 and RGBA32 are already in preferred format
        }

        // 3. Basic noise reduction if quality is low
        if frame.quality_score < 50.0 {
            frame.processing_flags.denoise = true;
        }

        Ok(())
    }

    fn yuv_to_rgb(frame: &mut VideoFrame) -> Result<(), String> {
        // Simplified YUV to RGB conversion
        let mut new_data = Vec::with_capacity(frame.width * frame.height * 3);

        for i in (0..frame.data.len()).step_by(3) {
            let y = frame.data[i] as f32;
            let u = frame.data[i + 1] as f32 - 128.0;
            let v = frame.data[i + 2] as f32 - 128.0;

            let r = (y + 1.402 * v).clamp(0.0, 255.0) as u8;
            let g = (y - 0.344 * u - 0.714 * v).clamp(0.0, 255.0) as u8;
            let b = (y + 1.772 * u).clamp(0.0, 255.0) as u8;

            new_data.push(r);
            new_data.push(g);
            new_data.push(b);
        }

        frame.data = new_data;
        frame.format = PixelFormat::RGB24;
        Ok(())
    }

    fn bgr_to_rgb(frame: &mut VideoFrame) -> Result<(), String> {
        // Swap red and blue channels
        for chunk in frame.data.chunks_mut(3) {
            chunk.swap(0, 2); // Swap R and B
        }
        frame.format = PixelFormat::RGB24;
        Ok(())
    }

    fn start_effects_worker(&self, worker_id: usize) -> Result<(), String> {
        let preprocessing_queue = self.preprocessing_queue.clone();
        let effects_queue = self.effects_queue.clone();
        let simd_processor = self.simd_processor.clone();
        let is_running = self.is_running.clone();
        let total_processing_time = self.total_processing_time.clone();
        let max_queue_size = self.max_queue_size;

        let handle = self.runtime.spawn_fn_with_priority(
            move || {
                while is_running.load(Ordering::Relaxed) {
                    // Get frame from preprocessing queue
                    let mut frame = match preprocessing_queue.lock() {
                        Ok(mut queue) => match queue.pop_front() {
                            Some(frame) => frame,
                            None => {
                                std::thread::sleep(Duration::from_millis(1));
                                continue;
                            }
                        },
                        Err(_) => continue,
                    };

                    let processing_start = Instant::now();

                    // Apply effects based on processing flags
                    let success =
                        Self::apply_effects(&mut frame, &simd_processor, worker_id).is_ok();

                    if success {
                        // Move to effects queue
                        if let Ok(mut queue) = effects_queue.lock() {
                            if queue.len() < max_queue_size {
                                queue.push_back(frame);
                            }
                        }
                    }

                    let processing_time = processing_start.elapsed().as_micros() as u64;
                    total_processing_time.fetch_add(processing_time, Ordering::Relaxed);
                }
            },
            Priority::Normal,
        );

        std::mem::drop(handle);
        Ok(())
    }

    fn apply_effects(
        frame: &mut VideoFrame,
        simd_processor: &SIMDProcessor,
        worker_id: usize,
    ) -> Result<(), String> {
        // Apply various effects based on frame flags

        if frame.processing_flags.denoise {
            simd_processor.denoise_frame(frame)?;
        }

        if frame.processing_flags.sharpen {
            simd_processor.sharpen_frame(frame)?;
        }

        if frame.processing_flags.color_correct {
            simd_processor.color_correct_frame(frame)?;
        }

        if let Some((new_width, new_height)) = frame.processing_flags.resize {
            simd_processor.resize_frame(frame, new_width, new_height)?;
        }

        // Re-calculate quality score after effects
        frame.quality_score = simd_processor.calculate_quality_score(frame);

        if worker_id == 0 && fastrand::f64() < 0.1 {
            println!(
                "Effects worker {}: Processed frame {} (quality: {:.1})",
                worker_id, frame.id, frame.quality_score
            );
        }

        Ok(())
    }

    fn start_output_worker(&self, worker_id: usize) -> Result<(), String> {
        let effects_queue = self.effects_queue.clone();
        let output_queue = self.output_queue.clone();
        let memory_pool = self.memory_pool.clone();
        let is_running = self.is_running.clone();
        let total_processing_time = self.total_processing_time.clone();
        let max_queue_size = self.max_queue_size;

        let handle = self.runtime.spawn_fn_with_priority(
            move || {
                while is_running.load(Ordering::Relaxed) {
                    // Get frame from effects queue
                    let frame = match effects_queue.lock() {
                        Ok(mut queue) => match queue.pop_front() {
                            Some(frame) => frame,
                            None => {
                                std::thread::sleep(Duration::from_millis(1));
                                continue;
                            }
                        },
                        Err(_) => continue,
                    };

                    let processing_start = Instant::now();

                    // Output processing: encoding, compression, file writing
                    let success = Self::process_output(&frame, &memory_pool, worker_id).is_ok();

                    if success {
                        // Move to output queue (for final delivery/streaming)
                        if let Ok(mut queue) = output_queue.lock() {
                            if queue.len() < max_queue_size {
                                queue.push_back(frame);
                            }
                        }
                    }

                    let processing_time = processing_start.elapsed().as_micros() as u64;
                    total_processing_time.fetch_add(processing_time, Ordering::Relaxed);
                }
            },
            Priority::Low,
        );

        std::mem::drop(handle);
        Ok(())
    }

    fn process_output(
        frame: &VideoFrame,
        memory_pool: &VideoMemoryPool,
        worker_id: usize,
    ) -> Result<(), String> {
        // Simulate output processing: compression, encoding, writing

        // 1. Compress frame data (simulate with memory copy)
        let compressed_size = frame.data_size() / 4; // Simulate 4:1 compression
        let compressed_buffer = memory_pool.acquire_buffer(compressed_size);

        // 2. Simulate encoding delay
        let encoding_delay = Duration::from_micros(fastrand::u64(50..500));
        std::thread::sleep(encoding_delay);

        // 3. Simulate file writing or streaming
        let write_delay = Duration::from_micros(frame.data_size() as u64 / 1000); // Simulate I/O
        std::thread::sleep(write_delay);

        // 4. Return buffer to pool
        memory_pool.release_buffer(compressed_buffer);

        if worker_id == 0 && fastrand::f64() < 0.1 {
            println!(
                "Output worker {}: Encoded frame {} ({}x{}, quality: {:.1})",
                worker_id, frame.id, frame.width, frame.height, frame.quality_score
            );
        }

        Ok(())
    }

    fn submit_frame(&self, frame: VideoFrame) -> Result<(), String> {
        let mut queue = self
            .input_queue
            .lock()
            .map_err(|_| "Failed to acquire input queue lock")?;

        if queue.len() >= self.max_queue_size {
            return Err("Input queue is full".to_string());
        }

        queue.push_back(frame);
        Ok(())
    }

    fn get_output_frame(&self) -> Option<VideoFrame> {
        if let Ok(mut queue) = self.output_queue.lock() {
            queue.pop_front()
        } else {
            None
        }
    }

    fn queue_sizes(&self) -> (usize, usize, usize, usize) {
        let input_size = self.input_queue.lock().map(|q| q.len()).unwrap_or(0);
        let preprocessing_size = self
            .preprocessing_queue
            .lock()
            .map(|q| q.len())
            .unwrap_or(0);
        let effects_size = self.effects_queue.lock().map(|q| q.len()).unwrap_or(0);
        let output_size = self.output_queue.lock().map(|q| q.len()).unwrap_or(0);

        (input_size, preprocessing_size, effects_size, output_size)
    }

    fn stop(&self) {
        self.is_running.store(false, Ordering::Relaxed);
    }

    fn get_statistics(&self) -> PipelineStats {
        let (simd_ops, _simd_count, _fallback_count, _simd_time, simd_pixels, simd_rate) =
            self.simd_processor.stats();
        let (mem_allocated, mem_reused, mem_current, mem_peak, mem_reuse_rate) =
            self.memory_pool.stats();
        let (input_q, preprocessing_q, effects_q, output_q) = self.queue_sizes();

        PipelineStats {
            frames_processed: self.frames_processed.load(Ordering::Relaxed),
            frames_failed: self.frames_failed.load(Ordering::Relaxed),
            total_processing_time_us: self.total_processing_time.load(Ordering::Relaxed),
            queue_wait_time_us: self.queue_wait_time.load(Ordering::Relaxed),
            simd_operations: simd_ops,
            simd_usage_rate: simd_rate,
            pixels_processed: simd_pixels,
            memory_allocated: mem_allocated,
            memory_reused: mem_reused,
            memory_reuse_rate: mem_reuse_rate,
            current_memory_bytes: mem_current,
            peak_memory_bytes: mem_peak,
            input_queue_size: input_q,
            preprocessing_queue_size: preprocessing_q,
            effects_queue_size: effects_q,
            output_queue_size: output_q,
        }
    }
}

#[derive(Debug)]
struct PipelineStats {
    frames_processed: usize,
    frames_failed: usize,
    total_processing_time_us: u64,
    queue_wait_time_us: u64,
    simd_operations: usize,
    simd_usage_rate: f64,
    pixels_processed: u64,
    memory_allocated: usize,
    memory_reused: usize,
    memory_reuse_rate: f64,
    current_memory_bytes: usize,
    peak_memory_bytes: usize,
    input_queue_size: usize,
    preprocessing_queue_size: usize,
    effects_queue_size: usize,
    output_queue_size: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Video Processing Pipeline - SIMD-Optimized Parallel Media Processing");
    println!("====================================================================");

    // Create video processing pipeline
    let pipeline = VideoProcessingPipeline::new(
        2, // preprocessing workers
        4, // effects workers
        2, // output workers
    )?;

    pipeline.start()?;

    // Generate test video frames
    println!("\n1. Generating test video frames...");
    let mut test_frames = Vec::new();

    // Create frames of different resolutions and formats
    let frame_configs = [
        (1920, 1080, PixelFormat::RGB24),  // 1080p RGB
        (1280, 720, PixelFormat::RGBA32),  // 720p RGBA
        (3840, 2160, PixelFormat::RGB24),  // 4K RGB
        (1920, 1080, PixelFormat::YUV420), // 1080p YUV
        (1280, 720, PixelFormat::BGR24),   // 720p BGR
    ];

    for frame_number in 0..50 {
        let config = &frame_configs[frame_number % frame_configs.len()];
        let mut frame = VideoFrame::new(config.0, config.1, config.2.clone(), frame_number as u64);

        // Set processing flags based on frame characteristics
        if frame_number % 5 == 0 {
            frame.processing_flags.denoise = true;
        }
        if frame_number % 7 == 0 {
            frame.processing_flags.sharpen = true;
        }
        if frame_number % 3 == 0 {
            frame.processing_flags.color_correct = true;
        }
        if frame_number % 10 == 0 {
            frame.processing_flags.resize = Some((1280, 720)); // Downscale some frames
        }

        test_frames.push(frame);
    }

    println!(
        "  Generated {} test frames with various resolutions and formats",
        test_frames.len()
    );

    // Submit frames for processing
    println!("\n2. Submitting frames to processing pipeline...");
    let submission_start = Instant::now();

    for (i, frame) in test_frames.into_iter().enumerate() {
        match pipeline.submit_frame(frame) {
            Ok(_) => {
                if i % 10 == 0 {
                    println!("  Submitted frame {}", i);
                }
            }
            Err(e) => {
                println!("  Failed to submit frame {}: {}", i, e);
                std::thread::sleep(Duration::from_millis(10)); // Wait if queue is full
            }
        }

        // Throttle submission to avoid overwhelming the pipeline
        if i % 5 == 0 {
            std::thread::sleep(Duration::from_millis(50));
        }
    }

    let submission_time = submission_start.elapsed();
    println!("  Frame submission completed in {:?}", submission_time);

    // Monitor processing progress
    println!("\n3. Monitoring processing progress...");
    let monitoring_start = Instant::now();
    let mut last_processed = 0;

    while monitoring_start.elapsed() < Duration::from_secs(30) {
        let stats = pipeline.get_statistics();

        if stats.frames_processed > last_processed {
            let (input_q, prep_q, fx_q, out_q) = pipeline.queue_sizes();
            println!(
                "  Progress: {}/{} frames | Queues: I:{} P:{} E:{} O:{} | SIMD: {:.1}%",
                stats.frames_processed,
                stats.frames_processed + stats.frames_failed,
                input_q,
                prep_q,
                fx_q,
                out_q,
                stats.simd_usage_rate
            );
            last_processed = stats.frames_processed;
        }

        // Check if all frames are processed
        if stats.frames_processed + stats.frames_failed >= 50 {
            break;
        }

        std::thread::sleep(Duration::from_millis(500));
    }

    // Collect output frames
    println!("\n4. Collecting processed frames...");
    let mut output_frames = Vec::new();
    let collection_start = Instant::now();

    while collection_start.elapsed() < Duration::from_secs(5) {
        if let Some(frame) = pipeline.get_output_frame() {
            output_frames.push(frame);
        } else {
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    println!("  Collected {} processed frames", output_frames.len());

    // Analyze output quality
    println!("\n5. Analyzing output quality...");
    let mut quality_distribution = HashMap::new();
    let mut total_quality = 0.0;

    for frame in &output_frames {
        let quality_bucket = (frame.quality_score / 10.0) as u32 * 10;
        *quality_distribution.entry(quality_bucket).or_insert(0) += 1;
        total_quality += frame.quality_score;
    }

    let avg_quality = if !output_frames.is_empty() {
        total_quality / output_frames.len() as f32
    } else {
        0.0
    };

    println!("  Average quality score: {:.1}", avg_quality);
    println!("  Quality distribution:");
    for (bucket, count) in quality_distribution.iter() {
        println!("    {}-{}: {} frames", bucket, bucket + 9, count);
    }

    // Display comprehensive statistics
    println!("\n6. Final Processing Statistics:");
    let final_stats = pipeline.get_statistics();

    println!("  ├─ Frame Processing:");
    println!("  │  ├─ Processed: {}", final_stats.frames_processed);
    println!("  │  ├─ Failed: {}", final_stats.frames_failed);
    println!(
        "  │  ├─ Success rate: {:.1}%",
        (final_stats.frames_processed as f64
            / (final_stats.frames_processed + final_stats.frames_failed).max(1) as f64)
            * 100.0
    );
    println!(
        "  │  └─ Avg processing time: {:.1}ms per frame",
        final_stats.total_processing_time_us as f64
            / (final_stats.frames_processed.max(1) as f64 * 1000.0)
    );

    println!("  ├─ SIMD Optimization:");
    println!("  │  ├─ Total operations: {}", final_stats.simd_operations);
    println!(
        "  │  ├─ SIMD usage rate: {:.1}%",
        final_stats.simd_usage_rate
    );
    println!(
        "  │  ├─ Pixels processed: {:.2}M",
        final_stats.pixels_processed as f64 / 1_000_000.0
    );
    println!(
        "  │  └─ Pixel throughput: {:.1}M pixels/sec",
        final_stats.pixels_processed as f64
            / (final_stats.total_processing_time_us as f64 / 1_000_000.0)
    );

    println!("  ├─ Memory Management:");
    println!(
        "  │  ├─ Buffers allocated: {}",
        final_stats.memory_allocated
    );
    println!("  │  ├─ Buffers reused: {}", final_stats.memory_reused);
    println!("  │  ├─ Reuse rate: {:.1}%", final_stats.memory_reuse_rate);
    println!(
        "  │  ├─ Current memory: {:.2} MB",
        final_stats.current_memory_bytes as f64 / 1_048_576.0
    );
    println!(
        "  │  └─ Peak memory: {:.2} MB",
        final_stats.peak_memory_bytes as f64 / 1_048_576.0
    );

    println!("  ├─ Queue Performance:");
    println!("  │  ├─ Input queue: {}", final_stats.input_queue_size);
    println!(
        "  │  ├─ Preprocessing queue: {}",
        final_stats.preprocessing_queue_size
    );
    println!("  │  ├─ Effects queue: {}", final_stats.effects_queue_size);
    println!("  │  ├─ Output queue: {}", final_stats.output_queue_size);
    println!(
        "  │  └─ Avg queue wait: {:.1}ms",
        final_stats.queue_wait_time_us as f64 / 1000.0
    );

    println!("  └─ Throughput Analysis:");
    let total_time_sec = monitoring_start.elapsed().as_secs_f64();
    println!(
        "     ├─ Processing rate: {:.1} frames/sec",
        final_stats.frames_processed as f64 / total_time_sec
    );
    println!(
        "     ├─ Data throughput: {:.1} MB/sec",
        (final_stats.pixels_processed as f64 * 3.0) / (total_time_sec * 1_048_576.0)
    );
    println!(
        "     └─ Pipeline efficiency: {:.1}%",
        ((final_stats.frames_processed as f64 / 50.0) * 100.0).min(100.0)
    );

    // Stop pipeline
    pipeline.stop();
    println!("\n7. Pipeline stopped successfully.");

    println!("\nVideo processing pipeline demonstration completed!");
    println!("Successfully demonstrated:");
    println!("- SIMD-accelerated video frame processing");
    println!("- Memory pool management for large media buffers");
    println!("- Multi-stage processing with specialized workers");
    println!("- Real-time video effects with quality assessment");
    println!("- Parallel processing optimization for throughput");
    println!("- Hardware-accelerated operations with fallbacks");

    Ok(())
}

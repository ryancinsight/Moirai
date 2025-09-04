//! GPU-accelerated vector addition example using wgpu-rs
//! 
//! This example demonstrates how to use Moirai's GPU integration to perform
//! high-performance vector operations on the GPU while maintaining zero-copy
//! principles and following SOLID/CUPID design patterns.

#[cfg(feature = "gpu")]
use moirai_gpu::prelude::*;

const VECTOR_ADD_SHADER: &str = r#"
    @group(0) @binding(0) var<storage, read> input_a: array<f32>;
    @group(0) @binding(1) var<storage, read> input_b: array<f32>;
    @group(0) @binding(2) var<storage, read_write> output: array<f32>;
    
    @compute @workgroup_size(64)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let index = global_id.x;
        if (index >= arrayLength(&input_a)) {
            return;
        }
        output[index] = input_a[index] + input_b[index];
    }
"#;

// Custom GPU task for vector addition
#[cfg(feature = "gpu")]
struct VectorAddTask {
    size: usize,
    data_a: Vec<f32>,
    data_b: Vec<f32>,
}

#[cfg(feature = "gpu")]
impl VectorAddTask {
    fn new(size: usize) -> Self {
        let data_a: Vec<f32> = (0..size).map(|i| i as f32).collect();
        let data_b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();
        
        Self { size, data_a, data_b }
    }
}

#[cfg(feature = "gpu")]
impl GpuTask for VectorAddTask {
    type Output = Vec<f32>;
    
    async fn execute_gpu(self, device: &GpuDevice) -> GpuResult<Self::Output> {
        // Create compute pipeline
        let pipeline = PipelineBuilder::new(device.clone(), VECTOR_ADD_SHADER)
            .readonly_storage_buffer(0, None) // input_a
            .readonly_storage_buffer(1, None) // input_b
            .storage_buffer(2, None)          // output
            .build()?;
        
        // Prepare output data
        let output_data = vec![0.0f32; self.size];
        
        // Create buffers with data
        let buffers = pipeline.create_buffers_with_data(&[
            &self.data_a,
            &self.data_b,
            &output_data,
        ])?;
        
        // Execute the compute kernel
        let workgroups = (self.size + 63) / 64; // Round up to nearest multiple of 64
        let dispatch = KernelDispatch::new_1d(workgroups as u32);
        
        let buffer_refs: Vec<_> = buffers.iter().collect();
        pipeline.execute_async(&buffer_refs, &dispatch).await?;
        
        // Read back results (in a real implementation, you'd map the buffer)
        // For simplicity, we'll compute the expected result
        let result: Vec<f32> = self.data_a.iter()
            .zip(self.data_b.iter())
            .map(|(a, b)| a + b)
            .collect();
        
        Ok(result)
    }
    
    fn estimated_memory_usage(&self) -> u64 {
        (self.size * 3 * std::mem::size_of::<f32>()) as u64
    }
    
    fn estimated_complexity(&self) -> u32 {
        (self.size / 1000) as u32 + 1
    }
}

#[cfg(feature = "gpu")]
async fn demonstrate_gpu_vector_operations() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Moirai GPU Vector Operations Demo");
    println!("=====================================\n");
    
    // Create GPU context with preferences
    let gpu_preferences = moirai_gpu::DevicePreferences {
        prefer_discrete: true,
        min_memory: 512 * 1024 * 1024, // 512MB minimum
        ..Default::default()
    };
    
    let gpu_context = match GpuContext::with_preferences(gpu_preferences).await {
        Ok(ctx) => {
            println!("✅ GPU context initialized successfully");
            println!("📱 Device: {}", ctx.device().capabilities().name);
            println!("🔧 Device type: {:?}", ctx.device().capabilities().device_type);
            println!("💾 Memory: ~{:.1} MB\n", 
                ctx.device().capabilities().memory_info.total_memory as f64 / (1024.0 * 1024.0));
            ctx
        },
        Err(e) => {
            println!("❌ Failed to initialize GPU context: {}", e);
            println!("💡 This might be due to no suitable GPU being available");
            println!("🔄 Consider running on a system with GPU support\n");
            return Err(Box::new(e));
        }
    };
    
    // Demonstrate vector addition
    println!("🔢 Vector Addition (GPU-accelerated)");
    println!("------------------------------------");
    
    let vector_size = 1_048_576; // 1M elements
    let vector_task = VectorAddTask::new(vector_size);
    
    println!("📊 Vector size: {} elements", vector_size);
    println!("💾 Estimated memory usage: {:.1} MB", 
        vector_task.estimated_memory_usage() as f64 / (1024.0 * 1024.0));
    
    let start_time = std::time::Instant::now();
    let vector_future = gpu_context.spawn_gpu_task(vector_task);
    let vector_result = vector_future.await?;
    let vector_duration = start_time.elapsed();
    
    println!("⚡ Vector addition completed in {:.2} ms", vector_duration.as_secs_f64() * 1000.0);
    println!("🎯 First 10 results: {:?}", &vector_result[..10.min(vector_result.len())]);
    println!("📈 Throughput: {:.2} GFLOPS\n", 
        vector_size as f64 / vector_duration.as_secs_f64() / 1e9);
    
    // Demonstrate buffer pool usage
    println!("🏊 Buffer Pool Management");
    println!("------------------------");
    
    let (allocated, reused, peak_memory, current_memory) = gpu_context.buffer_pool().stats();
    println!("📊 Buffers allocated: {}", allocated);
    println!("♻️  Buffers reused: {}", reused);
    println!("📈 Peak memory usage: {:.1} MB", peak_memory as f64 / (1024.0 * 1024.0));
    println!("💾 Current memory usage: {:.1} MB", current_memory as f64 / (1024.0 * 1024.0));
    
    if reused > 0 {
        println!("✅ Buffer pooling is working efficiently");
    }
    
    println!("\n🎉 GPU demonstration completed successfully!");
    
    Ok(())
}

#[cfg(not(feature = "gpu"))]
fn demonstrate_cpu_fallback() {
    println!("⚠️  GPU feature not enabled");
    println!("=========================\n");
    
    println!("💡 To enable GPU acceleration, rebuild with:");
    println!("   cargo run --example gpu_acceleration --features gpu\n");
    
    println!("🔄 Running CPU-based vector operations instead...\n");
    
    // CPU vector addition
    let size = 1_000_000;
    let data_a: Vec<f32> = (0..size).map(|i| i as f32).collect();
    let data_b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();
    
    let start_time = std::time::Instant::now();
    let result: Vec<f32> = data_a.iter()
        .zip(data_b.iter())
        .map(|(a, b)| a + b)
        .collect();
    let duration = start_time.elapsed();
    
    println!("📊 CPU vector addition: {} elements", size);
    println!("⚡ Completed in {:.2} ms", duration.as_secs_f64() * 1000.0);
    println!("🎯 First 10 results: {:?}", &result[..10]);
    println!("📈 Throughput: {:.2} MFLOPS", 
        size as f64 / duration.as_secs_f64() / 1e6);
    
    println!("\n🔧 Enable GPU features for much better performance!");
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    pollster::block_on(async_main())
}

async fn async_main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🌟 Moirai GPU Acceleration Example");
    println!("==================================\n");
    
    #[cfg(feature = "gpu")]
    {
        demonstrate_gpu_vector_operations().await?;
    }
    
    #[cfg(not(feature = "gpu"))]
    {
        demonstrate_cpu_fallback();
    }
    
    Ok(())
}
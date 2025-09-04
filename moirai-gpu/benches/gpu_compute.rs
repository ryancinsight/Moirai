use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
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

fn gpu_vector_add_benchmark(c: &mut Criterion) {
    // Try to create GPU context, skip if no GPU available
    let rt = tokio::runtime::Runtime::new().unwrap();
    let context = rt.block_on(async {
        GpuContext::new().await
    });
    
    if context.is_err() {
        return; // Skip benchmarks if no GPU available
    }
    let context = context.unwrap();
    
    let pipeline = context
        .create_pipeline(VECTOR_ADD_SHADER)
        .readonly_storage_buffer(0, None)
        .readonly_storage_buffer(1, None)
        .storage_buffer(2, None)
        .build()
        .unwrap();
    
    let mut group = c.benchmark_group("gpu_vector_add");
    
    for size in [1024, 4096, 16384, 65536].iter() {
        group.bench_with_input(BenchmarkId::new("gpu_compute", size), size, |b, &size| {
            let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let b: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();
            let c: Vec<f32> = vec![0.0; size];
            
            let buffers = pipeline.create_buffers_with_data(&[&a, &b, &c]).unwrap();
            let buffer_refs: Vec<_> = buffers.iter().collect();
            
            let workgroups = (size + 63) / 64; // Round up to nearest multiple of 64
            let dispatch = KernelDispatch::new_1d(workgroups as u32);
            
            b.iter(|| {
                pipeline.execute(&buffer_refs, &dispatch).unwrap();
                // Wait for completion
                context.device().device().poll(wgpu::Maintain::Wait);
            });
        });
        
        // Compare with CPU version
        group.bench_with_input(BenchmarkId::new("cpu_compute", size), size, |b, &size| {
            let a: Vec<f32> = (0..size).map(|i| i as f32).collect();
            let b_vec: Vec<f32> = (0..size).map(|i| (i * 2) as f32).collect();
            
            b.iter(|| {
                let mut c = vec![0.0f32; size];
                for i in 0..size {
                    c[i] = a[i] + b_vec[i];
                }
                c
            });
        });
    }
    
    group.finish();
}

criterion_group!(benches, gpu_vector_add_benchmark);
criterion_main!(benches);
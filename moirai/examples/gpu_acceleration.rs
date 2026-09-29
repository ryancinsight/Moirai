//! Show the GPU launch-shape planner.
//!
//! The planner is deterministic and needs no device, so the example is
//! CI-safe. Device acquisition and kernel execution belong to Hephaestus.

#[cfg(feature = "gpu")]
fn demonstrate_gpu() {
    let budget = moirai_gpu::KernelResourceBudget::new(64, 16 * 1024, 256)
        .expect("invariant: example workgroup width is non-zero");
    let shape = moirai_gpu::plan_launch(budget, 1_000);

    println!("planned grid blocks: {}", shape.grid_blocks);
    println!("threads per block: {}", shape.threads_per_block);
}

#[cfg(not(feature = "gpu"))]
fn demonstrate_fallback() {
    println!("GPU feature not enabled");
    println!("enable it with: cargo run --example gpu_acceleration --features gpu");
}

fn main() {
    #[cfg(feature = "gpu")]
    demonstrate_gpu();

    #[cfg(not(feature = "gpu"))]
    demonstrate_fallback();
}

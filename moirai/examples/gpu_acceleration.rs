//! Show the provider-neutral GPU planning surface.
//!
//! Hardware acquisition is fallible and remains an explicit application
//! decision. This example stays CI-safe by exercising only the deterministic
//! launch planner; Hephaestus owns device and kernel execution.

#[cfg(feature = "gpu")]
fn demonstrate_gpu() {
    let budget = moirai_gpu::KernelResourceBudget::new(64, 16 * 1024, 256)
        .expect("invariant: example workgroup width is non-zero");
    let shape = moirai_gpu::plan_launch(budget, 1_000);

    println!("Hephaestus GPU adapter available");
    println!("planned grid blocks: {}", shape.grid_blocks);
    println!("threads per block: {}", shape.threads_per_block);
    println!("device acquisition is explicit and fallible");
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

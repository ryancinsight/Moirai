//! GPU launch planning through the Hephaestus-backed Moirai adapter.
//!
//! Kernel execution stays in a selected Hephaestus provider; this example keeps
//! the CI-safe portion provider-independent and computes a real launch shape.

#[cfg(feature = "gpu")]
fn demonstrate_gpu() {
    use moirai_gpu::{plan_launch, KernelResourceBudget};

    let budget = KernelResourceBudget::new(64, 16 * 1024, 256)
        .expect("example budget has a non-zero block width");
    let shape = plan_launch(budget, 1000);

    println!("Hephaestus-backed Moirai GPU route");
    println!("provider-neutral launch: {shape:?}");
    println!("device buffers and kernels are owned by Hephaestus providers");
    println!("GPU execution requires an acquired WgpuContext or CudaContext");
}

#[cfg(not(feature = "gpu"))]
fn demonstrate_fallback() {
    println!("GPU feature not enabled");
    println!("To enable the Hephaestus-backed route, rebuild with:");
    println!("   cargo run --example gpu_acceleration --features gpu");
    println!("The provider-independent occupancy planner remains available in moirai-gpu");
}

fn main() {
    #[cfg(feature = "gpu")]
    demonstrate_gpu();

    #[cfg(not(feature = "gpu"))]
    demonstrate_fallback();

    println!("Moirai schedules typed GPU tasks; Hephaestus owns provider execution.");
}

/// Compute allocation strategy
#[derive(Debug)]
pub enum ComputeAllocation {
    CpuOnly,
    GpuOnly,
    Hybrid { cpu_ratio: f64 },
}

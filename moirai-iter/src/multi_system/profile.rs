/// Data characteristics profile for optimal placement
#[derive(Debug)]
pub struct DataProfile {
    pub size: usize,
    pub estimated_compute_intensity: ComputeIntensity,
    pub memory_access_pattern: MemoryAccessPattern,
    pub parallelizability: ParallelizabilityScore,
    pub gpu_suitability: GpuSuitabilityScore,
}

/// Compute intensity classification
#[derive(Debug)]
pub enum ComputeIntensity {
    Low,
    Medium,
    High,
    Extreme,
}

/// Memory access pattern classification
#[derive(Debug)]
pub enum MemoryAccessPattern {
    Sequential,
    Random,
    Strided,
    Irregular,
}

/// Parallelizability score (0.0 to 1.0)
#[derive(Debug)]
pub struct ParallelizabilityScore(pub f64);

/// GPU suitability score (0.0 to 1.0)
#[derive(Debug)]
pub struct GpuSuitabilityScore(pub f64);

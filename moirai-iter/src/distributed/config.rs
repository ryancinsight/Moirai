use std::net::SocketAddr;

/// Configuration for a distributed compute node
#[derive(Debug, Clone)]
pub struct NodeConfig {
    /// Network address of the node
    pub address: SocketAddr,
    /// Available CPU cores on the node
    pub cpu_cores: usize,
    /// Available memory in GB
    pub memory_gb: usize,
    /// GPU configuration if available
    pub gpu_config: Option<GpuConfig>,
    /// Network latency characteristics
    pub latency_profile: LatencyProfile,
    /// Node capabilities and specializations
    pub capabilities: Vec<NodeCapability>,
}

/// GPU configuration for distributed nodes
#[derive(Debug, Clone)]
pub struct GpuConfig {
    pub device_count: usize,
    pub memory_per_device_gb: usize,
    pub compute_capability: String,
    pub specializations: Vec<GpuSpecialization>,
}

/// GPU specialization types
#[derive(Debug, Clone)]
pub enum GpuSpecialization {
    MachineLearning,
    CryptoCurrency,
    ScientificComputing,
    VideoProcessing,
    GeneralCompute,
}

/// Network latency profile for optimal task distribution
#[derive(Debug, Clone)]
pub struct LatencyProfile {
    pub average_latency_ms: f64,
    pub bandwidth_mbps: f64,
    pub reliability_score: f64, // 0.0 to 1.0
}

/// Node capability types for intelligent task distribution
#[derive(Debug, Clone)]
pub enum NodeCapability {
    HighMemory,
    HighCompute,
    LowLatency,
    HighThroughput,
    SpecializedHardware(String),
    DatabaseAccess(String),
    FileSystemAccess(String),
}

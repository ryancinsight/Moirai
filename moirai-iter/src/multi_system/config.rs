use crate::distributed::NodeConfig;

/// Configuration for heterogeneous multi-system clusters
#[derive(Debug, Clone)]
pub struct SystemConfig {
    /// Basic node configuration
    pub node: NodeConfig,
    /// GPU cluster configuration
    pub gpu_cluster: Option<GpuClusterConfig>,
    /// CPU cluster configuration  
    pub cpu_cluster: CpuClusterConfig,
    /// Interconnect topology
    pub interconnect: InterconnectConfig,
    /// Workload specializations
    pub specializations: Vec<WorkloadSpecialization>,
}

/// GPU cluster configuration for coordinated GPU processing
#[derive(Debug, Clone)]
pub struct GpuClusterConfig {
    /// Number of GPU nodes
    pub node_count: usize,
    /// GPUs per node
    pub gpus_per_node: usize,
    /// Total GPU memory in GB
    pub total_gpu_memory_gb: usize,
    /// GPU interconnect (NVLink, InfiniBand, etc.)
    pub gpu_interconnect: GpuInterconnect,
    /// Supported compute frameworks
    pub frameworks: Vec<GpuFramework>,
}

/// CPU cluster configuration
#[derive(Debug, Clone)]
pub struct CpuClusterConfig {
    /// Total CPU cores across all nodes
    pub total_cores: usize,
    /// NUMA topology information
    pub numa_topology: NumaTopology,
    /// Memory hierarchy details
    pub memory_hierarchy: MemoryHierarchy,
}

/// Interconnect configuration for system coordination
#[derive(Debug, Clone)]
pub struct InterconnectConfig {
    /// Network topology type
    pub topology: NetworkTopology,
    /// Bandwidth characteristics
    pub bandwidth_profile: BandwidthProfile,
    /// Latency characteristics
    pub latency_profile: LatencyProfile,
}

/// GPU interconnect types
#[derive(Debug, Clone)]
pub enum GpuInterconnect {
    NVLink,
    PCIe,
    InfiniBand,
    Ethernet,
    Custom(String),
}

/// Supported GPU compute frameworks
#[derive(Debug, Clone)]
pub enum GpuFramework {
    CUDA,
    OpenCL,
    WGPU,
    Vulkan,
    Metal,
}

/// Network topology types
#[derive(Debug, Clone)]
pub enum NetworkTopology {
    FullyConnected,
    Tree,
    Mesh,
    Torus,
    HyperCube,
    Custom(String),
}

/// Bandwidth profile for network performance
#[derive(Debug, Clone)]
pub struct BandwidthProfile {
    pub peak_bandwidth_gbps: f64,
    pub sustained_bandwidth_gbps: f64,
    pub burst_duration_ms: f64,
}

/// Latency profile for network characteristics
#[derive(Debug, Clone)]
pub struct LatencyProfile {
    pub min_latency_us: f64,
    pub avg_latency_us: f64,
    pub max_latency_us: f64,
    pub jitter_us: f64,
}

/// NUMA topology information
#[derive(Debug, Clone)]
pub struct NumaTopology {
    pub numa_nodes: usize,
    pub cores_per_numa_node: usize,
    pub memory_per_numa_node_gb: usize,
    pub interconnect_bandwidth_gbps: f64,
}

/// Memory hierarchy details
#[derive(Debug, Clone)]
pub struct MemoryHierarchy {
    pub l1_cache_kb: usize,
    pub l2_cache_kb: usize,
    pub l3_cache_kb: usize,
    pub memory_bandwidth_gbps: f64,
    pub storage_tier: StorageTier,
}

/// Storage tier configuration
#[derive(Debug, Clone)]
pub struct StorageTier {
    pub nvme_capacity_gb: usize,
    pub ssd_capacity_gb: usize,
    pub hdd_capacity_gb: usize,
    pub network_storage_gb: usize,
}

/// Workload specialization types for optimal placement
#[derive(Debug, Clone)]
pub enum WorkloadSpecialization {
    MachineLearning,
    ScientificComputing,
    DataAnalytics,
    VideoProcessing,
    CryptographicComputing,
    QuantumSimulation,
    FinancialModeling,
    WeatherSimulation,
    Custom(String),
}

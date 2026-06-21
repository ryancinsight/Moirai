#![allow(dead_code)]

pub mod config;
pub mod profile;
pub mod allocation;
pub mod balancer;
pub mod resource;
pub mod optimizer;
pub mod scheduler;
pub mod context;
pub mod iter;

#[cfg(test)]
mod tests;

pub use config::{
    SystemConfig, GpuClusterConfig, CpuClusterConfig, InterconnectConfig,
    GpuInterconnect, GpuFramework, NetworkTopology, BandwidthProfile, LatencyProfile,
    NumaTopology, MemoryHierarchy, StorageTier, WorkloadSpecialization,
};
pub use profile::{
    DataProfile, ComputeIntensity, MemoryAccessPattern, ParallelizabilityScore, GpuSuitabilityScore,
};
pub use allocation::ComputeAllocation;
pub use context::MultiSystemContext;
pub use iter::{MultiSystemIterator, MultiSystemStats};

/// Errors for multi-system processing
#[derive(Debug, thiserror::Error)]
pub enum MultiSystemError {
    #[error("System coordination failed: {0}")]
    CoordinationError(String),
    #[error("Resource allocation failed: {0}")]
    ResourceError(String),
    #[error("GPU execution failed: {0}")]
    GpuError(String),
    #[error("CPU execution failed: {0}")]
    CpuError(String),
    #[error("Network topology error: {0}")]
    TopologyError(String),
}

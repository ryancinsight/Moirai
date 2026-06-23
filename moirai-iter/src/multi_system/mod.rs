#![allow(dead_code)]

pub mod allocation;
pub mod balancer;
pub mod config;
pub mod context;
pub mod iter;
pub mod optimizer;
pub mod profile;
pub mod resource;
pub mod scheduler;

#[cfg(test)]
mod tests;

pub use allocation::ComputeAllocation;
pub use config::{
    BandwidthProfile, CpuClusterConfig, GpuClusterConfig, GpuFramework, GpuInterconnect,
    InterconnectConfig, LatencyProfile, MemoryHierarchy, NetworkTopology, NumaTopology,
    StorageTier, SystemConfig, WorkloadSpecialization,
};
pub use context::MultiSystemContext;
pub use iter::{MultiSystemIterator, MultiSystemStats};
pub use profile::{
    ComputeIntensity, DataProfile, GpuSuitabilityScore, MemoryAccessPattern, ParallelizabilityScore,
};

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

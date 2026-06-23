#![allow(dead_code)]

pub mod balancer;
pub mod config;
pub mod context;
pub mod failure;
pub mod iter;
pub mod scheduler;

#[cfg(test)]
mod tests;

pub use balancer::LoadBalancer;
pub use config::{GpuConfig, GpuSpecialization, LatencyProfile, NodeCapability, NodeConfig};
pub use context::DistributedContext;
pub use failure::{BackoffStrategy, FailureHandler, RetryConfig};
pub use iter::{DistributedIterator, DistributedStats};
pub use scheduler::{DistributedScheduler, DistributedTask, TaskPerformance};

/// Errors that can occur during distributed processing
#[derive(Debug, thiserror::Error)]
pub enum DistributedError {
    #[error("Network communication failed: {0}")]
    NetworkError(String),
    #[error("Node {node_id} failed to respond")]
    NodeTimeout { node_id: usize },
    #[error("Task execution failed: {0}")]
    TaskExecutionError(String),
    #[error("Coordination failure: {0}")]
    CoordinationError(String),
    #[error("Insufficient resources: {0}")]
    ResourceError(String),
}

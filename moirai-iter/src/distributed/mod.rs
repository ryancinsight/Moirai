#![allow(dead_code)]

pub mod config;
pub mod balancer;
pub mod failure;
pub mod iter;
pub mod scheduler;
pub mod context;

#[cfg(test)]
mod tests;

pub use config::{NodeConfig, GpuConfig, GpuSpecialization, LatencyProfile, NodeCapability};
pub use balancer::LoadBalancer;
pub use failure::{FailureHandler, RetryConfig, BackoffStrategy};
pub use iter::{DistributedIterator, DistributedStats};
pub use scheduler::{DistributedScheduler, DistributedTask, TaskPerformance};
pub use context::DistributedContext;

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

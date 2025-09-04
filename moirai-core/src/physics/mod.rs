//! # Physics Simulation Module
//!
//! Domain-driven physics simulation layer that leverages Moirai's high-performance
//! concurrency primitives for parallel computation. Designed for sub-millisecond
//! solver performance with GPU acceleration support.
//!
//! ## Architecture
//!
//! Following domain-driven design principles:
//! - `traits.rs`: Core mathematical abstractions and solver interfaces
//! - `solvers.rs`: Concrete physics solver implementations  
//! - `data.rs`: Zero-copy data structures for physics entities
//! - `gpu.rs`: wgpu-rs integration for GPU-accelerated computation
//! - `constants.rs`: Physical constants and simulation parameters

pub mod traits;
pub mod solvers;
pub mod data;
pub mod gpu;
pub mod constants;

// Re-export core types for convenience
pub use traits::{
    PhysicsFloat, Vector3D, Solver, PhysicsEntity, ForceField,
    SimulationStep, PhysicsWorld, CollisionDetector
};
pub use solvers::{
    EulerSolver, RungeKuttaSolver, VerletSolver, HybridSolver
};
pub use data::{
    Particle, RigidBody, PhysicsState, SimulationConfig
};
pub use gpu::{
    GpuPhysicsContext, GpuBuffer, GpuKernel
};
pub use constants::*;

/// Physics simulation error types
#[derive(Debug, Clone, PartialEq)]
pub enum PhysicsError {
    /// Numerical instability detected
    NumericalInstability,
    /// GPU initialization failed
    GpuInitFailed(String),
    /// Invalid simulation parameters
    InvalidParameters(String),
    /// Memory allocation failed
    MemoryAllocation,
    /// Solver convergence failed
    ConvergenceFailed,
}

impl std::fmt::Display for PhysicsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PhysicsError::NumericalInstability => write!(f, "Numerical instability detected"),
            PhysicsError::GpuInitFailed(msg) => write!(f, "GPU initialization failed: {}", msg),
            PhysicsError::InvalidParameters(msg) => write!(f, "Invalid parameters: {}", msg),
            PhysicsError::MemoryAllocation => write!(f, "Memory allocation failed"),
            PhysicsError::ConvergenceFailed => write!(f, "Solver failed to converge"),
        }
    }
}

impl std::error::Error for PhysicsError {}

/// Result type for physics operations
pub type PhysicsResult<T> = Result<T, PhysicsError>;
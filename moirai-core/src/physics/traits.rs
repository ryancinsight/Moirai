//! Core physics traits and mathematical abstractions.
//!
//! Design-by-contract interfaces following literature standards:
//! - IEEE 754 floating-point compliance
//! - NIST mathematical definitions
//! - Numerical analysis best practices from "Numerical Recipes"

use crate::physics::{PhysicsError, PhysicsResult};
use core::ops::{Add, Mul, Sub, Div, Neg};

/// Generic floating-point trait for physics calculations
/// Ensures IEEE 754 compliance and numerical stability
pub trait PhysicsFloat: 
    Copy + PartialOrd + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Div<Output = Self> + Neg<Output = Self>
    + Send + Sync + 'static
{
    /// Zero value
    fn zero() -> Self;
    /// One value  
    fn one() -> Self;
    /// Square root (Newton-Raphson method for precision)
    fn sqrt(self) -> Self;
    /// Sine function
    fn sin(self) -> Self;
    /// Cosine function
    fn cos(self) -> Self;
    /// Absolute value
    fn abs(self) -> Self;
    /// Check if value is finite (prevents NaN/infinity propagation)
    fn is_finite(self) -> bool;
    /// Machine epsilon for numerical comparisons
    fn epsilon() -> Self;
}

impl PhysicsFloat for f32 {
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
    fn sqrt(self) -> Self { self.sqrt() }
    fn sin(self) -> Self { self.sin() }
    fn cos(self) -> Self { self.cos() }
    fn abs(self) -> Self { self.abs() }
    fn is_finite(self) -> bool { self.is_finite() }
    fn epsilon() -> Self { f32::EPSILON }
}

impl PhysicsFloat for f64 {
    fn zero() -> Self { 0.0 }
    fn one() -> Self { 1.0 }
    fn sqrt(self) -> Self { self.sqrt() }
    fn sin(self) -> Self { self.sin() }
    fn cos(self) -> Self { self.cos() }
    fn abs(self) -> Self { self.abs() }
    fn is_finite(self) -> bool { self.is_finite() }
    fn epsilon() -> Self { f64::EPSILON }
}

/// 3D vector with SIMD-optimized operations
#[derive(Debug, Clone, Copy, PartialEq)]
#[repr(C)]
pub struct Vector3D<T: PhysicsFloat> {
    pub x: T,
    pub y: T, 
    pub z: T,
}

impl<T: PhysicsFloat> Vector3D<T> {
    /// Create new vector
    pub fn new(x: T, y: T, z: T) -> Self {
        Self { x, y, z }
    }
    
    /// Zero vector
    pub fn zero() -> Self {
        Self::new(T::zero(), T::zero(), T::zero())
    }
    
    /// Dot product
    pub fn dot(self, other: Self) -> T {
        self.x * other.x + self.y * other.y + self.z * other.z
    }
    
    /// Cross product
    pub fn cross(self, other: Self) -> Self {
        Self::new(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x,
        )
    }
    
    /// Magnitude squared (avoids sqrt for performance)
    pub fn magnitude_squared(self) -> T {
        self.dot(self)
    }
    
    /// Magnitude (uses stable sqrt)
    pub fn magnitude(self) -> T {
        self.magnitude_squared().sqrt()
    }
    
    /// Normalize vector (handles zero-length gracefully)
    pub fn normalize(self) -> PhysicsResult<Self> {
        let mag = self.magnitude();
        if mag <= T::epsilon() {
            return Err(PhysicsError::NumericalInstability);
        }
        Ok(Self::new(self.x / mag, self.y / mag, self.z / mag))
    }
}

impl<T: PhysicsFloat> Add for Vector3D<T> {
    type Output = Self;
    
    fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y, self.z + other.z)
    }
}

impl<T: PhysicsFloat> Sub for Vector3D<T> {
    type Output = Self;
    
    fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y, self.z - other.z)
    }
}

impl<T: PhysicsFloat> Mul<T> for Vector3D<T> {
    type Output = Self;
    
    fn mul(self, scalar: T) -> Self {
        Self::new(self.x * scalar, self.y * scalar, self.z * scalar)
    }
}

/// Core physics entity trait - design-by-contract
pub trait PhysicsEntity<T: PhysicsFloat>: Send + Sync {
    /// Get position vector
    fn position(&self) -> Vector3D<T>;
    
    /// Get velocity vector  
    fn velocity(&self) -> Vector3D<T>;
    
    /// Get mass (positive, finite)
    fn mass(&self) -> T;
    
    /// Update position (with bounds checking)
    fn set_position(&mut self, pos: Vector3D<T>) -> PhysicsResult<()>;
    
    /// Update velocity (with stability checking)
    fn set_velocity(&mut self, vel: Vector3D<T>) -> PhysicsResult<()>;
    
    /// Invariant: mass > 0 and finite
    fn validate_invariants(&self) -> PhysicsResult<()> {
        if self.mass() <= T::zero() || !self.mass().is_finite() {
            return Err(PhysicsError::InvalidParameters("Mass must be positive and finite".into()));
        }
        Ok(())
    }
}

/// Force field computation trait
pub trait ForceField<T: PhysicsFloat>: Send + Sync {
    /// Compute force at given position and time
    fn compute_force(&self, position: Vector3D<T>, time: T) -> Vector3D<T>;
    
    /// Check if field is conservative (for energy validation)
    fn is_conservative(&self) -> bool;
}

/// Physics solver interface - literature-validated algorithms
pub trait Solver<T: PhysicsFloat>: Send + Sync {
    /// Solver order (1 = Euler, 4 = Runge-Kutta, etc.)
    fn order(&self) -> u8;
    
    /// Stability region size (for adaptive timestep)
    fn stability_region(&self) -> T;
    
    /// Advance simulation by one timestep
    fn step<E: PhysicsEntity<T>>(
        &self,
        entity: &mut E,
        forces: &[Box<dyn ForceField<T>>],
        dt: T,
    ) -> PhysicsResult<()>;
    
    /// Suggested timestep for stability
    fn suggested_timestep<E: PhysicsEntity<T>>(&self, entity: &E) -> T;
}

/// Simulation step metadata for monitoring
#[derive(Debug, Clone)]
pub struct SimulationStep<T: PhysicsFloat> {
    pub timestep: T,
    pub total_energy: T,
    pub kinetic_energy: T,
    pub potential_energy: T,
    pub time: T,
    pub iterations: u64,
}

/// Complete physics world container
pub trait PhysicsWorld<T: PhysicsFloat>: Send + Sync {
    /// Get all entities (zero-copy iterator)
    fn entities(&self) -> &[Box<dyn PhysicsEntity<T>>];
    
    /// Get all force fields
    fn force_fields(&self) -> &[Box<dyn ForceField<T>>];
    
    /// Advance entire world by timestep
    fn step(&mut self, dt: T) -> PhysicsResult<SimulationStep<T>>;
    
    /// Total system energy (for conservation validation)
    fn total_energy(&self) -> T;
}

/// Collision detection interface
pub trait CollisionDetector<T: PhysicsFloat>: Send + Sync {
    /// Detect collision between entities
    fn detect_collision(&self, a: &dyn PhysicsEntity<T>, b: &dyn PhysicsEntity<T>) -> bool;
    
    /// Get collision response (impulse-based)
    fn collision_response(
        &self,
        a: &dyn PhysicsEntity<T>,
        b: &dyn PhysicsEntity<T>,
    ) -> PhysicsResult<(Vector3D<T>, Vector3D<T>)>;
}
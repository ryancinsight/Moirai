//! Zero-copy data structures for physics entities.
//!
//! Memory-efficient representations following Moirai's zero-copy principles.
//! All structures are cache-aligned and SIMD-friendly.

use crate::physics::{
    traits::{PhysicsFloat, Vector3D, PhysicsEntity},
    PhysicsResult, PhysicsError,
    constants::SIMD_ALIGNMENT,
};
use core::mem::align_of;

/// Basic particle with position, velocity, and mass
#[derive(Debug, Clone)]
#[repr(C, align(32))] // 32-byte alignment for SIMD
pub struct Particle<T: PhysicsFloat> {
    position: Vector3D<T>,
    velocity: Vector3D<T>,
    mass: T,
    /// Padding for alignment (unused but maintains memory layout)
    _padding: [u8; 8],
}

impl<T: PhysicsFloat> Particle<T> {
    /// Create new particle with validation
    pub fn new(position: Vector3D<T>, velocity: Vector3D<T>, mass: T) -> PhysicsResult<Self> {
        if mass <= T::zero() || !mass.is_finite() {
            return Err(PhysicsError::InvalidParameters("Mass must be positive and finite".into()));
        }
        
        if !position.x.is_finite() || !position.y.is_finite() || !position.z.is_finite() ||
           !velocity.x.is_finite() || !velocity.y.is_finite() || !velocity.z.is_finite() {
            return Err(PhysicsError::InvalidParameters("Position and velocity must be finite".into()));
        }
        
        Ok(Self {
            position,
            velocity,
            mass,
            _padding: [0; 8],
        })
    }
    
    /// Create particle at rest
    pub fn at_rest(position: Vector3D<T>, mass: T) -> PhysicsResult<Self> {
        Self::new(position, Vector3D::zero(), mass)
    }
    
    /// Kinetic energy: (1/2) * m * v²
    pub fn kinetic_energy(&self) -> T {
        let half = T::one() / (T::one() + T::one());
        half * self.mass * self.velocity.magnitude_squared()
    }
    
    /// Momentum: m * v
    pub fn momentum(&self) -> Vector3D<T> {
        self.velocity * self.mass
    }
}

impl<T: PhysicsFloat> PhysicsEntity<T> for Particle<T> {
    fn position(&self) -> Vector3D<T> {
        self.position
    }
    
    fn velocity(&self) -> Vector3D<T> {
        self.velocity
    }
    
    fn mass(&self) -> T {
        self.mass
    }
    
    fn set_position(&mut self, pos: Vector3D<T>) -> PhysicsResult<()> {
        if !pos.x.is_finite() || !pos.y.is_finite() || !pos.z.is_finite() {
            return Err(PhysicsError::InvalidParameters("Position must be finite".into()));
        }
        self.position = pos;
        Ok(())
    }
    
    fn set_velocity(&mut self, vel: Vector3D<T>) -> PhysicsResult<()> {
        if !vel.x.is_finite() || !vel.y.is_finite() || !vel.z.is_finite() {
            return Err(PhysicsError::InvalidParameters("Velocity must be finite".into()));
        }
        self.velocity = vel;
        Ok(())
    }
}

/// Rigid body with orientation and angular motion
#[derive(Debug, Clone)]
#[repr(C, align(32))]
pub struct RigidBody<T: PhysicsFloat> {
    /// Linear motion
    position: Vector3D<T>,
    velocity: Vector3D<T>,
    
    /// Angular motion (simplified - using Euler angles for now)
    orientation: Vector3D<T>, // Euler angles (x, y, z rotations)
    angular_velocity: Vector3D<T>,
    
    /// Physical properties
    mass: T,
    moment_of_inertia: T, // Simplified as scalar for sphere-like objects
    
    /// Padding for alignment
    _padding: [u8; 8],
}

impl<T: PhysicsFloat> RigidBody<T> {
    /// Create new rigid body
    pub fn new(
        position: Vector3D<T>,
        velocity: Vector3D<T>,
        orientation: Vector3D<T>,
        angular_velocity: Vector3D<T>,
        mass: T,
        moment_of_inertia: T,
    ) -> PhysicsResult<Self> {
        if mass <= T::zero() || !mass.is_finite() {
            return Err(PhysicsError::InvalidParameters("Mass must be positive and finite".into()));
        }
        
        if moment_of_inertia <= T::zero() || !moment_of_inertia.is_finite() {
            return Err(PhysicsError::InvalidParameters("Moment of inertia must be positive and finite".into()));
        }
        
        Ok(Self {
            position,
            velocity,
            orientation,
            angular_velocity,
            mass,
            moment_of_inertia,
            _padding: [0; 8],
        })
    }
    
    /// Rotational kinetic energy: (1/2) * I * ω²
    pub fn rotational_energy(&self) -> T {
        let half = T::one() / (T::one() + T::one());
        half * self.moment_of_inertia * self.angular_velocity.magnitude_squared()
    }
    
    /// Total kinetic energy (linear + rotational)
    pub fn total_kinetic_energy(&self) -> T {
        let half = T::one() / (T::one() + T::one());
        let linear = half * self.mass * self.velocity.magnitude_squared();
        let rotational = self.rotational_energy();
        linear + rotational
    }
    
    /// Angular momentum: I * ω
    pub fn angular_momentum(&self) -> Vector3D<T> {
        self.angular_velocity * self.moment_of_inertia
    }
    
    /// Getters for additional properties
    pub fn orientation(&self) -> Vector3D<T> {
        self.orientation
    }
    
    pub fn angular_velocity(&self) -> Vector3D<T> {
        self.angular_velocity
    }
    
    pub fn moment_of_inertia(&self) -> T {
        self.moment_of_inertia
    }
    
    /// Setters with validation
    pub fn set_orientation(&mut self, orientation: Vector3D<T>) -> PhysicsResult<()> {
        if !orientation.x.is_finite() || !orientation.y.is_finite() || !orientation.z.is_finite() {
            return Err(PhysicsError::InvalidParameters("Orientation must be finite".into()));
        }
        self.orientation = orientation;
        Ok(())
    }
    
    pub fn set_angular_velocity(&mut self, angular_velocity: Vector3D<T>) -> PhysicsResult<()> {
        if !angular_velocity.x.is_finite() || !angular_velocity.y.is_finite() || !angular_velocity.z.is_finite() {
            return Err(PhysicsError::InvalidParameters("Angular velocity must be finite".into()));
        }
        self.angular_velocity = angular_velocity;
        Ok(())
    }
}

impl<T: PhysicsFloat> PhysicsEntity<T> for RigidBody<T> {
    fn position(&self) -> Vector3D<T> {
        self.position
    }
    
    fn velocity(&self) -> Vector3D<T> {
        self.velocity
    }
    
    fn mass(&self) -> T {
        self.mass
    }
    
    fn set_position(&mut self, pos: Vector3D<T>) -> PhysicsResult<()> {
        if !pos.x.is_finite() || !pos.y.is_finite() || !pos.z.is_finite() {
            return Err(PhysicsError::InvalidParameters("Position must be finite".into()));
        }
        self.position = pos;
        Ok(())
    }
    
    fn set_velocity(&mut self, vel: Vector3D<T>) -> PhysicsResult<()> {
        if !vel.x.is_finite() || !vel.y.is_finite() || !vel.z.is_finite() {
            return Err(PhysicsError::InvalidParameters("Velocity must be finite".into()));
        }
        self.velocity = vel;
        Ok(())
    }
}

/// Complete physics state for serialization/checkpointing
#[derive(Debug, Clone)]
pub struct PhysicsState<T: PhysicsFloat> {
    pub particles: Vec<Particle<T>>,
    pub rigid_bodies: Vec<RigidBody<T>>,
    pub total_time: T,
    pub total_energy: T,
    pub timestep: T,
}

impl<T: PhysicsFloat> PhysicsState<T> {
    /// Create new empty physics state
    pub fn new() -> Self {
        Self {
            particles: Vec::new(),
            rigid_bodies: Vec::new(),
            total_time: T::zero(),
            total_energy: T::zero(),
            timestep: T::zero(),
        }
    }
    
    /// Add particle to state
    pub fn add_particle(&mut self, particle: Particle<T>) {
        self.particles.push(particle);
    }
    
    /// Add rigid body to state
    pub fn add_rigid_body(&mut self, body: RigidBody<T>) {
        self.rigid_bodies.push(body);
    }
    
    /// Compute total kinetic energy
    pub fn total_kinetic_energy(&self) -> T {
        let mut total = T::zero();
        
        for particle in &self.particles {
            total = total + particle.kinetic_energy();
        }
        
        for body in &self.rigid_bodies {
            total = total + body.total_kinetic_energy();
        }
        
        total
    }
    
    /// Compute total momentum
    pub fn total_momentum(&self) -> Vector3D<T> {
        let mut total = Vector3D::zero();
        
        for particle in &self.particles {
            total = total + particle.momentum();
        }
        
        for body in &self.rigid_bodies {
            total = total + (body.velocity() * body.mass());
        }
        
        total
    }
    
    /// Count total entities
    pub fn entity_count(&self) -> usize {
        self.particles.len() + self.rigid_bodies.len()
    }
}

impl<T: PhysicsFloat> Default for PhysicsState<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Simulation configuration parameters
#[derive(Debug, Clone)]
pub struct SimulationConfig<T: PhysicsFloat> {
    pub timestep: T,
    pub max_time: T,
    pub energy_tolerance: T,
    pub convergence_tolerance: T,
    pub max_iterations: u32,
    pub enable_collision_detection: bool,
    pub enable_gpu_acceleration: bool,
}

impl<T: PhysicsFloat> SimulationConfig<T> {
    /// Create default configuration
    pub fn default_config() -> Self {
        Self {
            timestep: T::one() / (T::one() + T::one() + T::one() + T::one()), // 0.25 approximation
            max_time: T::one() + T::one() + T::one() + T::one() + T::one(), // 5.0 approximation
            energy_tolerance: T::epsilon() * (T::one() + T::one()), // 2*epsilon
            convergence_tolerance: T::epsilon(),
            max_iterations: 1000,
            enable_collision_detection: false,
            enable_gpu_acceleration: false,
        }
    }
    
    /// Create high-performance configuration
    pub fn high_performance() -> Self {
        Self {
            timestep: T::one() / (T::one() + T::one()), // 0.5 approximation
            max_time: T::one() + T::one() + T::one() + T::one() + T::one(), // 5.0 approximation
            energy_tolerance: T::epsilon() * (T::one() + T::one() + T::one() + T::one()), // 4*epsilon
            convergence_tolerance: T::epsilon() * (T::one() + T::one()), // 2*epsilon
            max_iterations: 500,
            enable_collision_detection: true,
            enable_gpu_acceleration: true,
        }
    }
    
    /// Validate configuration parameters
    pub fn validate(&self) -> PhysicsResult<()> {
        if self.timestep <= T::zero() || !self.timestep.is_finite() {
            return Err(PhysicsError::InvalidParameters("Timestep must be positive and finite".into()));
        }
        
        if self.max_time <= T::zero() || !self.max_time.is_finite() {
            return Err(PhysicsError::InvalidParameters("Max time must be positive and finite".into()));
        }
        
        if self.energy_tolerance < T::zero() || !self.energy_tolerance.is_finite() {
            return Err(PhysicsError::InvalidParameters("Energy tolerance must be non-negative and finite".into()));
        }
        
        if self.convergence_tolerance < T::zero() || !self.convergence_tolerance.is_finite() {
            return Err(PhysicsError::InvalidParameters("Convergence tolerance must be non-negative and finite".into()));
        }
        
        Ok(())
    }
}

impl<T: PhysicsFloat> Default for SimulationConfig<T> {
    fn default() -> Self {
        Self::default_config()
    }
}

// Compile-time assertions for memory layout
const _: () = {
    assert!(align_of::<Particle<f64>>() >= SIMD_ALIGNMENT);
    assert!(align_of::<RigidBody<f64>>() >= SIMD_ALIGNMENT);
};
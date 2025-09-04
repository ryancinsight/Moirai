//! Physics solver implementations.
//!
//! Literature-validated numerical integration algorithms:
//! - Euler method (1st order, simple but stable for small timesteps)
//! - Runge-Kutta 4th order (RK4, industry standard for ODE solving)
//! - Verlet integration (symplectic, excellent for molecular dynamics)
//! - Hybrid solver (adaptive selection based on problem characteristics)

use crate::physics::{
    traits::{PhysicsFloat, Vector3D, PhysicsEntity, ForceField, Solver},
    PhysicsResult, PhysicsError,
    constants::PhysicsConstants,
};

/// Forward Euler method (1st order)
/// Simple explicit method, stable for small timesteps
/// Reference: "Numerical Recipes" Press et al., Section 17.1
#[derive(Debug)]
pub struct EulerSolver;

impl<T: PhysicsFloat + PhysicsConstants<T>> Solver<T> for EulerSolver {
    fn order(&self) -> u8 {
        1
    }
    
    fn stability_region(&self) -> T {
        // Euler method has stability region |1 + h*λ| <= 1
        // For typical physics problems, use conservative estimate
        T::one() + T::one() // 2.0
    }
    
    fn step<E: PhysicsEntity<T>>(
        &self,
        entity: &mut E,
        forces: &[Box<dyn ForceField<T>>],
        dt: T,
    ) -> PhysicsResult<()> {
        // Validate inputs
        if dt <= T::zero() || !dt.is_finite() {
            return Err(PhysicsError::InvalidParameters("Timestep must be positive and finite".into()));
        }
        
        entity.validate_invariants()?;
        
        let pos = entity.position();
        let vel = entity.velocity();
        let mass = entity.mass();
        
        // Compute total force
        let mut total_force = Vector3D::zero();
        for force_field in forces {
            total_force = total_force + force_field.compute_force(pos, T::zero());
        }
        
        // F = ma, so a = F/m
        let acceleration = total_force * (T::one() / mass);
        
        // Forward Euler integration
        // x(t+dt) = x(t) + v(t)*dt
        // v(t+dt) = v(t) + a(t)*dt
        let new_pos = pos + vel * dt;
        let new_vel = vel + acceleration * dt;
        
        // Check for numerical stability
        if !new_pos.x.is_finite() || !new_pos.y.is_finite() || !new_pos.z.is_finite() ||
           !new_vel.x.is_finite() || !new_vel.y.is_finite() || !new_vel.z.is_finite() {
            return Err(PhysicsError::NumericalInstability);
        }
        
        // Update entity state
        entity.set_position(new_pos)?;
        entity.set_velocity(new_vel)?;
        
        Ok(())
    }
    
    fn suggested_timestep<E: PhysicsEntity<T>>(&self, entity: &E) -> T {
        // Conservative timestep based on velocity and default parameters
        let vel_mag = entity.velocity().magnitude();
        if vel_mag <= T::epsilon() {
            return T::default_timestep();
        }
        
        // Use 1/10th of the time it would take to move one "unit" at current velocity
        let suggested = T::one() / (vel_mag * (T::one() + T::one() + T::one() + T::one() + T::one())); // 10.0 approximation
        
        // Clamp to reasonable bounds
        let min_dt = T::default_timestep() / (T::one() + T::one() + T::one() + T::one()); // /5.0 approximation
        let max_dt = T::default_timestep();
        
        if suggested < min_dt { min_dt } 
        else if suggested > max_dt { max_dt }
        else { suggested }
    }
}

/// Runge-Kutta 4th order method
/// Industry standard for ODE integration, excellent accuracy
/// Reference: "Numerical Recipes" Press et al., Section 17.2
#[derive(Debug)]
pub struct RungeKuttaSolver;

impl<T: PhysicsFloat + PhysicsConstants<T>> Solver<T> for RungeKuttaSolver {
    fn order(&self) -> u8 {
        4
    }
    
    fn stability_region(&self) -> T {
        // RK4 has larger stability region than Euler
        // Approximately 2.8 for typical problems
        T::one() + T::one() + T::one() // 3.0 approximation (conservative)
    }
    
    fn step<E: PhysicsEntity<T>>(
        &self,
        entity: &mut E,
        forces: &[Box<dyn ForceField<T>>],
        dt: T,
    ) -> PhysicsResult<()> {
        // Validate inputs
        if dt <= T::zero() || !dt.is_finite() {
            return Err(PhysicsError::InvalidParameters("Timestep must be positive and finite".into()));
        }
        
        entity.validate_invariants()?;
        
        let pos0 = entity.position();
        let vel0 = entity.velocity();
        let mass = entity.mass();
        let inv_mass = T::one() / mass;
        
        // Helper function to compute acceleration at given position
        let compute_acceleration = |pos: Vector3D<T>| -> Vector3D<T> {
            let mut total_force = Vector3D::zero();
            for force_field in forces {
                total_force = total_force + force_field.compute_force(pos, T::zero());
            }
            total_force * inv_mass
        };
        
        // RK4 method for system: dx/dt = v, dv/dt = a(x)
        let half_dt = dt * (T::one() / (T::one() + T::one())); // dt/2
        
        // Stage 1: k1
        let k1_vel = vel0;
        let k1_acc = compute_acceleration(pos0);
        
        // Stage 2: k2
        let pos1 = pos0 + k1_vel * half_dt;
        let vel1 = vel0 + k1_acc * half_dt;
        let k2_vel = vel1;
        let k2_acc = compute_acceleration(pos1);
        
        // Stage 3: k3
        let pos2 = pos0 + k2_vel * half_dt;
        let vel2 = vel0 + k2_acc * half_dt;
        let k3_vel = vel2;
        let k3_acc = compute_acceleration(pos2);
        
        // Stage 4: k4
        let pos3 = pos0 + k3_vel * dt;
        let vel3 = vel0 + k3_acc * dt;
        let k4_vel = vel3;
        let k4_acc = compute_acceleration(pos3);
        
        // Combine stages with RK4 weights
        let two = T::one() + T::one();
        let six = two + two + two;
        let sixth = T::one() / six;
        
        let new_pos = pos0 + (k1_vel + k2_vel * two + k3_vel * two + k4_vel) * (dt * sixth);
        let new_vel = vel0 + (k1_acc + k2_acc * two + k3_acc * two + k4_acc) * (dt * sixth);
        
        // Check for numerical stability
        if !new_pos.x.is_finite() || !new_pos.y.is_finite() || !new_pos.z.is_finite() ||
           !new_vel.x.is_finite() || !new_vel.y.is_finite() || !new_vel.z.is_finite() {
            return Err(PhysicsError::NumericalInstability);
        }
        
        // Update entity state
        entity.set_position(new_pos)?;
        entity.set_velocity(new_vel)?;
        
        Ok(())
    }
    
    fn suggested_timestep<E: PhysicsEntity<T>>(&self, entity: &E) -> T {
        // RK4 can handle larger timesteps than Euler
        let vel_mag = entity.velocity().magnitude();
        if vel_mag <= T::epsilon() {
            return T::default_timestep();
        }
        
        let two = T::one() + T::one();
        // More generous timestep for RK4
        let suggested = T::one() / (vel_mag * two); // /2.0 instead of /10.0
        
        let min_dt = T::default_timestep() / (T::one() + T::one() + T::one()); // /3.0
        let max_dt = T::default_timestep() * two; // *2.0
        
        if suggested < min_dt { min_dt }
        else if suggested > max_dt { max_dt }
        else { suggested }
    }
}

/// Velocity Verlet integration (symplectic)
/// Excellent for molecular dynamics and conservative systems
/// Reference: "Computer Simulation of Liquids" Allen & Tildesley
#[derive(Debug)]
pub struct VerletSolver;

impl<T: PhysicsFloat + PhysicsConstants<T>> Solver<T> for VerletSolver {
    fn order(&self) -> u8 {
        2 // Second order accuracy
    }
    
    fn stability_region(&self) -> T {
        // Verlet is symplectic, excellent long-term stability
        T::one() + T::one() + T::one() + T::one() // 4.0
    }
    
    fn step<E: PhysicsEntity<T>>(
        &self,
        entity: &mut E,
        forces: &[Box<dyn ForceField<T>>],
        dt: T,
    ) -> PhysicsResult<()> {
        // Validate inputs
        if dt <= T::zero() || !dt.is_finite() {
            return Err(PhysicsError::InvalidParameters("Timestep must be positive and finite".into()));
        }
        
        entity.validate_invariants()?;
        
        let pos0 = entity.position();
        let vel0 = entity.velocity();
        let mass = entity.mass();
        let inv_mass = T::one() / mass;
        
        // Compute acceleration at current position
        let mut total_force = Vector3D::zero();
        for force_field in forces {
            total_force = total_force + force_field.compute_force(pos0, T::zero());
        }
        let acc0 = total_force * inv_mass;
        
        // Velocity Verlet algorithm:
        // x(t+dt) = x(t) + v(t)*dt + 0.5*a(t)*dt²
        // v(t+dt) = v(t) + 0.5*[a(t) + a(t+dt)]*dt
        
        let dt_sq = dt * dt;
        let half_dt = dt * (T::one() / (T::one() + T::one()));
        let half = T::one() / (T::one() + T::one());
        
        // Update position
        let new_pos = pos0 + vel0 * dt + acc0 * (half * dt_sq);
        
        // Compute acceleration at new position
        let mut total_force_new = Vector3D::zero();
        for force_field in forces {
            total_force_new = total_force_new + force_field.compute_force(new_pos, T::zero());
        }
        let acc1 = total_force_new * inv_mass;
        
        // Update velocity with average acceleration
        let new_vel = vel0 + (acc0 + acc1) * half_dt;
        
        // Check for numerical stability
        if !new_pos.x.is_finite() || !new_pos.y.is_finite() || !new_pos.z.is_finite() ||
           !new_vel.x.is_finite() || !new_vel.y.is_finite() || !new_vel.z.is_finite() {
            return Err(PhysicsError::NumericalInstability);
        }
        
        // Update entity state
        entity.set_position(new_pos)?;
        entity.set_velocity(new_vel)?;
        
        Ok(())
    }
    
    fn suggested_timestep<E: PhysicsEntity<T>>(&self, entity: &E) -> T {
        // Verlet can handle moderate timesteps
        let vel_mag = entity.velocity().magnitude();
        if vel_mag <= T::epsilon() {
            return T::default_timestep();
        }
        
        let two = T::one() + T::one();
        let three = two + T::one();
        let suggested = T::one() / (vel_mag * three); // /3.0
        
        let min_dt = T::default_timestep() / two; // /2.0
        let max_dt = T::default_timestep() * two; // *2.0
        
        if suggested < min_dt { min_dt }
        else if suggested > max_dt { max_dt }
        else { suggested }
    }
}

/// Hybrid adaptive solver
/// Uses enum dispatch instead of trait objects for better performance
#[derive(Debug)]
pub struct HybridSolver {
    solver_type: SolverType,
}

/// Solver type enum for dispatch
#[derive(Debug, Clone, Copy)]
pub enum SolverType {
    Euler,
    RungeKutta4,
    Verlet,
}

impl HybridSolver {
    /// Create new hybrid solver
    pub fn new() -> Self {
        Self {
            solver_type: SolverType::Verlet, // Default to Verlet
        }
    }
    
    /// Choose best solver for given entity characteristics
    fn choose_solver<T: PhysicsFloat + PhysicsConstants<T>, E: PhysicsEntity<T>>(&self, entity: &E, dt: T) -> SolverType {
        let vel_mag = entity.velocity().magnitude();
        
        // Use Verlet for moderate velocities (good for conservative systems)
        if vel_mag <= T::one() {
            SolverType::Verlet
        }
        // Use RK4 for high accuracy requirements
        else if dt <= T::default_timestep() {
            SolverType::RungeKutta4
        }
        // Use Euler for very small timesteps (simple and stable)
        else {
            SolverType::Euler
        }
    }
    
    /// Execute step with chosen solver
    fn execute_step<T: PhysicsFloat + PhysicsConstants<T>, E: PhysicsEntity<T>>(
        solver_type: SolverType,
        entity: &mut E,
        forces: &[Box<dyn ForceField<T>>],
        dt: T,
    ) -> PhysicsResult<()> {
        match solver_type {
            SolverType::Euler => {
                let solver = EulerSolver;
                solver.step(entity, forces, dt)
            }
            SolverType::RungeKutta4 => {
                let solver = RungeKuttaSolver;
                solver.step(entity, forces, dt)
            }
            SolverType::Verlet => {
                let solver = VerletSolver;
                solver.step(entity, forces, dt)
            }
        }
    }
    
    /// Get suggested timestep for chosen solver
    fn get_suggested_timestep<T: PhysicsFloat + PhysicsConstants<T>, E: PhysicsEntity<T>>(
        solver_type: SolverType,
        entity: &E,
    ) -> T {
        match solver_type {
            SolverType::Euler => {
                let solver = EulerSolver;
                solver.suggested_timestep(entity)
            }
            SolverType::RungeKutta4 => {
                let solver = RungeKuttaSolver;
                solver.suggested_timestep(entity)
            }
            SolverType::Verlet => {
                let solver = VerletSolver;
                solver.suggested_timestep(entity)
            }
        }
    }
}

impl<T: PhysicsFloat + PhysicsConstants<T>> Solver<T> for HybridSolver {
    fn order(&self) -> u8 {
        4 // Maximum order of constituent solvers
    }
    
    fn stability_region(&self) -> T {
        // Use most conservative stability region
        let verlet = VerletSolver;
        verlet.stability_region()
    }
    
    fn step<E: PhysicsEntity<T>>(
        &self,
        entity: &mut E,
        forces: &[Box<dyn ForceField<T>>],
        dt: T,
    ) -> PhysicsResult<()> {
        let solver_type = self.choose_solver(entity, dt);
        Self::execute_step(solver_type, entity, forces, dt)
    }
    
    fn suggested_timestep<E: PhysicsEntity<T>>(&self, entity: &E) -> T {
        // Use most conservative suggestion
        let euler_dt = Self::get_suggested_timestep(SolverType::Euler, entity);
        let rk4_dt = Self::get_suggested_timestep(SolverType::RungeKutta4, entity);
        let verlet_dt = Self::get_suggested_timestep(SolverType::Verlet, entity);
        
        // Return minimum (most conservative)
        if euler_dt <= rk4_dt && euler_dt <= verlet_dt {
            euler_dt
        } else if rk4_dt <= verlet_dt {
            rk4_dt
        } else {
            verlet_dt
        }
    }
}

impl Default for HybridSolver {
    fn default() -> Self {
        Self::new()
    }
}
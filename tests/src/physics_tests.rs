//! Unit tests for the physics simulation module.
//!
//! Tests mathematical correctness, numerical stability, and performance targets.

#[cfg(test)]
mod physics_tests {
    use super::*;
    use moirai_core::physics::{
        Vector3D, Particle, RigidBody, PhysicsEntity, ForceField, Solver,
        EulerSolver, RungeKuttaSolver, VerletSolver, HybridSolver,
        PhysicsResult, PhysicsError, PhysicsFloat,
        constants::{GRAVITY_EARTH_F64, DEFAULT_TIMESTEP_F64, PhysicsConstants},
    };
    
    const EPSILON: f64 = 1e-10;
    
    /// Simple constant force field for testing
    struct ConstantForce<T: PhysicsFloat> {
        force: Vector3D<T>,
    }
    
    impl<T: PhysicsFloat> ConstantForce<T> {
        fn new(force: Vector3D<T>) -> Self {
            Self { force }
        }
    }
    
    impl<T: PhysicsFloat> ForceField<T> for ConstantForce<T> {
        fn compute_force(&self, _position: Vector3D<T>, _time: T) -> Vector3D<T> {
            self.force
        }
        
        fn is_conservative(&self) -> bool {
            true
        }
    }
    
    #[test]
    fn test_vector3d_basic_operations() {
        let v1 = Vector3D::new(1.0, 2.0, 3.0);
        let v2 = Vector3D::new(4.0, 5.0, 6.0);
        
        // Test addition
        let sum = v1 + v2;
        assert!((sum.x - 5.0).abs() < EPSILON);
        assert!((sum.y - 7.0).abs() < EPSILON);
        assert!((sum.z - 9.0).abs() < EPSILON);
        
        // Test subtraction
        let diff = v2 - v1;
        assert!((diff.x - 3.0).abs() < EPSILON);
        assert!((diff.y - 3.0).abs() < EPSILON);
        assert!((diff.z - 3.0).abs() < EPSILON);
        
        // Test scalar multiplication
        let scaled = v1 * 2.0;
        assert!((scaled.x - 2.0).abs() < EPSILON);
        assert!((scaled.y - 4.0).abs() < EPSILON);
        assert!((scaled.z - 6.0).abs() < EPSILON);
        
        // Test dot product
        let dot = v1.dot(v2);
        assert!((dot - 32.0).abs() < EPSILON); // 1*4 + 2*5 + 3*6 = 32
        
        // Test magnitude
        let mag = v1.magnitude();
        let expected_mag = (1.0 + 4.0 + 9.0).sqrt(); // sqrt(14)
        assert!((mag - expected_mag).abs() < EPSILON);
    }
    
    #[test]
    fn test_vector3d_normalize() {
        let v = Vector3D::new(3.0, 4.0, 0.0);
        let normalized = v.normalize().unwrap();
        
        // Should have magnitude 1
        assert!((normalized.magnitude() - 1.0).abs() < EPSILON);
        
        // Should be in same direction
        assert!((normalized.x - 0.6).abs() < EPSILON);
        assert!((normalized.y - 0.8).abs() < EPSILON);
        assert!((normalized.z - 0.0).abs() < EPSILON);
    }
    
    #[test]
    fn test_vector3d_normalize_zero_vector() {
        let zero: Vector3D<f64> = Vector3D::zero();
        assert!(zero.normalize().is_err());
    }
    
    #[test]
    fn test_particle_creation() {
        let pos = Vector3D::new(1.0, 2.0, 3.0);
        let vel = Vector3D::new(0.1, 0.2, 0.3);
        let mass = 5.0;
        
        let particle = Particle::new(pos, vel, mass).unwrap();
        
        assert_eq!(particle.position(), pos);
        assert_eq!(particle.velocity(), vel);
        assert_eq!(particle.mass(), mass);
        
        // Test kinetic energy calculation
        let expected_ke = 0.5 * mass * vel.magnitude_squared();
        assert!((particle.kinetic_energy() - expected_ke).abs() < EPSILON);
    }
    
    #[test]
    fn test_particle_invalid_mass() {
        let pos = Vector3D::zero();
        let vel = Vector3D::zero();
        
        // Negative mass should fail
        assert!(Particle::new(pos, vel, -1.0).is_err());
        
        // Zero mass should fail
        assert!(Particle::new(pos, vel, 0.0).is_err());
        
        // Infinite mass should fail
        assert!(Particle::new(pos, vel, f64::INFINITY).is_err());
        
        // NaN mass should fail
        assert!(Particle::new(pos, vel, f64::NAN).is_err());
    }
    
    #[test]
    fn test_euler_solver_free_fall() {
        // Test free fall under constant gravity
        let initial_pos = Vector3D::new(0.0, 100.0, 0.0);
        let initial_vel = Vector3D::zero();
        let mass = 1.0;
        
        let mut particle = Particle::new(initial_pos, initial_vel, mass).unwrap();
        let gravity_force = ConstantForce::new(Vector3D::new(0.0, -mass * GRAVITY_EARTH_F64, 0.0));
        let forces: Vec<Box<dyn ForceField<f64>>> = vec![Box::new(gravity_force)];
        
        let solver = EulerSolver;
        let dt = 0.01; // 10ms timestep
        let time = 1.0; // 1 second
        let steps = (time / dt) as u32;
        
        for _ in 0..steps {
            solver.step(&mut particle, &forces, dt).unwrap();
        }
        
        // Analytical solution: y = y0 + v0*t - 0.5*g*t²
        let expected_y = initial_pos.y - 0.5 * GRAVITY_EARTH_F64 * time * time;
        let actual_y = particle.position().y;
        
        // Euler method has some error, but should be reasonable for small timesteps
        let error_percent = ((actual_y - expected_y) / expected_y).abs() * 100.0;
        assert!(error_percent < 5.0, "Error too large: {:.2}%", error_percent);
    }
    
    #[test]
    fn test_runge_kutta_accuracy() {
        // RK4 should be more accurate than Euler for the same timestep
        let initial_pos = Vector3D::new(0.0, 100.0, 0.0);
        let initial_vel = Vector3D::zero();
        let mass = 1.0;
        
        let gravity_force = ConstantForce::new(Vector3D::new(0.0, -mass * GRAVITY_EARTH_F64, 0.0));
        let forces: Vec<Box<dyn ForceField<f64>>> = vec![Box::new(gravity_force)];
        
        let dt = 0.1; // Larger timestep to see the difference
        let time = 1.0;
        let steps = (time / dt) as u32;
        
        // Test with Euler
        let mut particle_euler = Particle::new(initial_pos, initial_vel, mass).unwrap();
        let euler_solver = EulerSolver;
        for _ in 0..steps {
            euler_solver.step(&mut particle_euler, &forces, dt).unwrap();
        }
        
        // Test with RK4
        let mut particle_rk4 = Particle::new(initial_pos, initial_vel, mass).unwrap();
        let rk4_solver = RungeKuttaSolver;
        for _ in 0..steps {
            rk4_solver.step(&mut particle_rk4, &forces, dt).unwrap();
        }
        
        // Analytical solution
        let expected_y = initial_pos.y - 0.5 * GRAVITY_EARTH_F64 * time * time;
        
        let euler_error = (particle_euler.position().y - expected_y).abs();
        let rk4_error = (particle_rk4.position().y - expected_y).abs();
        
        // RK4 should be more accurate
        assert!(rk4_error < euler_error, 
            "RK4 error ({:.6}) should be less than Euler error ({:.6})", 
            rk4_error, euler_error);
    }
    
    #[test]
    fn test_energy_conservation_verlet() {
        // Test that Verlet solver conserves energy well for oscillatory motion
        let initial_pos = Vector3D::new(1.0, 0.0, 0.0);
        let initial_vel = Vector3D::new(0.0, 1.0, 0.0);
        let mass = 1.0;
        
        let mut particle = Particle::new(initial_pos, initial_vel, mass).unwrap();
        
        // Simple harmonic oscillator force: F = -k*x
        let k = 1.0; // Spring constant
        struct HarmonicOscillator { k: f64 }
        impl ForceField<f64> for HarmonicOscillator {
            fn compute_force(&self, position: Vector3D<f64>, _time: f64) -> Vector3D<f64> {
                Vector3D::new(-self.k * position.x, -self.k * position.y, 0.0)
            }
            fn is_conservative(&self) -> bool { true }
        }
        
        let forces: Vec<Box<dyn ForceField<f64>>> = vec![Box::new(HarmonicOscillator { k })];
        let initial_energy = particle.kinetic_energy() + 0.5 * k * initial_pos.magnitude_squared();
        
        let solver = VerletSolver;
        let dt = 0.01;
        let steps = 1000; // Multiple periods
        
        for _ in 0..steps {
            solver.step(&mut particle, &forces, dt).unwrap();
        }
        
        let final_pos = particle.position();
        let final_energy = particle.kinetic_energy() + 0.5 * k * final_pos.magnitude_squared();
        
        let energy_error = (final_energy - initial_energy).abs() / initial_energy;
        assert!(energy_error < 0.01, "Energy not conserved: {:.2}% error", energy_error * 100.0);
    }
    
    #[test]
    fn test_hybrid_solver_selection() {
        let initial_pos = Vector3D::zero();
        let initial_vel = Vector3D::zero();
        let mass = 1.0;
        
        let particle = Particle::new(initial_pos, initial_vel, mass).unwrap();
        let hybrid_solver: HybridSolver = HybridSolver::new();
        
        // Test that hybrid solver doesn't crash
        assert!((<HybridSolver as Solver<f64>>::order(&hybrid_solver)) > 0);
        assert!((<HybridSolver as Solver<f64>>::stability_region(&hybrid_solver)) > 0.0);
        assert!((<HybridSolver as Solver<f64>>::suggested_timestep(&hybrid_solver, &particle)) > 0.0);
    }
    
    #[test]
    fn test_performance_target() {
        // Test that we can meet the <1ms per step requirement
        use std::time::Instant;
        
        let initial_pos = Vector3D::new(0.0, 100.0, 0.0);
        let initial_vel = Vector3D::zero();
        let mass = 1.0;
        
        let mut particle = Particle::new(initial_pos, initial_vel, mass).unwrap();
        let gravity_force = ConstantForce::new(Vector3D::new(0.0, -mass * GRAVITY_EARTH_F64, 0.0));
        let forces: Vec<Box<dyn ForceField<f64>>> = vec![Box::new(gravity_force)];
        
        let solver = HybridSolver::new();
        let dt = DEFAULT_TIMESTEP_F64;
        let test_steps = 1000;
        
        let start = Instant::now();
        for _ in 0..test_steps {
            solver.step(&mut particle, &forces, dt).unwrap();
        }
        let elapsed = start.elapsed();
        
        let ns_per_step = elapsed.as_nanos() as f64 / test_steps as f64;
        let target_ns = 1_000_000.0; // 1ms in nanoseconds
        
        println!("Performance: {:.0} ns/step (target: {:.0} ns/step)", ns_per_step, target_ns);
        assert!(ns_per_step < target_ns, 
            "Performance target not met: {:.0} ns > 1ms per step", ns_per_step);
    }
    
    #[test]
    fn test_numerical_stability() {
        // Test that solvers remain stable with challenging conditions
        let initial_pos = Vector3D::new(1e6, 1e6, 1e6); // Large positions
        let initial_vel = Vector3D::new(1e3, 1e3, 1e3); // Large velocities
        let mass = 1e-6; // Small mass
        
        let mut particle = Particle::new(initial_pos, initial_vel, mass).unwrap();
        let small_force = ConstantForce::new(Vector3D::new(1e-12, 1e-12, 1e-12));
        let forces: Vec<Box<dyn ForceField<f64>>> = vec![Box::new(small_force)];
        
        let solver = VerletSolver; // Most stable solver
        let dt = 1e-6; // Very small timestep
        
        // Run for multiple steps without crashing
        for _ in 0..100 {
            let result = solver.step(&mut particle, &forces, dt);
            assert!(result.is_ok(), "Solver became unstable");
            
            let pos = particle.position();
            assert!(pos.x.is_finite() && pos.y.is_finite() && pos.z.is_finite(),
                "Position became non-finite");
                
            let vel = particle.velocity();
            assert!(vel.x.is_finite() && vel.y.is_finite() && vel.z.is_finite(),
                "Velocity became non-finite");
        }
    }
    
    #[test]
    fn test_rigid_body_physics() {
        let pos = Vector3D::new(0.0, 0.0, 0.0);
        let vel = Vector3D::new(1.0, 0.0, 0.0);
        let orientation = Vector3D::zero();
        let angular_vel = Vector3D::new(0.0, 0.0, 1.0); // Spinning around z-axis
        let mass = 2.0;
        let moment_of_inertia = 0.5; // Sphere-like
        
        let rigid_body = RigidBody::new(pos, vel, orientation, angular_vel, mass, moment_of_inertia).unwrap();
        
        assert_eq!(rigid_body.position(), pos);
        assert_eq!(rigid_body.velocity(), vel);
        assert_eq!(rigid_body.mass(), mass);
        assert_eq!(rigid_body.orientation(), orientation);
        assert_eq!(rigid_body.angular_velocity(), angular_vel);
        assert_eq!(rigid_body.moment_of_inertia(), moment_of_inertia);
        
        // Test energy calculations
        let linear_ke = 0.5 * mass * vel.magnitude_squared();
        let rotational_ke = 0.5 * moment_of_inertia * angular_vel.magnitude_squared();
        let total_ke = linear_ke + rotational_ke;
        
        assert!((rigid_body.total_kinetic_energy() - total_ke).abs() < EPSILON);
    }
}
//! Simple physics simulation example using Moirai's new physics module.
//!
//! Demonstrates particle motion under gravitational force with different solvers.

use moirai_core::physics::{
    Particle, Vector3D, PhysicsEntity, ForceField, Solver,
    EulerSolver, RungeKuttaSolver, VerletSolver, HybridSolver,
    PhysicsResult, PhysicsFloat,
    constants::{GRAVITY_EARTH_F64, DEFAULT_TIMESTEP_F64},
};
use std::time::Instant;

/// Simple gravitational force field pointing downward
struct GravitationalField<T: PhysicsFloat> {
    gravity: T,
}

impl<T: PhysicsFloat> GravitationalField<T> {
    fn new(gravity: T) -> Self {
        Self { gravity }
    }
}

impl<T: PhysicsFloat> ForceField<T> for GravitationalField<T> {
    fn compute_force(&self, _position: Vector3D<T>, _time: T) -> Vector3D<T> {
        Vector3D::new(T::zero(), -self.gravity, T::zero())
    }
    
    fn is_conservative(&self) -> bool {
        true
    }
}



fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Moirai Physics Simulation Demo");
    println!("================================\n");
    
    // Create a particle at height 100m with zero initial velocity
    let initial_pos = Vector3D::new(0.0, 100.0, 0.0);
    let initial_vel = Vector3D::zero();
    let mass = 1.0; // 1 kg
    
    let particle = Particle::new(initial_pos, initial_vel, mass)?;
    
    println!("Initial conditions:");
    println!("  Position: ({:.2}, {:.2}, {:.2}) m", initial_pos.x, initial_pos.y, initial_pos.z);
    println!("  Velocity: ({:.2}, {:.2}, {:.2}) m/s", initial_vel.x, initial_vel.y, initial_vel.z);
    println!("  Mass: {:.2} kg", mass);
    println!("  Gravity: {:.2} m/s²\n", GRAVITY_EARTH_F64);
    
    let timestep = DEFAULT_TIMESTEP_F64;
    let duration = 1.0; // 1 second simulation
    
    println!("Simulation parameters:");
    println!("  Timestep: {:.6} s", timestep);
    println!("  Duration: {:.2} s", duration);
    println!("  Steps: {}\n", (duration / timestep) as u32);
    
    // Test different solvers using an enum approach instead of trait objects
    #[derive(Debug)]
    enum SolverType {
        Euler(EulerSolver),
        RungeKutta4(RungeKuttaSolver),
        Verlet(VerletSolver),
        Hybrid(HybridSolver),
    }
    
    impl SolverType {
        fn name(&self) -> &str {
            match self {
                SolverType::Euler(_) => "Euler",
                SolverType::RungeKutta4(_) => "Runge-Kutta 4",
                SolverType::Verlet(_) => "Verlet",
                SolverType::Hybrid(_) => "Hybrid",
            }
        }
        
        fn solve(&self, mut particle: Particle<f64>, forces: &[Box<dyn ForceField<f64>>], timestep: f64, duration: f64) -> PhysicsResult<(Vector3D<f64>, f64)> {
            let steps = (duration / timestep) as u32;
            let start_time = Instant::now();
            
            for _ in 0..steps {
                match self {
                    SolverType::Euler(solver) => solver.step(&mut particle, forces, timestep)?,
                    SolverType::RungeKutta4(solver) => solver.step(&mut particle, forces, timestep)?,
                    SolverType::Verlet(solver) => solver.step(&mut particle, forces, timestep)?,
                    SolverType::Hybrid(solver) => solver.step(&mut particle, forces, timestep)?,
                }
            }
            
            let elapsed = start_time.elapsed();
            Ok((particle.position(), elapsed.as_secs_f64()))
        }
    }
    
    let solvers = vec![
        SolverType::Euler(EulerSolver),
        SolverType::RungeKutta4(RungeKuttaSolver),
        SolverType::Verlet(VerletSolver),
        SolverType::Hybrid(HybridSolver::new()),
    ];
    
    println!("Solver comparison:");
    println!("{:<15} {:<20} {:<15} {:<15}", "Solver", "Final Position (m)", "Runtime (ms)", "Error (%)");
    println!("{:-<65}", "");
    
    // Analytical solution for free fall: y = y0 - 0.5*g*t²
    let analytical_y = initial_pos.y - 0.5 * GRAVITY_EARTH_F64 * duration * duration;
    
    // Create forces for the simulation
    let force_field = GravitationalField::new(GRAVITY_EARTH_F64);
    let forces: Vec<Box<dyn ForceField<f64>>> = vec![Box::new(force_field)];
    
    for solver in solvers {
        let particle_copy = Particle::new(initial_pos, initial_vel, mass)?;
        
        match solver.solve(particle_copy, &forces, timestep, duration) {
            Ok((final_pos, runtime)) => {
                let error_percent = ((final_pos.y - analytical_y) / analytical_y).abs() * 100.0;
                println!("{:<15} ({:.2}, {:.2}, {:.2}) {:<15.3} {:<15.3}", 
                    solver.name(), final_pos.x, final_pos.y, final_pos.z, 
                    runtime * 1000.0, error_percent);
            }
            Err(e) => {
                println!("{:<15} ERROR: {}", solver.name(), e);
            }
        }
    }
    
    println!("\nAnalytical solution: y = {:.2} m", analytical_y);
    
    // Performance benchmark
    println!("\n🏁 Performance Benchmark:");
    println!("========================");
    
    let benchmark_steps = 10000;
    let benchmark_timestep = 1e-6;
    
    let particle_bench = Particle::new(initial_pos, initial_vel, mass)?;
    let benchmark_solver = SolverType::Hybrid(HybridSolver::new());
    
    let start = Instant::now();
    match benchmark_solver.solve(particle_bench, &forces, benchmark_timestep, benchmark_steps as f64 * benchmark_timestep) {
        Ok((_, _)) => {
            let elapsed = start.elapsed();
            let steps_per_second = benchmark_steps as f64 / elapsed.as_secs_f64();
            let runtime_per_step = elapsed.as_nanos() as f64 / benchmark_steps as f64;
            
            println!("Steps: {}", benchmark_steps);
            println!("Total time: {:.3} ms", elapsed.as_millis());
            println!("Steps/second: {:.2e}", steps_per_second);
            println!("Time/step: {:.0} ns", runtime_per_step);
            
            // Check if we meet the <1ms requirement
            let target_ns = 1_000_000.0; // 1ms in nanoseconds
            if runtime_per_step < target_ns {
                println!("✅ TARGET MET: {:.0} ns < 1ms per step", runtime_per_step);
            } else {
                println!("❌ TARGET MISSED: {:.0} ns > 1ms per step", runtime_per_step);
            }
        }
        Err(e) => {
            println!("Benchmark failed: {}", e);
        }
    }
    
    Ok(())
}
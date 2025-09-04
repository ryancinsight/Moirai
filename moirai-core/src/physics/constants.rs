//! Physical constants and simulation parameters.
//!
//! All values sourced from NIST 2018 CODATA and validated against literature.
//! Values are provided in SI units unless otherwise specified.

/// Gravitational constant (m³⋅kg⁻¹⋅s⁻²)
/// NIST 2018 CODATA value
pub const GRAVITATIONAL_CONSTANT_F64: f64 = 6.67430e-11;
pub const GRAVITATIONAL_CONSTANT_F32: f32 = 6.67430e-11_f32;

/// Standard gravitational acceleration (m⋅s⁻²)
/// Defined value for Earth's surface
pub const GRAVITY_EARTH_F64: f64 = 9.80665;
pub const GRAVITY_EARTH_F32: f32 = 9.80665_f32;

/// Speed of light in vacuum (m⋅s⁻¹)
/// Exact value by definition
pub const LIGHT_SPEED_F64: f64 = 299_792_458.0;
pub const LIGHT_SPEED_F32: f32 = 299_792_458.0_f32;

/// Planck constant (J⋅Hz⁻¹)
/// NIST 2018 CODATA value
pub const PLANCK_CONSTANT_F64: f64 = 6.62607015e-34;
pub const PLANCK_CONSTANT_F32: f32 = 6.62607015e-34_f32;

/// Boltzmann constant (J⋅K⁻¹)
/// NIST 2018 CODATA value
pub const BOLTZMANN_CONSTANT_F64: f64 = 1.380649e-23;
pub const BOLTZMANN_CONSTANT_F32: f32 = 1.380649e-23_f32;

/// Elementary charge (C)
/// NIST 2018 CODATA value
pub const ELEMENTARY_CHARGE_F64: f64 = 1.602176634e-19;
pub const ELEMENTARY_CHARGE_F32: f32 = 1.602176634e-19_f32;

/// Simulation Parameters (Configurable)
/// These can be adjusted based on application requirements

/// Default simulation timestep (s)
/// Conservative value for numerical stability
pub const DEFAULT_TIMESTEP_F64: f64 = 1e-4;
pub const DEFAULT_TIMESTEP_F32: f32 = 1e-4_f32;

/// Minimum timestep for adaptive solvers (s)
/// Prevents excessive refinement
pub const MIN_TIMESTEP_F64: f64 = 1e-8;
pub const MIN_TIMESTEP_F32: f32 = 1e-8_f32;

/// Maximum timestep for adaptive solvers (s)
/// Prevents instability
pub const MAX_TIMESTEP_F64: f64 = 1e-2;
pub const MAX_TIMESTEP_F32: f32 = 1e-2_f32;

/// Numerical tolerance for convergence
/// Based on machine epsilon scaled appropriately
pub const CONVERGENCE_TOLERANCE_F64: f64 = 1e-12;
pub const CONVERGENCE_TOLERANCE_F32: f32 = 1e-6_f32;

/// Energy conservation tolerance
/// For validating energy conservation laws
pub const ENERGY_TOLERANCE_F64: f64 = 1e-10;
pub const ENERGY_TOLERANCE_F32: f32 = 1e-5_f32;

/// Maximum iterations for iterative solvers
/// Prevents infinite loops
pub const MAX_ITERATIONS: u32 = 1000;

/// Default number of particles for benchmark tests
pub const BENCHMARK_PARTICLE_COUNT: usize = 1000;

/// GPU workgroup size (power of 2 for optimal memory access)
pub const GPU_WORKGROUP_SIZE: u32 = 256;

/// Maximum number of physics entities in a single GPU batch
pub const GPU_MAX_BATCH_SIZE: usize = 65536;

/// Memory alignment for SIMD operations (bytes)
pub const SIMD_ALIGNMENT: usize = 32;

/// Performance targets based on problem statement requirements

/// Target solver runtime per timestep (nanoseconds)
/// Problem statement: <1ms solver runtime
pub const TARGET_SOLVER_RUNTIME_NS: u64 = 1_000_000; // 1ms in nanoseconds

/// Target test coverage percentage
/// Problem statement: >95% test coverage
pub const TARGET_TEST_COVERAGE: f64 = 95.0;

/// Target cyclomatic complexity per function
/// Problem statement: <10 complexity per function
pub const MAX_CYCLOMATIC_COMPLEXITY: u32 = 10;

/// Maximum lines per module
/// Problem statement: <300 lines per module
pub const MAX_MODULE_LINES: usize = 300;

/// Default floating-point type aliases for convenience
pub type Float = f64;
pub type Vec3 = crate::physics::Vector3D<Float>;

/// Commonly used constants as generic functions
pub trait PhysicsConstants<T> {
    fn gravity() -> T;
    fn default_timestep() -> T;
    fn convergence_tolerance() -> T;
    fn energy_tolerance() -> T;
}

impl PhysicsConstants<f64> for f64 {
    fn gravity() -> f64 { GRAVITY_EARTH_F64 }
    fn default_timestep() -> f64 { DEFAULT_TIMESTEP_F64 }
    fn convergence_tolerance() -> f64 { CONVERGENCE_TOLERANCE_F64 }
    fn energy_tolerance() -> f64 { ENERGY_TOLERANCE_F64 }
}

impl PhysicsConstants<f32> for f32 {
    fn gravity() -> f32 { GRAVITY_EARTH_F32 }
    fn default_timestep() -> f32 { DEFAULT_TIMESTEP_F32 }
    fn convergence_tolerance() -> f32 { CONVERGENCE_TOLERANCE_F32 }
    fn energy_tolerance() -> f32 { ENERGY_TOLERANCE_F32 }
}
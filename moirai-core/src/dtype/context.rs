use super::{Dtype, FloatDtype};

/// Common type aliases for convenience
///
/// Default integer type for algorithms
pub type DefaultInt = i64;

/// Default unsigned integer type for algorithms  
pub type DefaultUint = u64;

/// Default floating-point type for algorithms
pub type DefaultFloat = f64;

/// Computational context for generic algorithms
pub struct ComputeContext<T: Dtype> {
    /// Tolerance for floating-point comparisons
    pub tolerance: Option<T>,
    /// Maximum number of iterations for iterative algorithms
    pub max_iterations: usize,
    /// Whether to check for arithmetic overflow
    pub check_overflow: bool,
}

impl<T: Dtype> Default for ComputeContext<T> {
    fn default() -> Self {
        Self {
            tolerance: None,
            max_iterations: 1000,
            check_overflow: true,
        }
    }
}

impl<T: FloatDtype> ComputeContext<T> {
    /// Create context with specified tolerance
    pub fn with_tolerance(tolerance: T) -> Self {
        Self {
            tolerance: Some(tolerance),
            max_iterations: 1000,
            check_overflow: true,
        }
    }

    /// Create context with machine epsilon tolerance
    pub fn with_epsilon() -> Self {
        Self {
            tolerance: Some(T::epsilon()),
            max_iterations: 1000,
            check_overflow: true,
        }
    }
}

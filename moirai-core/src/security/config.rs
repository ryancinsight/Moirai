use crate::{Priority, TaskId};
use std::time::{Duration, SystemTime};

// Memory size constants
pub(crate) const GIGABYTE: usize = 1024 * 1024 * 1024;
pub(crate) const MEGABYTE: usize = 1024 * 1024;

// Time constants
pub(crate) const SECONDS_PER_DAY: u64 = 24 * 3600;
pub(crate) const DAYS_IN_WEEK: u64 = 7;
pub(crate) const DAYS_IN_MONTH: u64 = 30;

// Security thresholds
pub(crate) const DEFAULT_MAX_ALLOCATION_SIZE: usize = GIGABYTE;
pub(crate) const PRODUCTION_MAX_ALLOCATION_SIZE: usize = 512 * MEGABYTE;
pub(crate) const DEFAULT_TASK_SPAWN_RATE: u64 = 10_000;
pub(crate) const PRODUCTION_TASK_SPAWN_RATE: u64 = 5_000;
pub(crate) const DEFAULT_RATE_LIMITER_WINDOWS: usize = 10;
pub(crate) const MAX_REPRESENTABLE_UNIX_NANOS: u64 = u64::MAX - 1;

/// Security levels that can be applied to task execution environments.
#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum SecurityLevel {
    /// Development environment - minimal security checks
    Development,
    /// Testing environment - moderate security validation
    Testing,
    /// Staging environment - comprehensive security checks
    Staging,
    /// Production environment - maximum security validation
    Production,
}

/// Security-related events that can occur during task execution.
#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone)]
pub enum SecurityEvent {
    /// Task spawning with security context
    TaskSpawn {
        /// The ID of the spawned task
        task_id: TaskId,
        /// The priority of the spawned task
        priority: Priority,
        /// When the event occurred
        timestamp: SystemTime,
    },
    /// Memory allocation beyond normal bounds
    MemoryAnomalous {
        /// Size of the allocation in bytes
        size: usize,
        /// Location where the allocation occurred
        location: String,
        /// When the event occurred
        timestamp: SystemTime,
    },
    /// Potential race condition detected
    RaceCondition {
        /// Description of the potential race condition
        description: String,
        /// When the event occurred
        timestamp: SystemTime,
    },
    /// Resource exhaustion detected
    ResourceExhaustion {
        /// Name of the exhausted resource
        resource: String,
        /// Current usage level
        current: u64,
        /// Maximum allowed limit
        limit: u64,
        /// When the event occurred
        timestamp: SystemTime,
    },
}

/// Configuration settings for security policies and enforcement.
#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone)]
pub struct SecurityConfig {
    /// Security level for this deployment
    pub level: SecurityLevel,
    /// Maximum memory allocation size before triggering audit
    pub max_allocation_size: usize,
    /// Maximum number of tasks per second before triggering audit
    pub max_task_spawn_rate: u64,
    /// Enable memory safety validation
    pub enable_memory_validation: bool,
    /// Enable race condition detection
    pub enable_race_detection: bool,
    /// Audit log retention period
    pub audit_retention: Duration,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            level: SecurityLevel::Development,
            max_allocation_size: DEFAULT_MAX_ALLOCATION_SIZE,
            max_task_spawn_rate: DEFAULT_TASK_SPAWN_RATE,
            enable_memory_validation: true,
            enable_race_detection: true,
            audit_retention: Duration::from_secs(DAYS_IN_WEEK * SECONDS_PER_DAY),
        }
    }
}

impl SecurityConfig {
    /// Creates a production-ready security configuration.
    ///
    /// # Returns
    /// A security configuration suitable for production environments
    #[must_use]
    pub fn production() -> Self {
        Self {
            level: SecurityLevel::Production,
            max_allocation_size: PRODUCTION_MAX_ALLOCATION_SIZE,
            max_task_spawn_rate: PRODUCTION_TASK_SPAWN_RATE,
            enable_memory_validation: true,
            enable_race_detection: true,
            audit_retention: Duration::from_secs(DAYS_IN_MONTH * SECONDS_PER_DAY),
        }
    }
}

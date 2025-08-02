//! Error types and handling for the Moirai runtime.

use core::fmt;

/// Errors that can occur during task operations.
#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskError {
    /// Task was cancelled before completion
    Cancelled,
    /// Task panicked during execution
    Panicked,
    /// Task exceeded its execution time limit
    Timeout,
    /// Task failed due to resource exhaustion
    ResourceExhausted,
    /// Task failed due to an invalid operation
    InvalidOperation,
    /// Generic task execution error
    ExecutionFailed(TaskErrorKind),
    /// Task execution timed out waiting for completion
    ExecutionTimeout,
    /// Task result was not found in storage
    ResultNotFound,
    /// Task failed to spawn
    SpawnFailed,
    /// Task is not in a valid state for the operation
    InvalidState,
    /// Task has already completed
    AlreadyCompleted,
}

/// Specific kinds of task execution errors.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TaskErrorKind {
    /// I/O operation failed
    Io,
    /// Network operation failed
    Network,
    /// File system operation failed
    FileSystem,
    /// Permission denied
    PermissionDenied,
    /// Resource not found
    NotFound,
    /// Operation would block
    WouldBlock,
    /// Operation interrupted
    Interrupted,
    /// Invalid input provided
    InvalidInput,
    /// Other error
    Other,
}

impl fmt::Display for TaskError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Cancelled => write!(f, "Task was cancelled"),
            Self::Panicked => write!(f, "Task panicked during execution"),
            Self::ExecutionTimeout => write!(f, "Task execution timed out"),
            Self::ResultNotFound => write!(f, "Task result not found"),
            Self::SpawnFailed => write!(f, "Task failed to spawn"),
            Self::Timeout => write!(f, "Task exceeded execution time limit"),
            Self::ResourceExhausted => write!(f, "Task failed due to resource exhaustion"),
            Self::InvalidOperation => write!(f, "Invalid operation"),
            Self::ExecutionFailed(kind) => write!(f, "Task execution failed: {kind}"),
            Self::InvalidState => write!(f, "Task is not in a valid state for the operation"),
            Self::AlreadyCompleted => write!(f, "Task has already completed"),
        }
    }
}

impl fmt::Display for TaskErrorKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io => write!(f, "I/O error"),
            Self::Network => write!(f, "Network error"),
            Self::FileSystem => write!(f, "File system error"),
            Self::PermissionDenied => write!(f, "Permission denied"),
            Self::NotFound => write!(f, "Resource not found"),
            Self::WouldBlock => write!(f, "Operation would block"),
            Self::Interrupted => write!(f, "Operation interrupted"),
            Self::InvalidInput => write!(f, "Invalid input"),
            Self::Other => write!(f, "Other error"),
        }
    }
}

/// Errors that can occur during executor operations.
#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutorError {
    /// Executor is shutting down
    ShuttingDown,
    /// Executor is already running
    AlreadyRunning,
    /// Executor configuration is invalid
    InvalidConfiguration,
    /// Thread pool creation failed
    ThreadPoolCreationFailed,
    /// Task spawn failed
    SpawnFailed(TaskError),
    /// Resource exhaustion detected
    ResourceExhausted(String),
    /// Performance anomaly detected
    PerformanceAnomaly(String),
    /// No scheduler available
    NoSchedulerAvailable,
    /// Scheduler error
    SchedulerError(SchedulerError),
}

impl fmt::Display for ExecutorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShuttingDown => write!(f, "Executor is shutting down"),
            Self::AlreadyRunning => write!(f, "Executor is already running"),
            Self::InvalidConfiguration => write!(f, "Invalid executor configuration"),
            Self::ThreadPoolCreationFailed => write!(f, "Failed to create thread pool"),
            Self::SpawnFailed(err) => write!(f, "Failed to spawn task: {err}"),
            Self::ResourceExhausted(msg) => write!(f, "Resource exhausted: {msg}"),
            Self::PerformanceAnomaly(msg) => write!(f, "Performance anomaly: {msg}"),
            Self::NoSchedulerAvailable => write!(f, "No scheduler available"),
            Self::SchedulerError(err) => write!(f, "Scheduler error: {}", err),
        }
    }
}

/// Errors that can occur during scheduler operations.
#[allow(clippy::module_name_repetitions)]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchedulerError {
    /// Queue is full and cannot accept more tasks
    QueueFull,
    /// Queue is empty
    QueueEmpty,
    /// Work stealing failed
    StealFailed,
    /// Invalid scheduler state
    InvalidState,
    /// System failure occurred
    SystemFailure(String),
    /// Invalid scheduler reference
    InvalidScheduler,
}

impl fmt::Display for SchedulerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::QueueFull => write!(f, "Task queue is full"),
            Self::QueueEmpty => write!(f, "Task queue is empty"),
            Self::StealFailed => write!(f, "Work stealing failed"),
            Self::InvalidState => write!(f, "Invalid scheduler state"),
            Self::SystemFailure(msg) => write!(f, "System failure: {}", msg),
            Self::InvalidScheduler => write!(f, "Invalid scheduler reference"),
        }
    }
}

/// A result type for task operations.
pub type TaskResult<T> = Result<T, TaskError>;

/// A result type for executor operations.
pub type ExecutorResult<T> = Result<T, ExecutorError>;

/// A result type for scheduler operations.
pub type SchedulerResult<T> = Result<T, SchedulerError>;

#[cfg(feature = "std")]
impl std::error::Error for TaskError {}

#[cfg(feature = "std")]
impl std::error::Error for ExecutorError {}

#[cfg(feature = "std")]
impl std::error::Error for SchedulerError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        assert_eq!(
            format!("{}", TaskError::Cancelled),
            "Task was cancelled"
        );
        assert_eq!(
            format!("{}", TaskError::ExecutionFailed(TaskErrorKind::Io)),
            "Task execution failed: I/O error"
        );
        assert_eq!(
            format!("{}", ExecutorError::ShuttingDown),
            "Executor is shutting down"
        );
        assert_eq!(
            format!("{}", SchedulerError::QueueFull),
            "Task queue is full"
        );
    }
}
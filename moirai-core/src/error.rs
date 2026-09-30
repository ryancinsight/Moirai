//! Error types and handling for the Moirai runtime.

use core::fmt;

/// Errors that can occur during task operations.
#[allow(clippy::module_name_repetitions)]
#[must_use]
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
#[must_use]
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
#[must_use]
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExecutorError {
    /// Executor is shutting down
    ShuttingDown,
    /// Executor is already running
    AlreadyRunning,
    /// Executor configuration is invalid
    InvalidConfiguration,
    /// The requested local queue initial capacity cannot form a supported allocation.
    InvalidLocalQueueInitialCapacity {
        /// Requested slot count before power-of-two normalization.
        requested: usize,
    },
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
    /// A worker could not be confined to its planned logical processor.
    ///
    /// Reported by construction under
    /// [`WorkerPlacement::Pinned`](crate::executor::WorkerPlacement::Pinned).
    /// When several workers fail, the lowest-numbered one is named. The
    /// scheduler was not started: no worker outlives the failed construction.
    WorkerPlacementFailed {
        /// Index of the worker whose binding failed.
        worker: usize,
        /// Flattened logical processor id the worker was planned onto.
        processor: u32,
        /// Why the binding did not take effect.
        cause: PlacementFailure,
    },
}

/// Why a worker could not be confined to a logical processor.
///
/// Every variant leaves the thread with the affinity it had before the attempt.
#[must_use]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum PlacementFailure {
    /// This target has no thread-binding backend.
    Unsupported,
    /// The processor id is beyond what the target affinity interface can name.
    OutOfRange,
    /// The operating system refused the request, for example because the
    /// processor is offline or outside the allowed set of the process.
    Os {
        /// Raw operating-system error code: `GetLastError` on Windows,
        /// `errno` on Linux.
        code: i32,
    },
}

impl fmt::Display for ExecutorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ShuttingDown => write!(f, "Executor is shutting down"),
            Self::AlreadyRunning => write!(f, "Executor is already running"),
            Self::InvalidConfiguration => write!(f, "Invalid executor configuration"),
            Self::InvalidLocalQueueInitialCapacity { requested } => write!(
                f,
                "local queue initial capacity {requested} cannot form a supported allocation"
            ),
            Self::ThreadPoolCreationFailed => write!(f, "Failed to create thread pool"),
            Self::SpawnFailed(err) => write!(f, "Failed to spawn task: {err}"),
            Self::ResourceExhausted(msg) => write!(f, "Resource exhausted: {msg}"),
            Self::PerformanceAnomaly(msg) => write!(f, "Performance anomaly: {msg}"),
            Self::NoSchedulerAvailable => write!(f, "No scheduler available"),
            Self::SchedulerError(err) => write!(f, "Scheduler error: {err}"),
            Self::WorkerPlacementFailed {
                worker, processor, ..
            } => write!(
                f,
                "worker {worker} could not be pinned to logical processor {processor}"
            ),
        }
    }
}

impl fmt::Display for PlacementFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unsupported => write!(f, "thread binding is unsupported on this target"),
            Self::OutOfRange => {
                write!(f, "the processor is outside the range this target can bind")
            }
            Self::Os { code } => write!(
                f,
                "the operating system refused the binding (error code {code})"
            ),
        }
    }
}

/// Errors that can occur during scheduler operations.
#[allow(clippy::module_name_repetitions)]
#[must_use]
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
            Self::SystemFailure(msg) => write!(f, "System failure: {msg}"),
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
impl std::error::Error for ExecutorError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::WorkerPlacementFailed { cause, .. } => Some(cause),
            _ => None,
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for PlacementFailure {}

#[cfg(feature = "std")]
impl std::error::Error for SchedulerError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display() {
        assert_eq!(format!("{}", TaskError::Cancelled), "Task was cancelled");
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

    #[cfg(feature = "std")]
    #[test]
    fn placement_failure_names_worker_processor_and_keeps_cause_as_source() {
        let error = ExecutorError::WorkerPlacementFailed {
            worker: 3,
            processor: 17,
            cause: PlacementFailure::Os { code: 87 },
        };
        assert_eq!(
            format!("{error}"),
            "worker 3 could not be pinned to logical processor 17"
        );
        let source = std::error::Error::source(&error).expect("invariant: cause is the source");
        assert_eq!(
            format!("{source}"),
            "the operating system refused the binding (error code 87)"
        );
        assert!(std::error::Error::source(&ExecutorError::ShuttingDown).is_none());
        assert_eq!(
            format!("{}", PlacementFailure::Unsupported),
            "thread binding is unsupported on this target"
        );
        assert_eq!(
            format!("{}", PlacementFailure::OutOfRange),
            "the processor is outside the range this target can bind"
        );
    }
}

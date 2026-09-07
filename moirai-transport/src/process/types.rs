//! Process specification and lifecycle result contracts.
use std::{ffi::OsString, path::PathBuf, process::ExitStatus, time::Duration};
/// Result of a process lifecycle operation.
pub type ProcessResult<T> = Result<T, ProcessError>;
/// Lifecycle operation whose operating-system call failed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessOperation {
    /// Process or resource creation.
    Spawn,
    /// Completion observation.
    Wait,
    /// Process termination.
    Terminate,
}
/// Process lifecycle failure with preserved operating-system classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessError {
    /// Process spawn failed.
    SpawnFailed,
    /// Process wait failed.
    WaitFailed,
    /// Process termination failed.
    TerminateFailed,
    /// A finite lifecycle deadline elapsed without confirmed completion.
    DeadlineExceeded,
    /// A command, environment, or deadline cannot be represented by the OS API.
    InvalidSpecification,
    /// The target does not implement requested tree containment.
    UnsupportedContainment,
    /// Native operation failed, with its platform error code if available.
    OperatingSystem {
        /// Failed lifecycle operation.
        operation: ProcessOperation,
        /// Native OS error classification.
        code: Option<i32>,
    },
}
impl ProcessError {
    pub(super) const fn operation(self) -> ProcessOperation {
        match self {
            Self::WaitFailed => ProcessOperation::Wait,
            Self::TerminateFailed => ProcessOperation::Terminate,
            _ => ProcessOperation::Spawn,
        }
    }
}
impl std::fmt::Display for ProcessError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Process lifecycle failure: {self:?}")
    }
}
impl std::error::Error for ProcessError {}
/// Valid operating-system process identity observed at creation.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ManagedProcessId(u32);
impl ManagedProcessId {
    /// Constructs an identifier from an OS-reported value.
    #[must_use]
    pub const fn new(id: u32) -> Self {
        Self(id)
    }
    /// Returns the platform value.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}
/// Lifecycle behavior when the owned process is dropped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessDropPolicy {
    /// Request nonblocking termination; Windows applies this to its whole job.
    TerminateOnDrop,
    /// Leave the process running after handles close.
    DetachOnDrop,
}
/// Existing bounded polling policy, interpreted as attempts times delay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessWaitPolicy {
    attempts: usize,
    delay: Duration,
}
impl ProcessWaitPolicy {
    /// Creates a policy; zero attempts becomes one observation.
    #[must_use]
    pub const fn new(attempts: usize, delay: Duration) -> Self {
        Self {
            attempts: if attempts == 0 { 1 } else { attempts },
            delay,
        }
    }
    pub(super) fn duration(self) -> ProcessResult<Duration> {
        self.delay
            .checked_mul(
                u32::try_from(self.attempts).map_err(|_| ProcessError::InvalidSpecification)?,
            )
            .ok_or(ProcessError::InvalidSpecification)
    }
}
/// Process command, environment, and pipe policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessSpec {
    pub(super) program: PathBuf,
    pub(super) args: Vec<OsString>,
    pub(super) envs: Vec<(OsString, OsString)>,
    pub(super) clear_environment: bool,
    pub(super) piped: bool,
    pub(super) require_tree: bool,
}
impl ProcessSpec {
    /// Creates a command inheriting its environment and standard handles.
    #[must_use]
    pub fn new(program: impl Into<PathBuf>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            envs: Vec::new(),
            clear_environment: false,
            piped: false,
            require_tree: false,
        }
    }
    /// Appends one argument.
    #[must_use]
    pub fn arg(mut self, arg: impl Into<OsString>) -> Self {
        self.args.push(arg.into());
        self
    }
    /// Appends arguments without shell interpretation.
    #[must_use]
    pub fn args<I, A>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = A>,
        A: Into<OsString>,
    {
        self.args.extend(args.into_iter().map(Into::into));
        self
    }
    /// Sets a child environment variable.
    #[must_use]
    pub fn env(mut self, key: impl Into<OsString>, value: impl Into<OsString>) -> Self {
        self.envs.push((key.into(), value.into()));
        self
    }
    /// Starts from an empty environment before applying explicit variables.
    #[must_use]
    pub const fn env_clear(mut self) -> Self {
        self.clear_environment = true;
        self
    }
    /// Creates owned parent-side stdin/stdout pipes, inheriting stderr.
    #[must_use]
    pub const fn piped_stdio(mut self) -> Self {
        self.piped = true;
        self
    }
    /// Requires containment of normally created descendants; rejects unsupported targets.
    #[must_use]
    pub const fn tree_containment(mut self) -> Self {
        self.require_tree = true;
        self
    }
}
/// Successful or unsuccessful OS process termination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessOutcome {
    /// Successful exit status.
    Succeeded,
    /// Nonzero exit or platform termination status.
    Failed,
}
/// Confirmed process exit status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessStatus {
    /// OS process identity.
    pub id: ManagedProcessId,
    /// Portable exit code when available.
    pub code: Option<i32>,
    /// Classified completion.
    pub outcome: ProcessOutcome,
    exit_status: ExitStatus,
}
impl ProcessStatus {
    pub(super) fn from_exit_status(id: ManagedProcessId, exit_status: ExitStatus) -> Self {
        Self {
            id,
            code: exit_status.code(),
            outcome: if exit_status.success() {
                ProcessOutcome::Succeeded
            } else {
                ProcessOutcome::Failed
            },
            exit_status,
        }
    }
    /// Returns the original platform exit status without numeric reinterpretation.
    #[must_use]
    pub const fn exit_status(self) -> ExitStatus {
        self.exit_status
    }
}

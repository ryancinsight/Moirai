//! OS process lifecycle primitives for transport-backed execution.

#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

use std::{
    ffi::OsString,
    path::PathBuf,
    process::{Child, Command, ExitStatus},
    thread,
    time::Duration,
};

/// Result type for process lifecycle operations.
pub type ProcessResult<T> = Result<T, ProcessError>;

/// Process lifecycle error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessError {
    /// Process spawn failed.
    SpawnFailed,
    /// Process wait failed.
    WaitFailed,
    /// Process termination failed.
    TerminateFailed,
}

/// Managed OS process identifier.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ManagedProcessId(u32);

impl ManagedProcessId {
    /// Construct a managed process id.
    pub const fn new(id: u32) -> Self {
        Self(id)
    }

    /// Return the raw OS process id.
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// Process drop behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessDropPolicy {
    /// Kill a still-running child on drop.
    TerminateOnDrop,
    /// Leave a still-running child detached on drop.
    DetachOnDrop,
}

/// Bounded wait policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessWaitPolicy {
    attempts: usize,
    delay: Duration,
}

impl ProcessWaitPolicy {
    /// Construct a bounded wait policy.
    pub const fn new(attempts: usize, delay: Duration) -> Self {
        Self {
            attempts: if attempts == 0 { 1 } else { attempts },
            delay,
        }
    }
}

/// OS process command specification.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProcessSpec {
    program: PathBuf,
    args: Vec<OsString>,
    envs: Vec<(OsString, OsString)>,
}

impl ProcessSpec {
    /// Construct a process specification.
    pub fn new(program: impl Into<PathBuf>) -> Self {
        Self {
            program: program.into(),
            args: Vec::new(),
            envs: Vec::new(),
        }
    }

    /// Add one command-line argument.
    pub fn arg(mut self, arg: impl Into<OsString>) -> Self {
        self.args.push(arg.into());
        self
    }

    /// Add command-line arguments.
    pub fn args<I, A>(mut self, args: I) -> Self
    where
        I: IntoIterator<Item = A>,
        A: Into<OsString>,
    {
        self.args.extend(args.into_iter().map(Into::into));
        self
    }

    /// Add one environment variable.
    pub fn env(mut self, key: impl Into<OsString>, value: impl Into<OsString>) -> Self {
        self.envs.push((key.into(), value.into()));
        self
    }
}

/// Completed process status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessStatus {
    /// OS process id.
    pub id: ManagedProcessId,
    /// Process exit code when available.
    pub code: Option<i32>,
    /// Process completion outcome.
    pub outcome: ProcessOutcome,
}

/// Process completion outcome.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessOutcome {
    /// Process exited successfully.
    Succeeded,
    /// Process exited with a non-success status or platform-specific termination status.
    Failed,
}

impl ProcessStatus {
    fn from_exit_status(id: ManagedProcessId, status: ExitStatus) -> Self {
        Self {
            id,
            code: status.code(),
            outcome: if status.success() {
                ProcessOutcome::Succeeded
            } else {
                ProcessOutcome::Failed
            },
        }
    }
}

/// Process lifecycle supervisor.
#[derive(Debug, Clone, Copy, Default)]
pub struct ProcessSupervisor;

impl ProcessSupervisor {
    /// Construct a process supervisor.
    pub const fn new() -> Self {
        Self
    }

    /// Spawn a process from a specification.
    pub fn spawn(
        &self,
        spec: ProcessSpec,
        drop_policy: ProcessDropPolicy,
    ) -> ProcessResult<ManagedProcess> {
        let mut command = Command::new(spec.program);
        command.args(spec.args);
        command.envs(spec.envs);
        let child = command.spawn().map_err(|_| ProcessError::SpawnFailed)?;
        let id = ManagedProcessId::new(child.id());
        Ok(ManagedProcess {
            child,
            id,
            drop_policy,
            completed: false,
        })
    }
}

/// Managed process handle.
#[derive(Debug)]
pub struct ManagedProcess {
    child: Child,
    id: ManagedProcessId,
    drop_policy: ProcessDropPolicy,
    completed: bool,
}

impl ManagedProcess {
    /// Return the process id.
    pub const fn id(&self) -> ManagedProcessId {
        self.id
    }

    /// Try to observe process completion without blocking.
    pub fn try_wait(&mut self) -> ProcessResult<Option<ProcessStatus>> {
        let status = self
            .child
            .try_wait()
            .map_err(|_| ProcessError::WaitFailed)?;
        Ok(status.map(|status| {
            self.completed = true;
            ProcessStatus::from_exit_status(self.id, status)
        }))
    }

    /// Wait until process completion.
    pub fn wait(&mut self) -> ProcessResult<ProcessStatus> {
        let status = self.child.wait().map_err(|_| ProcessError::WaitFailed)?;
        self.completed = true;
        Ok(ProcessStatus::from_exit_status(self.id, status))
    }

    /// Wait for process completion under a bounded polling policy.
    pub fn wait_bounded(
        &mut self,
        policy: ProcessWaitPolicy,
    ) -> ProcessResult<Option<ProcessStatus>> {
        for _ in 0..policy.attempts {
            if let Some(status) = self.try_wait()? {
                return Ok(Some(status));
            }
            thread::sleep(policy.delay);
        }

        Ok(None)
    }

    /// Terminate the process if it is still running.
    pub fn terminate(&mut self) -> ProcessResult<Option<ProcessStatus>> {
        if let Some(status) = self.try_wait()? {
            return Ok(Some(status));
        }

        self.child
            .kill()
            .map_err(|_| ProcessError::TerminateFailed)?;
        self.wait().map(Some)
    }
}

impl Drop for ManagedProcess {
    fn drop(&mut self) {
        if self.completed || self.drop_policy == ProcessDropPolicy::DetachOnDrop {
            return;
        }

        if matches!(self.child.try_wait(), Ok(None)) {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ProcessDropPolicy, ProcessOutcome, ProcessSpec, ProcessSupervisor, ProcessWaitPolicy,
    };
    use std::time::Duration;

    #[test]
    fn process_supervisor_waits_for_successful_child() {
        let supervisor = ProcessSupervisor::new();
        let spec = ProcessSpec::new(std::env::current_exe().unwrap()).arg("--help");
        let mut process = supervisor
            .spawn(spec, ProcessDropPolicy::TerminateOnDrop)
            .unwrap();
        let process_id = process.id();
        let status = process.wait().unwrap();

        assert_eq!(status.id, process_id);
        assert_eq!(status.outcome, ProcessOutcome::Succeeded);
        assert_eq!(status.code, Some(0));
    }

    #[test]
    fn process_supervisor_times_out_and_terminates_child() {
        let supervisor = ProcessSupervisor::new();
        let spec = ProcessSpec::new(std::env::current_exe().unwrap()).args([
            "--ignored",
            "--exact",
            "process::tests::process_supervisor_child_waits_until_terminated",
            "--nocapture",
        ]);
        let mut process = supervisor
            .spawn(spec, ProcessDropPolicy::TerminateOnDrop)
            .unwrap();

        let observed = process
            .wait_bounded(ProcessWaitPolicy::new(5, Duration::from_millis(1)))
            .unwrap();
        assert_eq!(observed, None);

        let status = process.terminate().unwrap().unwrap();
        assert_eq!(status.id, process.id());
        assert_eq!(status.outcome, ProcessOutcome::Failed);
    }

    #[test]
    #[ignore]
    fn process_supervisor_child_waits_until_terminated() {
        std::thread::sleep(Duration::from_secs(30));
    }
}

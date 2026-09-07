//! Direct-child lifecycle for targets without Windows job objects.
use super::{ProcessDropPolicy, ProcessError, ProcessOperation, ProcessResult, ProcessSpec};
use std::{
    fs::File,
    process::{Child, Command, ExitStatus, Stdio},
    time::{Duration, Instant},
};
#[derive(Debug)]
pub(super) struct Process {
    child: Child,
    drop_policy: ProcessDropPolicy,
    pub stdin: Option<File>,
    pub stdout: Option<File>,
}
impl Process {
    pub fn spawn(spec: ProcessSpec, drop_policy: ProcessDropPolicy) -> ProcessResult<Self> {
        if spec.require_tree {
            return Err(ProcessError::UnsupportedContainment);
        }
        #[cfg(not(unix))]
        if spec.piped {
            return Err(ProcessError::InvalidSpecification);
        }
        let mut command = Command::new(spec.program);
        command.args(spec.args);
        if spec.clear_environment {
            command.env_clear();
        }
        command.envs(spec.envs);
        if spec.piped {
            command.stdin(Stdio::piped()).stdout(Stdio::piped());
        }
        let mut child = command
            .spawn()
            .map_err(|error| os_error(ProcessOperation::Spawn, &error))?;
        #[cfg(unix)]
        let stdin = child
            .stdin
            .take()
            .map(|pipe| File::from(std::os::fd::OwnedFd::from(pipe)));
        #[cfg(unix)]
        let stdout = child
            .stdout
            .take()
            .map(|pipe| File::from(std::os::fd::OwnedFd::from(pipe)));
        #[cfg(not(unix))]
        let (stdin, stdout) = (None, None);
        Ok(Self {
            child,
            drop_policy,
            stdin,
            stdout,
        })
    }
    pub fn id(&self) -> u32 {
        self.child.id()
    }
    pub fn try_wait(&mut self) -> ProcessResult<Option<ExitStatus>> {
        self.child
            .try_wait()
            .map_err(|error| os_error(ProcessOperation::Wait, &error))
    }
    pub fn wait_timeout(&mut self, timeout: Duration) -> ProcessResult<Option<ExitStatus>> {
        let deadline = Instant::now()
            .checked_add(timeout)
            .ok_or(ProcessError::InvalidSpecification)?;
        loop {
            if let Some(status) = self.try_wait()? {
                return Ok(Some(status));
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Ok(None);
            }
            // Portable Child has no deadline wait. Poll only at this OS boundary.
            std::thread::sleep(remaining.min(Duration::from_millis(1)));
        }
    }
    pub fn terminate(&mut self) -> ProcessResult<()> {
        if self.try_wait()?.is_some() {
            return Ok(());
        }
        self.child
            .kill()
            .map_err(|error| os_error(ProcessOperation::Terminate, &error))
    }

    pub fn terminate_timeout(&mut self, timeout: Duration) -> ProcessResult<Option<ExitStatus>> {
        self.terminate()?;
        self.wait_timeout(timeout)
    }
}
impl Drop for Process {
    fn drop(&mut self) {
        if self.drop_policy == ProcessDropPolicy::TerminateOnDrop {
            // Portable Drop cannot report OS failure and must not block. This
            // is explicitly best effort; callers requiring a confirmed outcome
            // use terminate_timeout, which retains all errors. No reaping wait
            // follows this last-resort request.
            drop(self.child.kill());
        }
    }
}
fn os_error(operation: ProcessOperation, error: &std::io::Error) -> ProcessError {
    ProcessError::OperatingSystem {
        operation,
        code: error.raw_os_error(),
    }
}

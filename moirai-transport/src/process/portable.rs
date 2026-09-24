//! Child lifecycle for targets without Windows job objects.
//!
//! Linux, Android and Apple targets contain a requested process tree in a
//! POSIX process group (see [`group`]); other targets reject containment.
use super::{ProcessDropPolicy, ProcessError, ProcessOperation, ProcessResult, ProcessSpec};

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
#[expect(
    unsafe_code,
    reason = "Reviewed POSIX boundary: waitid(WNOWAIT) observation and killpg"
)]
mod group;
use std::{
    fs::File,
    process::{Child, Command, ExitStatus, Stdio},
    time::{Duration, Instant},
};
#[derive(Debug)]
pub(super) struct Process {
    child: Child,
    drop_policy: ProcessDropPolicy,
    /// Whether the child leads a contained process group.
    contained: bool,
    /// The contained leader's exit, observed but not yet reaped.
    #[cfg_attr(
        not(any(target_os = "linux", target_os = "android", target_vendor = "apple")),
        expect(
            dead_code,
            reason = "only group-containment targets observe without reaping"
        )
    )]
    exited: Option<ExitStatus>,
    pub stdin: Option<File>,
    pub stdout: Option<File>,
    pub stderr: Option<File>,
}
impl Process {
    pub fn spawn(spec: ProcessSpec, drop_policy: ProcessDropPolicy) -> ProcessResult<Self> {
        if spec.require_tree && !CONTAINMENT {
            return Err(ProcessError::UnsupportedContainment);
        }
        #[cfg(not(unix))]
        if spec.piped || spec.piped_stderr {
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
        if spec.piped_stderr {
            command.stderr(Stdio::piped());
        }
        #[cfg(unix)]
        if spec.require_tree {
            use std::os::unix::process::CommandExt;
            // The child becomes the leader of a new group that its normally
            // created descendants join.
            command.process_group(0);
        }
        let child = command
            .spawn()
            .map_err(|error| os_error(ProcessOperation::Spawn, &error))?;
        #[cfg(unix)]
        let mut child = child;
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
        #[cfg(unix)]
        let stderr = child
            .stderr
            .take()
            .map(|pipe| File::from(std::os::fd::OwnedFd::from(pipe)));
        #[cfg(not(unix))]
        let (stdin, stdout, stderr) = (None, None, None);
        Ok(Self {
            child,
            drop_policy,
            contained: spec.require_tree,
            exited: None,
            stdin,
            stdout,
            stderr,
        })
    }
    pub fn id(&self) -> u32 {
        self.child.id()
    }
    pub fn try_wait(&mut self) -> ProcessResult<Option<ExitStatus>> {
        #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
        if self.contained {
            // The leader stays unreaped so its group ID stays reserved.
            if self.exited.is_none() {
                self.exited = group::observe_exit(self.child.id())
                    .map_err(|error| os_error(ProcessOperation::Wait, &error))?;
            }
            return Ok(self.exited);
        }
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
        #[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
        if self.contained {
            // Signal the group even after the leader exits: descendants it
            // left behind are still members.
            return group::kill_group(self.child.id())
                .map_err(|error| os_error(ProcessOperation::Terminate, &error));
        }
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
            // use terminate_timeout, which retains all errors.
            if self.contained {
                let _ = self.terminate();
            } else {
                drop(self.child.kill());
            }
        }
        if self.contained {
            // Reap a leader that has already exited, releasing the group ID
            // only after the group was signalled; a leader still dying is
            // reaped by the OS when this process exits, as before.
            drop(self.child.try_wait());
        }
    }
}
/// Whether this target implements [`ProcessSpec::tree_containment`].
const CONTAINMENT: bool = cfg!(any(
    target_os = "linux",
    target_os = "android",
    target_vendor = "apple"
));

fn os_error(operation: ProcessOperation, error: &std::io::Error) -> ProcessError {
    ProcessError::OperatingSystem {
        operation,
        code: error.raw_os_error(),
    }
}

//! OS process lifecycle primitives, owned pipes, and bounded cleanup.
//!
//! # Examples
//!
//! A real worker executable supplies the protocol over the requested pipes.
//! The caller bounds pipe I/O independently from process completion.
//!
//! ```no_run
//! use moirai_transport::process::{ProcessDropPolicy, ProcessSpec, ProcessSupervisor};
//! use std::time::Duration;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let spec = ProcessSpec::new("worker.exe")
//!     .env_clear()
//!     .piped_stdio()
//!     .tree_containment();
//! let mut process = ProcessSupervisor::new().spawn(spec, ProcessDropPolicy::TerminateOnDrop)?;
//! let completion = process.wait_timeout(Duration::from_secs(2))?;
//! let status = match completion {
//!     Some(status) => status,
//!     None => process.terminate_timeout(Duration::from_secs(1))?,
//! };
//! println!("{}", status.exit_status());
//! # Ok(())
//! # }
//! ```
#[cfg(not(windows))]
mod portable;
mod types;
#[cfg(windows)]
#[expect(
    unsafe_code,
    reason = "Reviewed Win32 ABI boundary owns process, job and pipe handles"
)]
mod windows;
#[cfg(not(windows))]
use portable::Process;
use std::{fs::File, time::Duration};
pub use types::{
    ManagedProcessId, ProcessDropPolicy, ProcessError, ProcessOperation, ProcessOutcome,
    ProcessResult, ProcessSpec, ProcessStatus, ProcessWaitPolicy,
};
#[cfg(windows)]
use windows::Process;

/// Process lifecycle supervisor.
#[derive(Debug, Clone, Copy, Default)]
pub struct ProcessSupervisor;
impl ProcessSupervisor {
    /// Constructs a supervisor without spawning a process.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
    /// Spawns a process with the requested environment and pipe ownership.
    ///
    /// Windows uses atomic job assignment; other targets reject explicitly
    /// requested process-tree containment rather than silently weakening it.
    /// # Errors
    /// Returns invalid specification, unsupported containment, or OS failures.
    pub fn spawn(
        &self,
        spec: ProcessSpec,
        drop_policy: ProcessDropPolicy,
    ) -> ProcessResult<ManagedProcess> {
        let process = Process::spawn(spec, drop_policy)?;
        let id = ManagedProcessId::new(process.id());
        Ok(ManagedProcess { process, id })
    }
}
/// Managed process and its optional parent-side pipe handles.
///
/// Windows kill-on-close jobs terminate descendants without waiting in Drop.
/// Other platforms provide direct-child termination only; explicit cleanup
/// reports errors, while Drop is a nonblocking last-resort termination request.
#[derive(Debug)]
pub struct ManagedProcess {
    process: Process,
    id: ManagedProcessId,
}
impl ManagedProcess {
    /// Returns the OS process identifier.
    #[must_use]
    pub const fn id(&self) -> ManagedProcessId {
        self.id
    }
    /// Transfers the parent-side stdin writer once, when pipes were requested.
    pub fn take_stdin(&mut self) -> Option<File> {
        self.process.stdin.take()
    }
    /// Transfers the parent-side stdout reader once, when pipes were requested.
    pub fn take_stdout(&mut self) -> Option<File> {
        self.process.stdout.take()
    }
    /// Checks for process completion without blocking.
    /// # Errors
    /// Returns an OS wait failure.
    pub fn try_wait(&mut self) -> ProcessResult<Option<ProcessStatus>> {
        Ok(self
            .process
            .try_wait()?
            .map(|status| ProcessStatus::from_exit_status(self.id, status)))
    }
    /// Waits under the default finite 30-second lifecycle budget.
    /// # Errors
    /// Returns an OS failure or `DeadlineExceeded`.
    pub fn wait(&mut self) -> ProcessResult<ProcessStatus> {
        self.wait_timeout(Duration::from_secs(30))?
            .ok_or(ProcessError::DeadlineExceeded)
    }
    /// Waits for at most the given finite duration; expiry returns None.
    /// # Errors
    /// Returns an unsupported duration or OS wait failure.
    pub fn wait_timeout(&mut self, timeout: Duration) -> ProcessResult<Option<ProcessStatus>> {
        Ok(self
            .process
            .wait_timeout(timeout)?
            .map(|status| ProcessStatus::from_exit_status(self.id, status)))
    }
    /// Applies the existing attempt/delay policy as one finite time budget.
    /// # Errors
    /// Returns an overflowing budget or OS wait failure.
    pub fn wait_bounded(
        &mut self,
        policy: ProcessWaitPolicy,
    ) -> ProcessResult<Option<ProcessStatus>> {
        self.wait_timeout(policy.duration()?)
    }
    /// Terminates the process and waits under a finite one-second cleanup budget.
    /// # Errors
    /// Returns an OS failure or `DeadlineExceeded`.
    pub fn terminate(&mut self) -> ProcessResult<Option<ProcessStatus>> {
        self.terminate_timeout(Duration::from_secs(1)).map(Some)
    }
    /// Requests termination and waits for confirmation within the supplied budget.
    ///
    /// Windows terminates the entire job even when its root process has exited.
    /// # Errors
    /// Returns termination/wait failures or unconfirmed cleanup at the deadline.
    pub fn terminate_timeout(&mut self, timeout: Duration) -> ProcessResult<ProcessStatus> {
        self.process
            .terminate_timeout(timeout)?
            .map(|status| ProcessStatus::from_exit_status(self.id, status))
            .ok_or(ProcessError::DeadlineExceeded)
    }
}
#[cfg(test)]
mod tests;

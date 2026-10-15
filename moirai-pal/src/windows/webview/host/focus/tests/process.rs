//! Process isolation and OS-observed collection for the real UIA client.
//!
//! Only `try_wait` returning an exit status changes Running to Collected.
//! Termination admission is not collection. Close performs no blocking wait,
//! attempts termination at most once, and retains every operation failure.
//! The caller must observe `is_collected()` before removing the owned profile;
//! closing the host may unblock UIA, but it cannot prove child termination.

use crate::windows::webview::WebViewHost;
use std::{
    error::Error,
    fmt,
    io::{self, Read},
    process::{Child, Command, ExitStatus, Stdio},
    time::{Duration, Instant},
};

#[derive(Clone, Copy, Debug)]
enum Collection {
    Running,
    Collected(ExitStatus),
}

pub(super) struct NativeClient {
    child: Child,
    collection: Collection,
    termination_attempted: bool,
}

#[derive(Debug)]
struct CollectionFailure {
    operation: &'static str,
    error: io::Error,
}

#[derive(Debug)]
struct CollectionErrors {
    process_id: u32,
    collected: bool,
    failures: Vec<CollectionFailure>,
}

impl fmt::Display for CollectionErrors {
    fn fmt(&self, output: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            output,
            "actual UIA client {} collection observed={}",
            self.process_id, self.collected
        )?;
        for failure in &self.failures {
            write!(output, "; {}: {}", failure.operation, failure.error)?;
        }
        Ok(())
    }
}

impl Error for CollectionErrors {
    // The standard error-source ABI erases the actual error type at this cold
    // diagnostic boundary; every original io::Error remains owned in failures.
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        self.failures.first().map(|failure| &failure.error as _)
    }
}

impl NativeClient {
    pub(super) fn spawn(handle: usize, mode: super::ObservationMode) -> io::Result<Self> {
        Command::new(std::env::current_exe()?)
            .args(["--exact", super::TEST_NAME, "--ignored", "--nocapture"])
            .env(super::CLIENT_ENV, handle.to_string())
            .env(super::MODE_ENV, mode.as_str())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map(|child| Self {
                child,
                collection: Collection::Running,
                termination_attempted: false,
            })
    }

    pub(super) fn wait(
        &mut self,
        host: &mut WebViewHost,
        deadline: Instant,
    ) -> io::Result<ExitStatus> {
        loop {
            if let Some(status) = self.observe_exit()? {
                return Ok(status);
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "actual MTA UIA client exceeded 30 seconds",
                ));
            }
            host.wait_events(remaining.min(Duration::from_millis(25)))?;
        }
    }

    pub(super) fn close(&mut self) -> io::Result<()> {
        let mut failures = Vec::new();
        match self.observe_exit() {
            Ok(Some(_)) => return Ok(()),
            Ok(None) => {}
            Err(error) => {
                failures.push(CollectionFailure {
                    operation: "Child::try_wait before termination",
                    error,
                });
                return self.collection_result(failures);
            }
        }
        if !self.termination_attempted {
            self.termination_attempted = true;
            if let Err(error) = self.child.kill() {
                failures.push(CollectionFailure {
                    operation: "Child::kill",
                    error,
                });
            }
        }
        match self.observe_exit() {
            Ok(Some(_)) => {}
            Ok(None) => failures.push(CollectionFailure {
                operation: "Child::try_wait after termination attempt",
                error: io::Error::new(
                    io::ErrorKind::WouldBlock,
                    "native UIA client termination remains unconfirmed",
                ),
            }),
            Err(error) => failures.push(CollectionFailure {
                operation: "Child::try_wait after termination attempt",
                error,
            }),
        }
        self.collection_result(failures)
    }

    pub(super) fn is_collected(&self) -> bool {
        matches!(self.collection, Collection::Collected(_))
    }

    fn observe_exit(&mut self) -> io::Result<Option<ExitStatus>> {
        if let Collection::Collected(status) = self.collection {
            return Ok(Some(status));
        }
        let status = self.child.try_wait()?;
        if let Some(status) = status {
            self.collection = Collection::Collected(status);
        }
        Ok(status)
    }

    fn collection_result(&self, failures: Vec<CollectionFailure>) -> io::Result<()> {
        if failures.is_empty() {
            return Ok(());
        }
        for failure in &failures {
            tracing::error!(
                process_id = self.child.id(),
                operation = failure.operation,
                kind = ?failure.error.kind(),
                raw_os_error = ?failure.error.raw_os_error(),
                error = %failure.error,
                "actual UIA client collection operation failed"
            );
        }
        Err(io::Error::other(CollectionErrors {
            process_id: self.child.id(),
            collected: self.is_collected(),
            failures,
        }))
    }

    pub(super) fn completion(&mut self) -> io::Result<super::client::Completion> {
        if !self.is_collected() {
            return Err(io::Error::other(
                "native UIA client has no OS-observed collection",
            ));
        }
        // Semantic assertions belong to the parent after explicit host/client
        // cleanup. Parsing here returns errors without unwinding those owners.
        self.read_completion()
    }

    fn read_completion(&mut self) -> io::Result<super::client::Completion> {
        // This client creates no browser or descendant process. Its only
        // stdout is libtest status plus a computed completion (<1 KiB for the
        // declared four-field domain); tracing uses inherited stderr. Exit
        // collection precedes this read, so its sole writer has terminated.
        let stdout = self
            .child
            .stdout
            .take()
            .ok_or_else(|| io::Error::other("UIA client stdout missing"))?;
        let mut bytes = Vec::new();
        stdout
            .take(crate::webview_limits::MAX_WEBVIEW_MESSAGE_BYTES as u64 + 1)
            .read_to_end(&mut bytes)?;
        if bytes.len() > crate::webview_limits::MAX_WEBVIEW_MESSAGE_BYTES {
            return Err(io::Error::other(
                "UIA client output exceeds the existing byte limit",
            ));
        }
        let text = std::str::from_utf8(&bytes).map_err(io::Error::other)?;
        let mut receipts = text
            .lines()
            .filter_map(|line| line.strip_prefix(super::COMPLETION_PREFIX));
        let receipt = receipts
            .next()
            .ok_or_else(|| io::Error::other(format!("computed UIA completion missing: {text}")))?;
        if receipts.next().is_some() {
            return Err(io::Error::other(format!(
                "multiple computed UIA completions: {text}"
            )));
        }
        serde_json::from_str(receipt).map_err(io::Error::other)
    }
}

impl Drop for NativeClient {
    fn drop(&mut self) {
        if let Err(error) = self.close() {
            tracing::error!(%error, "actual UIA client RAII cleanup failed");
        }
    }
}

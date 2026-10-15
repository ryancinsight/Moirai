//! Process isolation keeps a blocked UIA native call from outliving its owner.

use crate::windows::webview::WebViewHost;
use std::{
    io::{self, Read},
    process::{Child, Command, ExitStatus, Stdio},
    time::{Duration, Instant},
};

pub(super) struct NativeClient(Child);

impl NativeClient {
    pub(super) fn spawn(handle: usize) -> io::Result<Self> {
        Command::new(std::env::current_exe()?)
            .args(["--exact", super::TEST_NAME, "--ignored", "--nocapture"])
            .env(super::CLIENT_ENV, handle.to_string())
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .map(Self)
    }

    pub(super) fn wait(
        &mut self,
        host: &mut WebViewHost,
        deadline: Instant,
    ) -> io::Result<ExitStatus> {
        loop {
            if let Some(status) = self.0.try_wait()? {
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
        let mut first_error = None;
        let mut may_collect = match self.0.try_wait() {
            Ok(status) => status.is_some(),
            Err(error) => {
                first_error = Some(error);
                false
            }
        };
        if !may_collect {
            match self.0.kill() {
                Ok(()) => may_collect = true,
                Err(error) => {
                    first_error.get_or_insert(error);
                }
            }
        }
        if may_collect {
            if let Err(error) = self.0.wait() {
                first_error.get_or_insert(error);
            }
        } else if let Err(error) = self.0.try_wait().and_then(|status| {
            status.ok_or_else(|| {
                io::Error::other("native UIA client termination remains unconfirmed")
            })
        }) {
            first_error.get_or_insert(error);
        }
        first_error.map_or(Ok(()), Err)
    }

    pub(super) fn completion(&mut self) -> io::Result<super::client::Completion> {
        if self.0.try_wait()?.is_none() {
            return Err(io::Error::other("native UIA client has not exited"));
        }
        // This client creates no browser or descendant process. Its only
        // stdout is libtest status plus a computed completion (<1 KiB for the
        // declared four-field domain); tracing uses inherited stderr.
        let stdout = self
            .0
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
        let receipts: Vec<_> = text
            .lines()
            .filter_map(|line| line.strip_prefix(super::COMPLETION_PREFIX))
            .collect();
        if receipts.len() != 1 {
            return Err(io::Error::other(format!(
                "expected one computed UIA completion, received {}: {text}",
                receipts.len()
            )));
        }
        serde_json::from_str(receipts[0]).map_err(io::Error::other)
    }
}

impl Drop for NativeClient {
    fn drop(&mut self) {
        if let Err(error) = self.close() {
            tracing::error!(%error, "actual UIA client RAII cleanup failed");
        }
    }
}

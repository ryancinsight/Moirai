//! Process isolation keeps a blocked UIA native call from outliving its owner.

use crate::windows::webview::WebViewHost;
use std::{
    io,
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
            .stdout(Stdio::inherit())
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
        if self.0.try_wait()?.is_none() {
            self.0.kill()?;
        }
        self.0.wait()?;
        Ok(())
    }
}

impl Drop for NativeClient {
    fn drop(&mut self) {
        if let Err(error) = self.close() {
            tracing::error!(%error, "actual UIA client RAII cleanup failed");
        }
    }
}

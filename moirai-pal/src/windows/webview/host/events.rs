//! Application waits own fresh sampling; private wakes only settle current work.

use super::super::{
    event::WebViewHostEvent,
    pump::{dispatch_pending, wait_deadline, wait_for_messages},
};
use super::view::WebViewHost;
use std::{
    io,
    time::{Duration, Instant},
};
use windows::Win32::UI::WindowsAndMessaging::MSG;

impl WebViewHost {
    /// Pumps pending owner-thread messages and returns native and WebView2
    /// events in one bounded batch.
    ///
    /// # Errors
    /// Returns queue overflow, callback decoding, native window or controller
    /// errors.
    pub fn poll_events(&mut self) -> io::Result<Vec<WebViewHostEvent>> {
        self.poll_with_sampling(true, None)
    }

    /// Waits for owner-thread input or a WebView2 event for a finite duration.
    ///
    /// Private continuous-observation completions advance within the same
    /// caller-supplied wait. They neither restart its deadline nor admit a
    /// second sample. Ready application events take precedence over outgoing
    /// diagnostics. SDK admission checks the original deadline after native
    /// dispatch and command preparation. Ordinary and zero-duration calls
    /// retain their existing polling behavior.
    ///
    /// # Errors
    /// Returns queue overflow, callback decoding, native window or controller
    /// errors, or an invalid duration.
    pub fn wait_events(&mut self, timeout: Duration) -> io::Result<Vec<WebViewHostEvent>> {
        if self.closed {
            return self.poll_events();
        }
        let entered = Instant::now();
        let deadline = match self.startup.as_ref().filter(|_| !timeout.is_zero()) {
            Some(startup) => {
                let deadline = wait_deadline(entered, timeout)?;
                if startup.repeating_samples() {
                    startup.begin_wait_turn()?;
                }
                Some(deadline)
            }
            _ => None,
        };
        let events = self.poll_with_sampling(true, deadline)?;
        if !events.is_empty() {
            return Ok(events);
        }
        let Some(deadline) = deadline else {
            if !wait_for_messages(timeout)? {
                return Ok(Vec::new());
            }
            return self.poll_events();
        };
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            if remaining.is_zero() || !wait_for_messages(remaining)? {
                return Ok(Vec::new());
            }
            // All native/COM messages still run. Existing DOM stages settle,
            // but an observation-only wake cannot rearm its own next chain.
            let events = self.poll_with_sampling(false, Some(deadline))?;
            if !events.is_empty() {
                return Ok(events);
            }
        }
    }

    pub(super) fn poll_with_sampling(
        &mut self,
        admit_new_sample: bool,
        deadline: Option<Instant>,
    ) -> io::Result<Vec<WebViewHostEvent>> {
        if self.closed {
            return self.collect_events(admit_new_sample, deadline);
        }
        let mut message = MSG::default();
        dispatch_pending(&mut message);
        self.collect_events(admit_new_sample, deadline)
    }
}

//! Opt-in live-value evidence, never a retry or an input acceptance substitute.
//!
//! One returned mismatch after the existing UIA evidence quota triggers one
//! asynchronous DOMSnapshot command, within the original parent deadline. It
//! owns a separate Weak callback state, retired before host Close. Snapshot
//! flushes layout: values describe its execution, not an atomic UIA/CDP read.

mod channel;
mod protocol;
pub(super) use channel::Emitter;

use super::super::super::diagnostics::FixtureInput;
use super::{ObservationMode, WebViewHost};
use crate::webview_startup::{SELECTION_ENV, Selection};
use std::{
    cell::{Cell, RefCell},
    io,
    process::ChildStdout,
    rc::{Rc, Weak},
    time::Instant,
};
use webview2_com::Microsoft::Web::WebView2::Win32::{
    ICoreWebView2CallDevToolsProtocolMethodCompletedHandler,
    ICoreWebView2CallDevToolsProtocolMethodCompletedHandler_Impl,
};
use windows::core::{HRESULT, HSTRING, PCWSTR, implement};

#[derive(Default)]
struct Reply {
    closed: bool,
    response: Option<(Instant, HRESULT, windows::core::Result<String>)>,
    failure: Option<windows::core::Error>,
}

#[implement(ICoreWebView2CallDevToolsProtocolMethodCompletedHandler)]
struct Callback {
    completed: Cell<bool>,
    owner: Weak<RefCell<Reply>>,
}

impl ICoreWebView2CallDevToolsProtocolMethodCompletedHandler_Impl for Callback_Impl {
    fn Invoke(&self, status: HRESULT, encoded: &PCWSTR) -> windows::core::Result<()> {
        let received = Instant::now();
        let Some(owner) = self.owner.upgrade() else {
            return Ok(());
        };
        let mut owner = owner.try_borrow_mut().map_err(|_| {
            super::super::super::error::callback_error("renderer probe state borrowed")
        })?;
        if owner.closed {
            return Ok(());
        }
        if self.completed.replace(true) {
            let error =
                super::super::super::error::callback_error("renderer probe callback invoked twice");
            owner.failure = Some(error.clone());
            return Err(error);
        }
        // SAFETY: SDK owns the terminated callback payload through Invoke.
        // Existing UTF-16/UTF-8 bounds apply before its first retained copy.
        let value = unsafe { super::super::super::text::read_borrowed_message(encoded) };
        owner.response = Some((received, status, value));
        Ok(())
    }
}

pub(super) struct Probe {
    handle: usize,
    mode: ObservationMode,
    started: Instant,
    channel: channel::Channel,
    controller: FixtureInput,
    reply: Rc<RefCell<Reply>>,
    callback: Option<ICoreWebView2CallDevToolsProtocolMethodCompletedHandler>,
    triggered: bool,
    attempted: bool,
    admitted: bool,
    completed: bool,
    deadline: Option<Instant>,
    late: bool,
    error: Option<io::Error>,
}

impl Probe {
    pub(super) fn new(host: &WebViewHost, mode: ObservationMode) -> io::Result<Option<Self>> {
        if !channel::enabled() {
            return Ok(None);
        }
        let started = Instant::now();
        let encoded = std::env::var(SELECTION_ENV).map_err(io::Error::other)?;
        let selection = Selection::decode(&encoded, host.config.start_uri())?;
        let controller = host
            .controller
            .as_ref()
            .ok_or_else(|| io::Error::other("controller probe has no actual controller"))?;
        let controller = FixtureInput::new(controller, selection)?;
        Ok(Some(Self {
            handle: host.window.hwnd.0 as usize,
            mode,
            started,
            channel: channel::Channel::default(),
            controller,
            reply: Rc::default(),
            callback: None,
            triggered: false,
            attempted: false,
            admitted: false,
            completed: false,
            deadline: None,
            late: false,
            error: None,
        }))
    }

    pub(super) fn poll(&mut self, stdout: &mut ChildStdout, host: &WebViewHost, deadline: Instant) {
        // Callbacks have returned before this owner-thread flush. This separate
        // quota never retires or exhausts the still-active startup sampler.
        self.controller.flush();
        if self.error.is_some() {
            return;
        }
        if let Err(error) = self.advance(stdout, host, deadline) {
            tracing::error!(event="uia_renderer_probe_error", error=%error, "live renderer observation failed");
            if self.error.is_none() {
                self.error = Some(error);
            }
        }
    }

    fn advance(
        &mut self,
        stdout: &mut ChildStdout,
        host: &WebViewHost,
        deadline: Instant,
    ) -> io::Result<()> {
        if let Some(progress) = self.channel.poll(stdout)? {
            progress.validate(self.handle, self.mode)?;
            let (origin, sequence) = self.controller.checkpoint()?;
            let origin = origin
                .checked_duration_since(self.started)
                .ok_or_else(|| io::Error::other("controller probe origin precedes parent probe"))?;
            tracing::info!(event="uia_controller_probe_checkpoint", sequence=%sequence,
                queue_origin_parent_ns=%origin.as_nanos(),
                parent_elapsed_ns=%self.started.elapsed().as_nanos(),
                "callback sequence at returned UIA mismatch; both origins belong to this parent clock");
            tracing::info!(event="uia_renderer_probe_trigger", progress=?progress,
                parent_elapsed_ns=%self.started.elapsed().as_nanos(), "returned native mismatch point observation");
            if self.triggered {
                return Err(io::Error::other("renderer probe already attempted"));
            }
            self.triggered = true;
            let callback: ICoreWebView2CallDevToolsProtocolMethodCompletedHandler = Callback {
                completed: Cell::new(false),
                owner: Rc::downgrade(&self.reply),
            }
            .into();
            self.callback = Some(callback.clone());
            let webview = host
                .webview
                .as_ref()
                .ok_or_else(|| io::Error::other("renderer probe has no actual WebView"))?;
            let method = HSTRING::from("DOMSnapshot.captureSnapshot");
            let parameters = HSTRING::from(
                r#"{"computedStyles":[],"includePaintOrder":false,"includeDOMRects":false}"#,
            );
            // Recheck immediately before SDK entry; setup never renews time.
            let entered = Instant::now();
            if entered >= deadline {
                return Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    "renderer probe not admitted before original deadline",
                ));
            }
            // Count exactly one actual SDK entry, including failed admission.
            self.attempted = true;
            self.deadline = Some(deadline);
            // SAFETY: the STA owns Core; SDK copies strings and retains the
            // callback, which holds no Core/controller or strong owner cycle.
            let result =
                unsafe { webview.CallDevToolsProtocolMethod(&method, &parameters, &callback) };
            // Output follows SDK entry; a blocked trace sink cannot reopen
            // admission after the final deadline check.
            tracing::info!(event="uia_renderer_probe_attempt", parent_elapsed_ns=%entered.duration_since(self.started).as_nanos(), "one actual DOMSnapshot command");
            self.admitted = result.is_ok();
            tracing::info!(
                event = "uia_renderer_probe_admission",
                hresult = result
                    .as_ref()
                    .err()
                    .map(|error| error.code().0)
                    .unwrap_or(0),
                "actual SDK admission status"
            );
            result.map_err(io::Error::other)?;
        }
        self.complete_response()
    }

    fn complete_response(&mut self) -> io::Result<()> {
        let failure = self.reply.borrow_mut().failure.take();
        if let Some(error) = failure {
            return Err(io::Error::other(error));
        }
        let response = self.reply.borrow_mut().response.take();
        if let Some((received, status, encoded)) = response {
            self.completed = true;
            let deadline = self
                .deadline
                .ok_or_else(|| io::Error::other("renderer callback has no admitted deadline"))?;
            self.late = !within_deadline(received, deadline);
            tracing::info!(event="uia_renderer_probe_completion", hresult=status.0,
                within_original_deadline=!self.late,
                parent_elapsed_ns=%received.duration_since(self.started).as_nanos(), "actual snapshot callback returned");
            let encoded = encoded.map_err(io::Error::other)?;
            // Preserve the bounded original body, including protocol errors.
            tracing::info!(event="uia_renderer_probe_payload", payload=%encoded);
            if self.late {
                tracing::info!(
                    event = "uia_renderer_probe_unobserved",
                    "snapshot callback arrives outside original parent deadline; raw reply is late evidence, not an admitted live-value observation"
                );
                return Ok(());
            }
            status.ok().map_err(io::Error::other)?;
            let snapshot = protocol::decode(&encoded)?;
            tracing::info!(event="uia_renderer_probe_values", snapshot=%serde_json::to_string(&snapshot).map_err(io::Error::other)?, "live INPUT values at snapshot execution");
        }
        Ok(())
    }

    pub(super) fn take_stdout(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.channel.bytes)
    }

    pub(super) fn validate_stdout(&mut self, bytes: &[u8]) -> io::Result<()> {
        if channel::validate_terminal(bytes, self.handle, self.mode)? && !self.triggered {
            tracing::info!(
                event = "uia_renderer_probe_unobserved",
                "native progress is first observed after child collection; no SDK command issued"
            );
        }
        Ok(())
    }

    pub(super) fn close(&mut self) -> io::Result<()> {
        // A callback may already have returned before try_wait observed exit.
        // Decode that owned response; never issue a new command after exit.
        if let Err(error) = self.complete_response() {
            tracing::error!(%error, "renderer probe returned diagnostic failure");
            if self.error.is_none() {
                self.error = Some(error);
            }
        }
        let mut reply = self.reply.borrow_mut();
        let retiring = !reply.closed;
        let retired_at = Instant::now();
        if retiring {
            reply.closed = true;
            reply.response = None;
            self.callback = None;
        }
        drop(reply);
        // No reply borrow crosses token-removal COM; the snapshot owner is
        // already closed, even if removal reenters a late completion.
        if let Err(error) = self.controller.close() {
            tracing::error!(%error, "independent controller probe close failed");
            if self.error.is_none() {
                self.error = Some(error);
            }
        }
        if retiring {
            tracing::info!(event="uia_renderer_probe_closed", triggered=self.triggered, attempted=self.attempted, admitted=self.admitted, completed=self.completed,
                late=self.late, pending=self.admitted && !self.completed, failed=self.error.is_some(),
                retired_parent_ns=%retired_at.duration_since(self.started).as_nanos(),
                parent_elapsed_ns=%self.started.elapsed().as_nanos(), "snapshot owner retires before controller token cleanup and host Close; final status retains cleanup errors");
        }
        self.error.take().map_or(Ok(()), Err)
    }
}

fn within_deadline(received: Instant, deadline: Instant) -> bool {
    received < deadline
}

#[test]
fn exact_parent_deadline_rejects_equal_and_later_callbacks() {
    let (_trace, _subscriber) = crate::windows::webview::tests::startup::tracing();
    let deadline = tracing::info_span!("arrange").in_scope(Instant::now);
    let tick = std::time::Duration::from_nanos(1);
    for (received, expected) in [
        (
            deadline
                .checked_sub(tick)
                .expect("representable prior tick"),
            true,
        ),
        (deadline, false),
        (
            deadline
                .checked_add(tick)
                .expect("representable later tick"),
            false,
        ),
    ] {
        let actual = tracing::info_span!("act").in_scope(|| within_deadline(received, deadline));
        tracing::info_span!("assert").in_scope(|| assert_eq!(actual, expected));
    }
}

impl Drop for Probe {
    fn drop(&mut self) {
        if let Err(error) = self.close() {
            tracing::error!(%error, "renderer probe cleanup retains diagnostic failure");
        }
    }
}

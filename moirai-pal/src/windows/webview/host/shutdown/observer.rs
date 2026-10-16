//! SDK exit evidence for this exclusively owned fresh-profile browser PID.

use std::{
    io,
    sync::mpsc::{self, Receiver},
    time::Duration,
};

use webview2_com::{
    BrowserProcessExitedEventHandler,
    Microsoft::Web::WebView2::Win32::{
        COREWEBVIEW2_BROWSER_PROCESS_EXIT_KIND_NORMAL, ICoreWebView2,
        ICoreWebView2BrowserProcessExitedEventHandler, ICoreWebView2Environment,
        ICoreWebView2Environment5,
    },
};
use windows::core::Interface;

use super::super::super::pump::wait_for;
use super::super::error::{callback_context, callback_error, windows_error};

pub(super) struct BrowserExit {
    environment: ICoreWebView2Environment5,
    receiver: Receiver<windows::core::Result<ExitObservation>>,
    token: i64,
    _handler: ICoreWebView2BrowserProcessExitedEventHandler,
}

enum ExitObservation {
    Exited(u32),
    Failed(windows::core::Error),
}

impl BrowserExit {
    pub(super) fn register(
        webview: &ICoreWebView2,
        environment: &ICoreWebView2Environment,
    ) -> io::Result<Self> {
        let mut expected = 0;
        // SAFETY: the host owns the live webview and exclusive PID output.
        unsafe { webview.BrowserProcessId(&mut expected) }
            .inspect_err(|error| getter_failure("webview.BrowserProcessId", error, None, None))
            .map_err(windows_error)?;
        if expected == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "WebView2 owned browser has an invalid process ID",
            ));
        }
        let environment = environment
            .cast::<ICoreWebView2Environment5>()
            .map_err(|error| {
                io::Error::other(format!("WebView2 owned-profile exit interface: {error}"))
            })?;
        let (sender, receiver) = mpsc::sync_channel(1);
        // Type erasure is the SDK's dynamically invoked COM callback boundary
        // (interop condition 2); no application execution policy is erased.
        let handler = BrowserProcessExitedEventHandler::create(Box::new(move |_sender, args| {
            let result = (|| {
                let args =
                    args.ok_or_else(|| callback_error("WebView2 browser exit omitted arguments"))?;
                let mut actual = 0;
                let mut kind = COREWEBVIEW2_BROWSER_PROCESS_EXIT_KIND_NORMAL;
                // SAFETY: these exclusive outputs live through the synchronous SDK getters.
                unsafe { args.BrowserProcessId(&mut actual) }
                    .inspect_err(|error| {
                        getter_failure("exit.BrowserProcessId", error, Some(expected), None);
                    })
                    .map_err(|error| callback_context("WebView2 exited browser PID", error))?;
                unsafe { args.BrowserProcessExitKind(&mut kind) }
                    .inspect_err(|error| {
                        getter_failure(
                            "exit.BrowserProcessExitKind",
                            error,
                            Some(expected),
                            Some(actual),
                        );
                    })
                    .map_err(|error| callback_context("WebView2 browser exit kind", error))?;
                // These are successful SDK getter outputs, not an OS exit code.
                // Retain them before either existing validation can reject them.
                tracing::info!(
                    event = "owned_browser_exit_observed",
                    expected_browser_process_id = expected,
                    browser_process_id = actual,
                    exit_kind = kind.0,
                    normal_exit = kind == COREWEBVIEW2_BROWSER_PROCESS_EXIT_KIND_NORMAL,
                    "Actual owned browser SDK exit classification"
                );
                if actual != expected {
                    return Err(callback_error(
                        "WebView2 owned profile reported a different browser PID",
                    ));
                }
                if kind != COREWEBVIEW2_BROWSER_PROCESS_EXIT_KIND_NORMAL {
                    return Err(callback_error("WebView2 owned browser exited abnormally"));
                }
                tracing::info!(
                    browser_process_id = actual,
                    "WebView2 owned profile browser resources released"
                );
                Ok(actual)
            })();
            let observation = match result {
                Ok(pid) => ExitObservation::Exited(pid),
                Err(error) => ExitObservation::Failed(error),
            };
            sender.try_send(Ok(observation)).map_err(|error| {
                windows::core::Error::new(
                    windows::Win32::Foundation::E_ABORT,
                    format!("WebView2 browser exit observation channel: {error}"),
                )
            })
        }));
        let mut token = 0;
        // SAFETY: the environment retains the COM-refcounted handler until
        // removal; this observer holds both owners through the bounded wait.
        unsafe { environment.add_BrowserProcessExited(&handler, &mut token) }
            .map_err(windows_error)?;
        Ok(Self {
            environment,
            receiver,
            token,
            _handler: handler,
        })
    }

    pub(super) fn wait(self, timeout: Duration) -> io::Result<()> {
        let result = wait_for(self.receiver, timeout)
            .and_then(|observation| match observation {
                ExitObservation::Exited(pid) => {
                    tracing::info!(
                        browser_process_id = pid,
                        "WebView2 owned browser shutdown observed"
                    );
                    Ok(())
                }
                ExitObservation::Failed(error) => Err(io::Error::other(format!(
                    "WebView2 owned browser exit observation: {error}"
                ))),
            })
            .map_err(|error| {
                io::Error::new(
                    error.kind(),
                    format!("WebView2 owned-profile browser shutdown: {error}"),
                )
            });
        // SAFETY: the environment remains live and token belongs to this
        // observer; the callback owner is released only after this SDK call.
        let removal =
            unsafe { self.environment.remove_BrowserProcessExited(self.token) }.map_err(|error| {
                io::Error::other(format!("WebView2 remove browser exit observer: {error}"))
            });
        match (result, removal) {
            (Ok(()), Ok(())) => Ok(()),
            (Err(error), Ok(())) | (Ok(()), Err(error)) => Err(error),
            (Err(wait), Err(removal)) => {
                Err(io::Error::new(wait.kind(), format!("{wait}; {removal}")))
            }
        }
    }
}

fn getter_failure(
    operation: &'static str,
    error: &windows::core::Error,
    expected: Option<u32>,
    actual: Option<u32>,
) {
    tracing::error!(
        event = "owned_browser_exit_getter_failed",
        operation,
        expected_browser_process_id = expected,
        browser_process_id = actual,
        hresult = error.code().0,
        %error,
        "Actual owned browser SDK getter failure"
    );
}

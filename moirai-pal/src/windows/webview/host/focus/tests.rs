//! Real MTA UI Automation focus and physical input, without a page handler.
//!
//! This isolates field delivery; the full Metis actor interleaves additional
//! selection/ownership observations and remains a separate acceptance gate.

mod client;
mod composition;
mod input;
mod process;

use crate::windows::{
    webview::{
        WebViewHost,
        tests::{TestPackage, startup::tracing},
    },
    window::{NativeWindow, WindowConfig, WindowVisibility},
};
use std::io::Write;
use std::time::{Duration, Instant};
use windows::Win32::UI::WindowsAndMessaging::SetForegroundWindow;

const CLIENT_ENV: &str = "MOIRAI_NATIVE_UIA_CLIENT_WINDOW";
const MODE_ENV: &str = "MOIRAI_NATIVE_UIA_OBSERVATION_MODE";
const COMPLETION_PREFIX: &str = "MOIRAI_UIA_COMPLETED ";
const TEST_NAME: &str = "windows::webview::host::focus::tests::installed_runtime_delivers_uia_focused_unicode_replacements";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ObservationMode {
    Ordinary,
    Controller,
    Startup,
    Selected,
}

impl ObservationMode {
    fn from_environment() -> Self {
        match std::env::var(MODE_ENV)
            .expect("explicit fixture observation mode")
            .as_str()
        {
            "ordinary" => Self::Ordinary,
            "controller" => Self::Controller,
            "startup" => Self::Startup,
            "selected" => Self::Selected,
            mode => panic!("unknown fixture observation mode: {mode}"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Ordinary => "ordinary",
            Self::Controller => "controller",
            Self::Startup => "startup",
            Self::Selected => "selected",
        }
    }

    fn verify_selection(self) {
        let selection = std::env::var_os(crate::webview_startup::SELECTION_ENV);
        match self {
            Self::Ordinary => assert!(selection.is_none(), "ordinary fixture has no selection"),
            Self::Selected | Self::Controller | Self::Startup => {
                let actual: serde_json::Value = serde_json::from_str(
                    selection
                        .expect("selected fixture configuration")
                        .to_str()
                        .expect("UTF-8 selection"),
                )
                .expect("actual selected fixture JSON");
                let mut expected = serde_json::json!({
                    "start_uri": "https://metis.invalid/index.html",
                    "selectors": ["#host-status", "#result", "#missing-observer-fixture"],
                });
                if self != Self::Selected {
                    expected["components"] = self.as_str().into();
                }
                assert_eq!(actual, expected);
            }
        }
    }
}

#[test]
#[ignore = "requires installed WebView2, a visible desktop and real UI Automation"]
fn installed_runtime_delivers_uia_focused_unicode_replacements() {
    let (trace, _subscriber) = tracing();
    let mode = ObservationMode::from_environment();
    mode.verify_selection();
    let _test = tracing::info_span!("native_uia_focus_delivery", mode = mode.as_str()).entered();
    if let Some(handle) = std::env::var_os(CLIENT_ENV) {
        let handle = handle
            .to_str()
            .expect("decimal HWND")
            .parse()
            .expect("native HWND");
        let completion =
            client::replace_fields(handle, mode, Instant::now() + Duration::from_secs(30));
        completion.verify(handle, mode);
        // Computed child-protocol receipt, separate from libtest's status text.
        let encoded = serde_json::to_string(&completion).expect("actual UIA completion JSON");
        writeln!(std::io::stdout(), "\n{COMPLETION_PREFIX}{encoded}")
            .expect("actual UIA completion output");
        return;
    }
    let (mut package, mut host) = tracing::info_span!("arrange").in_scope(|| {
        let mut package = TestPackage::create_with_script(br#"<!doctype html><meta charset="utf-8"><label>Reference<input id="reference" value="demo"></label><label>Weight<input id="weight" type="number" value="60"></label><label>Concentration<input id="concentration" type="number" value="1"></label><label>Dose<input id="dose" type="number" value="0.2"></label><p id="host-status">native UIA focus</p><output id="result">unsubmitted</output>"#);
        // Defer removal before any native owner exists. Unwinding during host
        // creation or foreground activation cannot prove browser shutdown.
        package.permit_cleanup(false);
        let window = NativeWindow::new(&WindowConfig::with_visibility(
            "Moirai actual UIA focus delivery", 640, 480, WindowVisibility::Visible,
        ).expect("window configuration")).expect("native window");
        let host = WebViewHost::new(window, package.fresh_config()).expect("installed WebView2");
        unsafe { assert!(SetForegroundWindow(host.window.hwnd).as_bool()); }
        (package, host)
    });
    mode.verify_installed(&host, &trace.observations());
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut child = process::NativeClient::spawn(host.window.hwnd.0 as usize, mode)
        .expect("real UIA client process");
    let result = tracing::info_span!("act").in_scope(|| child.wait(&mut host, deadline));
    let child_cleanup = child.close();
    let completion = child.completion();
    let host_cleanup = host.close();
    // Native termination errors remain failures. Closing the remote host can
    // unblock an outstanding UIA call; collect again before profile cleanup.
    let final_child_cleanup = child.close();
    package.permit_cleanup(child.is_collected() && host_cleanup.is_ok());
    let profile_cleanup = package.cleanup();
    tracing::info_span!("assert").in_scope(|| {
        for (operation, error) in [
            ("UIA client close", child_cleanup.as_ref().err()),
            ("UIA final collection", final_child_cleanup.as_ref().err()),
            ("WebView close", host_cleanup.as_ref().err()),
            ("profile cleanup", profile_cleanup.as_ref().err()),
        ] {
            if let Some(error) = error {
                tracing::error!(operation, %error, "native cleanup failed");
            }
        }
        let status = result.unwrap_or_else(|error| {
            panic!(
                "actual UIA client failed: {error}; {:?}",
                trace.observations()
            )
        });
        assert!(
            status.success(),
            "actual UIA client exits {status}; {:?}",
            trace.observations()
        );
        child_cleanup.expect("UIA client cleanup");
        final_child_cleanup.expect("UIA final client collection");
        completion
            .expect("actual computed child completion")
            .verify(host.window.hwnd.0 as usize, mode);
        host_cleanup.expect("WebView close");
        mode.verify_closed(&trace.observations());
        profile_cleanup.expect("owned profile cleanup");
    });
}

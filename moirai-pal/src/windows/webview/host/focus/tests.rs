//! Real MTA UI Automation focus and physical input, without a page handler.
//!
//! This isolates field delivery; the full Metis actor interleaves additional
//! selection/ownership observations and remains a separate acceptance gate.

mod client;
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
const COMPLETION_PREFIX: &str = "MOIRAI_UIA_COMPLETED ";
const TEST_NAME: &str = "windows::webview::host::focus::tests::installed_runtime_delivers_uia_focused_unicode_replacements";

#[test]
#[ignore = "requires installed WebView2, a visible desktop and real UI Automation"]
fn installed_runtime_delivers_uia_focused_unicode_replacements() {
    let (trace, _subscriber) = tracing();
    let _test = tracing::info_span!("native_uia_focus_delivery").entered();
    if let Some(handle) = std::env::var_os(CLIENT_ENV) {
        let handle = handle
            .to_str()
            .expect("decimal HWND")
            .parse()
            .expect("native HWND");
        let completion = client::replace_fields(handle, Instant::now() + Duration::from_secs(30));
        completion.verify(handle);
        // Computed child-protocol receipt, separate from libtest's status text.
        let encoded = serde_json::to_string(&completion).expect("actual UIA completion JSON");
        writeln!(std::io::stdout(), "\n{COMPLETION_PREFIX}{encoded}")
            .expect("actual UIA completion output");
        return;
    }
    let (package, mut host) = tracing::info_span!("arrange").in_scope(|| {
        let package = TestPackage::create_with_script(br#"<!doctype html><meta charset="utf-8"><label>Reference<input id="reference" value="demo"></label><label>Weight<input id="weight" type="number" value="60"></label><label>Concentration<input id="concentration" type="number" value="1"></label><label>Dose<input id="dose" type="number" value="0.2"></label><p id="host-status">native UIA focus</p><output id="result">unsubmitted</output>"#);
        let window = NativeWindow::new(&WindowConfig::with_visibility(
            "Moirai actual UIA focus delivery", 640, 480, WindowVisibility::Visible,
        ).expect("window configuration")).expect("native window");
        let host = WebViewHost::new(window, package.fresh_config()).expect("installed WebView2");
        unsafe { assert!(SetForegroundWindow(host.window.hwnd).as_bool()); }
        (package, host)
    });
    let deadline = Instant::now() + Duration::from_secs(30);
    let mut child =
        process::NativeClient::spawn(host.window.hwnd.0 as usize).expect("real UIA client process");
    let result = tracing::info_span!("act").in_scope(|| child.wait(&mut host, deadline));
    let child_cleanup = child.close();
    let completion = child.completion();
    let host_cleanup = host.close();
    // Native termination errors remain failures. Closing the remote host can
    // unblock an outstanding UIA call; collect again before profile cleanup.
    let final_child_cleanup = child.close();
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
            .verify(host.window.hwnd.0 as usize);
        host_cleanup.expect("WebView close");
        profile_cleanup.expect("owned profile cleanup");
    });
}

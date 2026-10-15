//! Baseline-only installed regression; no candidate-private interfaces.

use super::Startup;
use crate::webview_limits::MAX_WEBVIEW_WAIT_MILLISECONDS;
use crate::webview_startup::SELECTION_ENV;
use crate::windows::{
    webview::{WebViewHost, WebViewHostEvent, tests::TestPackage},
    window::{NativeWindow, WindowConfig, WindowEvent, WindowVisibility},
};
use serde_json::{Value, json};
use std::time::{Duration, Instant};
use windows::Win32::{
    Foundation::RECT,
    UI::WindowsAndMessaging::{
        GetClientRect, GetWindowRect, SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOZORDER, SetWindowPos,
    },
};

const START_URI: &str = "https://metis.invalid/index.html";
const SELECTORS: [&str; 3] = ["#host-status", "#result", "#missing-observer-fixture"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Counts {
    samples: u64,
    document: u64,
    dom: u64,
    activation: u64,
}

struct Fixture {
    host: WebViewHost,
    package: TestPackage,
    deadline: Instant,
}

impl Fixture {
    fn new() -> Self {
        let selected: Value = serde_json::from_str(
            &std::env::var(SELECTION_ENV).expect("CI supplies the actual child selection"),
        )
        .expect("selected startup JSON");
        assert_eq!(
            selected,
            json!({"start_uri":START_URI,"selectors":SELECTORS})
        );
        let deadline =
            Instant::now() + Duration::from_millis(u64::from(MAX_WEBVIEW_WAIT_MILLISECONDS));
        let mut package = TestPackage::create_with_script(
            "<!doctype html><meta charset=\"utf-8\"><p id=\"host-status\">Native λ🦀</p><output id=\"result\">32896</output>".as_bytes(),
        );
        package.permit_cleanup(false);
        let window = NativeWindow::new(
            &WindowConfig::with_visibility(
                "Moirai baseline actual admission priority",
                320,
                240,
                WindowVisibility::Visible,
            )
            .expect("native window configuration"),
        )
        .expect("real native window");
        let host =
            WebViewHost::new(window, package.fresh_config()).expect("installed WebView2 runtime");
        Self {
            host,
            package,
            deadline,
        }
    }

    fn startup(&self) -> &Startup {
        self.host
            .startup
            .as_ref()
            .expect("actual selected startup owner")
    }

    fn counts(&self) -> Counts {
        let state = self.startup().state.borrow();
        assert!(!state.failed && !state.budget.snapshot().exhausted);
        Counts {
            samples: state.samples,
            document: state.document_calls_attempted,
            dom: state.dom_calls_attempted,
            activation: state.activation_calls_attempted,
        }
    }

    fn settle_first(&mut self) {
        loop {
            assert!(Instant::now() < self.deadline, "original fixture deadline");
            let counts = self.counts();
            assert!(counts.samples <= 1);
            if counts.samples == 1 {
                break;
            }
            self.host
                .poll_events()
                .expect("actual existing baseline native SDK pump");
            std::thread::yield_now();
        }
        let state = self.startup().state.borrow();
        assert!(!state.slot.snapshot().pending);
        assert_eq!(
            state.last,
            [
                Some(json!("<p id=\"host-status\">Native λ🦀</p>")),
                Some(json!("<output id=\"result\">32896</output>")),
                Some(Value::Null),
            ]
        );
    }

    fn resize(&self) -> (u32, u32) {
        let mut outer = RECT::default();
        let mut old = RECT::default();
        let mut actual = RECT::default();
        // SAFETY: this thread owns the live fixture HWND and writable RECTs.
        // The real outer dimensions change while position/activation are kept.
        unsafe {
            GetWindowRect(self.host.window.hwnd, &mut outer).expect("actual outer bounds");
            GetClientRect(self.host.window.hwnd, &mut old).expect("prior actual client bounds");
            SetWindowPos(
                self.host.window.hwnd,
                None,
                0,
                0,
                outer.right - outer.left + 1,
                outer.bottom - outer.top + 2,
                SWP_NOMOVE | SWP_NOZORDER | SWP_NOACTIVATE,
            )
            .expect("real synchronous resize");
            GetClientRect(self.host.window.hwnd, &mut actual)
                .expect("resized actual client bounds");
        }
        assert_ne!((actual.right, actual.bottom), (old.right, old.bottom));
        (
            u32::try_from(actual.right - actual.left).expect("positive actual width"),
            u32::try_from(actual.bottom - actual.top).expect("positive actual height"),
        )
    }

    fn finish(&mut self) {
        self.host
            .close()
            .expect("close actual controller and owned browser");
        self.package.permit_cleanup(true);
        self.package
            .cleanup()
            .expect("owned profile files are released");
        assert!(!self.package.directory.exists() && !self.package.profile.exists());
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let result = self.host.close();
        self.package.permit_cleanup(result.is_ok());
        if let Err(error) = result {
            tracing::error!(%error, "baseline fixture retains files after failed native close");
        }
    }
}

mod admission {
    use super::*;

    #[test]
    #[ignore = "requires an installed WebView2 runtime and child-selected native observations"]
    fn installed_runtime_prioritizes_native_events_and_expired_admission() {
        let (_trace, _subscriber) = crate::windows::webview::tests::startup::tracing();
        let _test = tracing::info_span!("baseline_installed_startup_admission_priority").entered();
        let mut fixture = tracing::info_span!("arrange").in_scope(Fixture::new);
        fixture.settle_first();
        let (before, after, size, events) = tracing::info_span!("act").in_scope(|| {
            let before = fixture.counts();
            assert_eq!(before.samples, 1);
            assert_eq!(before.document, 1);
            assert_eq!(
                before.dom,
                1 + u64::try_from(SELECTORS.len()).expect("selector count") + 2
            );
            let size = fixture.resize();
            let events = fixture
                .host
                .poll_events()
                .expect("actual ready resize batch");
            (before, fixture.counts(), size, events)
        });
        // Retire the actual pending SDK owner and browser before the expected
        // baseline assertion fails, so the counter difference is not cleanup.
        fixture.finish();
        tracing::info_span!("assert").in_scope(|| {
            assert!(
                events.iter().any(|event| matches!(event,
                WebViewHostEvent::Window(WindowEvent::Resized { width, height })
                if (*width, *height) == size)),
                "actual resized event: {events:?}"
            );
            assert_eq!(
                after, before,
                "ready application event precedes new SDK work"
            );
        });
    }
}

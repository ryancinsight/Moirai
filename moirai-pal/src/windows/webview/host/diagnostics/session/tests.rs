//! Installed-runtime admission tests use actual callbacks and native events.

use super::super::protocol::Stage;
use super::Startup;
use crate::webview_limits::MAX_WEBVIEW_WAIT_MILLISECONDS;
use crate::webview_startup::{RequestId, SELECTION_ENV};
use crate::windows::{
    webview::{WebViewHost, WebViewHostEvent, tests::TestPackage},
    window::{NativeWindow, WindowConfig, WindowEvent, WindowVisibility},
};
use serde_json::{Value, json};
use std::time::{Duration, Instant};
use windows::Win32::{
    Foundation::RECT,
    UI::WindowsAndMessaging::{
        GetClientRect, GetWindowRect, MSG, SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOZORDER, SetWindowPos,
    },
};

mod admission;

const START_URI: &str = "https://metis.invalid/index.html";
const SELECTORS: [&str; 3] = ["#host-status", "#result", "#missing-observer-fixture"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Counts {
    samples: u64,
    document: u64,
    dom: u64,
    activation: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Owner {
    Reply {
        id: RequestId,
        stage: Stage,
        value: Value,
    },
    Prepared {
        id: RequestId,
        stage: Stage,
        method: &'static str,
        parameters: Value,
    },
}

impl Owner {
    const fn id(&self) -> RequestId {
        match self {
            Self::Reply { id, .. } | Self::Prepared { id, .. } => *id,
        }
    }
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
                "Moirai actual admission priority",
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

    fn owner(&self) -> Option<Owner> {
        let state = self.startup().state.borrow();
        if let Some(command) = &state.prepared {
            Some(Owner::Prepared {
                id: command.id,
                stage: command.stage,
                method: command.method,
                parameters: command.parameters.clone(),
            })
        } else {
            state.reply.as_ref().map(|(id, stage, value)| Owner::Reply {
                id: *id,
                stage: *stage,
                value: value
                    .as_ref()
                    .expect("actual successful SDK response")
                    .clone(),
            })
        }
    }

    fn before_deadline(&self) {
        assert!(
            Instant::now() < self.deadline,
            "actual admission fixture exceeded its original deadline"
        );
    }

    fn resume(&mut self, admit_new_sample: bool) {
        let before = self.counts();
        loop {
            self.before_deadline();
            self.host
                .poll_with_sampling(admit_new_sample, Some(self.deadline))
                .expect("actual bounded native pump");
            let now = self.counts();
            if now.dom != before.dom {
                assert_eq!(now.dom, before.dom + 1, "one actual SDK entry in one turn");
                return;
            }
            std::thread::yield_now();
        }
    }

    fn settle(&mut self, samples: u64) {
        loop {
            self.before_deadline();
            let counts = self.counts();
            assert!(counts.samples <= samples);
            if counts.samples == samples {
                break;
            }
            self.host
                .poll_with_sampling(false, Some(self.deadline))
                .expect("settle the actual in-flight chain");
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

    fn await_document(&self) -> (RequestId, i64) {
        loop {
            self.before_deadline();
            if let Some(Owner::Reply { id, stage, value }) = self.owner() {
                assert_eq!(stage, Stage::Document);
                assert_eq!(value["root"]["documentURL"], START_URI);
                let root = value["root"]["nodeId"]
                    .as_i64()
                    .expect("actual root NodeId");
                assert!(root > 0);
                return (id, root);
            }
            let mut message = MSG::default();
            crate::windows::webview::pump::dispatch_pending(&mut message);
            std::thread::yield_now();
        }
    }

    fn resize(&self) -> (u32, u32) {
        let mut outer = RECT::default();
        let mut old = RECT::default();
        let mut actual = RECT::default();
        // SAFETY: the fixture owns this live HWND on the calling thread. Each
        // rectangle is writable; the changed size comes from its actual bounds.
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

    fn poll_expired(&mut self, admit_new_sample: bool) {
        let expired = Instant::now()
            .checked_sub(Duration::from_nanos(1))
            .expect("representable strictly earlier instant");
        self.host
            .poll_with_sampling(admit_new_sample, Some(expired))
            .expect("expired admission still drains actual events");
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
            tracing::error!(%error, "admission fixture retains files after failed native close");
        }
    }
}

fn assert_resize(events: &[WebViewHostEvent], expected: (u32, u32)) {
    assert!(
        events.iter().any(|event| matches!(event,
        WebViewHostEvent::Window(WindowEvent::Resized { width, height })
        if (*width, *height) == expected)),
        "actual resized event: {events:?}"
    );
}

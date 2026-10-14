//! Real selected DOM observations and cancellation on an installed runtime.

use super::{
    NativeWindow, TestPackage, WebViewEvent, WebViewHost, WebViewHostEvent, WindowConfig,
    WindowVisibility,
};
use crate::{
    webview_limits::{
        MAX_WEBVIEW_EVENTS, MAX_WEBVIEW_MESSAGE_BYTES, MAX_WEBVIEW_WAIT_MILLISECONDS,
    },
    webview_startup::{SELECTION_ENV, TRACE_TARGET},
    windows::window::WindowEvent,
};
use serde_json::{Value, json};
use std::{
    io::{self, Write},
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};
use tracing_subscriber::{Layer, filter::filter_fn, fmt::format::FmtSpan, layer::SubscriberExt};

const START_URI: &str = "https://metis.invalid/index.html";
const SELECTORS: [&str; 3] = ["#host-status", "#result", "#missing-observer-fixture"];
const STATUS: &str = "Native λ🦀";

#[test]
#[ignore = "requires an installed WebView2 runtime and child-selected native observations"]
fn installed_runtime_observes_selected_dom_and_owned_shutdown() {
    let (trace, _subscriber) = tracing();
    let _test = tracing::info_span!("webview_native_startup_observation").entered();
    let (package, config, window, expected) = tracing::info_span!("arrange").in_scope(|| {
        let total = (1..=256_u32).sum::<u32>();
        assert_eq!(total, 256 * 257 / 2);
        let result = total.to_string();
        let page = format!(
            "<!doctype html><meta charset=\"utf-8\"><body><p id=\"host-status\">{STATUS}</p><output id=\"result\">{result}</output></body>"
        );
        let package = TestPackage::create_with_script(page.as_bytes());
        let config = package.fresh_config();
        assert_eq!(config.start_uri(), START_URI);
        let selected: Value = serde_json::from_str(
            &std::env::var(SELECTION_ENV).expect("CI supplies an explicit child-only selection"),
        )
        .expect("actual child selection JSON");
        assert_eq!(selected, json!({"start_uri":START_URI,"selectors":SELECTORS}));
        let window = NativeWindow::new(
            &WindowConfig::with_visibility(
                "Moirai selected native startup observation",
                320,
                240,
                WindowVisibility::Visible,
            )
            .expect("native fixture window configuration"),
        )
        .expect("actual native window");
        let expected = [
            Value::String(format!("<p id=\"host-status\">{STATUS}</p>")),
            Value::String(format!("<output id=\"result\">{result}</output>")),
            Value::Null,
        ];
        (package, config, window, expected)
    });
    let (mut host, events) = tracing::info_span!("act").in_scope(|| {
        let deadline =
            Instant::now() + Duration::from_millis(u64::from(MAX_WEBVIEW_WAIT_MILLISECONDS));
        let mut host = WebViewHost::new(window, config).expect("installed WebView2 runtime");
        let mut events = Vec::new();
        loop {
            assert!(
                Instant::now() < deadline,
                "native DOM observations did not settle: {:?}",
                trace.observations()
            );
            events.extend(host.poll_events().expect("actual owner-thread SDK pump"));
            let observed = trace.observations();
            reject_errors(&observed);
            if dom_complete(&observed, &expected) {
                break;
            }
            std::thread::yield_now();
        }
        (host, events)
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_document(&trace.observations(), &events, &expected);
        // The completed DOM chain has released its request slot. This real
        // poll starts another chain; close precedes the next owner-thread pump.
        host.poll_events().expect("begin next actual DOM request");
        host.close()
            .expect("remove receivers and close owned browser profile");
        assert!(host.is_closed());
        let final_events = host.poll_events().expect("drain actual destruction event");
        assert_eq!(
            final_events
                .iter()
                .filter(|event| matches!(event, WebViewHostEvent::Window(WindowEvent::Destroyed)))
                .count(),
            1
        );
        assert_closed(&trace.observations());
        let after_close = trace.bytes();
        assert!(
            host.poll_events()
                .expect("closed host event drain")
                .is_empty()
        );
        drop(host);
        assert_eq!(
            trace.bytes(),
            after_close,
            "late callbacks cannot revive observations"
        );
        package
            .cleanup()
            .expect("actual profile resources released");
        assert!(!package.directory.exists() && !package.profile.exists());
    });
}

fn dom_complete(observed: &[Value], expected: &[Value; 3]) -> bool {
    SELECTORS.iter().zip(expected).all(|(selector, expected)| {
        observed.iter().any(|value| {
            value["kind"] == "dom_status"
                && value["selector"] == *selector
                && value["value"] == *expected
        })
    })
}

fn assert_document(observed: &[Value], events: &[WebViewHostEvent], expected: &[Value; 3]) {
    reject_errors(observed);
    assert!(events.iter().any(|event| matches!(event,
        WebViewHostEvent::WebView(WebViewEvent::NavigationStarting {uri, allowed:true}) if uri == START_URI)));
    assert!(events.iter().any(|event| matches!(event,
        WebViewHostEvent::WebView(WebViewEvent::NavigationCompleted {uri, success:true}) if uri == START_URI)));
    let registered = one(observed, "registered");
    let enabled: Vec<_> = observed
        .iter()
        .filter(|value| value["kind"] == "enabled")
        .collect();
    assert_eq!(
        enabled
            .iter()
            .map(|value| value["method"].as_str().expect("method"))
            .collect::<Vec<_>>(),
        ["Runtime.enable", "Log.enable"]
    );
    let starting = one(observed, "navigation_starting");
    let completed = one(observed, "navigation_completed");
    assert_eq!(completed["success"], true);
    let id = decimal(&starting["navigation_id"]);
    assert!(id > 0);
    assert_eq!(decimal(&completed["navigation_id"]), id);
    assert_eq!(
        decimal(&one(observed, "content_loading")["navigation_id"]),
        id
    );
    assert_eq!(one(observed, "content_loading")["error_page"], false);
    assert_eq!(
        decimal(&one(observed, "dom_content_loaded")["navigation_id"]),
        id
    );
    let index = |value: &Value| {
        observed
            .iter()
            .position(|candidate| candidate == value)
            .expect("actual observation position")
    };
    assert!(
        index(registered) < index(enabled[0])
            && index(enabled[0]) < index(enabled[1])
            && index(enabled[1]) < index(starting)
    );
    let resources: Vec<_> = observed
        .iter()
        .filter(|value| value["kind"] == "resource_response" && value["uri"] == START_URI)
        .collect();
    assert_eq!(resources.len(), 1);
    assert_eq!(resources[0]["status"], 200);
    assert!(
        resources[0]["content_type"]
            .as_str()
            .expect("actual Content-Type")
            .split(';')
            .next()
            .expect("MIME type")
            .trim()
            .eq_ignore_ascii_case("text/html")
    );
    let dom: Vec<_> = observed
        .iter()
        .filter(|value| value["kind"] == "dom_status")
        .collect();
    assert_eq!(dom.len(), expected.len());
    for ((value, selector), expected) in dom.iter().zip(SELECTORS).zip(expected) {
        assert_eq!(value["selector"], selector);
        assert_eq!(value["value"], *expected);
        assert!(decimal(&value["generation"]) > 0);
        assert!(index(starting) < index(value));
    }
}

fn assert_closed(observed: &[Value]) {
    reject_errors(observed);
    let requested = one(observed, "close_requested");
    assert_eq!(
        requested["pending"], true,
        "actual next request is in flight at close"
    );
    let closed = one(observed, "closed");
    assert_eq!(closed["closed"], true);
    assert_eq!(closed["failed"], false);
    assert_eq!(closed["cleanup_failed"], false);
    assert_eq!(closed["exhausted"], false);
    assert_eq!(closed["omitted"], "0");
    assert!(
        closed["pending"].is_boolean(),
        "retain actual final pending state"
    );
    assert_eq!(
        decimal(&closed["generation"]),
        decimal(&requested["generation"])
    );
    assert!(decimal(&closed["samples"]) >= 1);
    decimal(&closed["stale"]);
    let ordinary: Vec<_> = observed
        .iter()
        .filter(|value| value["kind"] != "closed")
        .collect();
    assert_eq!(closed["records"], ordinary.len());
    assert_eq!(
        closed["bytes"],
        ordinary
            .iter()
            .map(|value| value.to_string().len())
            .sum::<usize>()
    );
    assert!(ordinary.len() <= MAX_WEBVIEW_EVENTS);
    assert_eq!(observed.last(), Some(closed));
}

fn reject_errors(observed: &[Value]) {
    for value in observed {
        assert!(
            !matches!(
                value["kind"].as_str(),
                Some(
                    "native_error"
                        | "state_error"
                        | "activation_incomplete"
                        | "process_failed"
                        | "quota_exhausted"
                )
            ),
            "actual startup diagnostic failure: {value}"
        );
        if value["kind"] == "cdp_event" {
            assert_ne!(
                value["name"], "Runtime.exceptionThrown",
                "actual browser exception: {value}"
            );
        }
    }
}

fn one<'record>(observed: &'record [Value], kind: &str) -> &'record Value {
    let selected: Vec<_> = observed
        .iter()
        .filter(|value| value["kind"] == kind)
        .collect();
    assert_eq!(selected.len(), 1, "one actual {kind}: {observed:?}");
    selected[0]
}

fn decimal(value: &Value) -> u64 {
    let text = value.as_str().expect("canonical decimal string");
    let number = text.parse::<u64>().expect("exact unsigned decimal value");
    assert_eq!(text, number.to_string());
    number
}

#[derive(Clone)]
struct Trace(Arc<Mutex<Buffer>>);

struct Buffer {
    bytes: Vec<u8>,
    limit: usize,
}

impl Trace {
    fn bytes(&self) -> Vec<u8> {
        self.0
            .lock()
            .expect("actual diagnostic writer")
            .bytes
            .clone()
    }

    fn observations(&self) -> Vec<Value> {
        self.bytes()
            .split_inclusive(|byte| *byte == b'\n')
            .map(|line| {
                assert_eq!(
                    line.last(),
                    Some(&b'\n'),
                    "complete native diagnostic record"
                );
                let value: Value = serde_json::from_slice(line).expect("actual formatter JSON");
                assert_eq!(value["target"], TRACE_TARGET);
                assert_eq!(value["level"], "DEBUG");
                assert_eq!(value["fields"]["event"], "webview_startup_observation");
                let encoded = value["fields"]["observation"]
                    .as_str()
                    .expect("opaque encoded observation");
                assert_eq!(
                    line.len(),
                    serde_json::to_vec(&value)
                        .expect("compact physical envelope")
                        .len()
                        + 1
                );
                assert!(
                    line.len() <= 133 + 2 * encoded.len(),
                    "actual second JSON-string escaping bound"
                );
                serde_json::from_str(encoded).expect("actual native observation JSON")
            })
            .collect()
    }
}

struct Capture(Trace);

impl Write for Capture {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let mut output = self
            .0
            .0
            .lock()
            .map_err(|error| io::Error::other(error.to_string()))?;
        if bytes.len() > output.limit - output.bytes.len() {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "native diagnostic formatter exceeds the derived trace bound",
            ));
        }
        output.bytes.extend_from_slice(bytes);
        drop(output);
        io::stderr().write_all(bytes)?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        drop(
            self.0
                .0
                .lock()
                .map_err(|error| io::Error::other(error.to_string()))?,
        );
        io::stderr().flush()
    }
}

fn tracing() -> (Trace, tracing::subscriber::DefaultGuard) {
    let envelope = json!({"level":"DEBUG", "fields":{"event":"webview_startup_observation","observation":""},"target":TRACE_TARGET}).to_string().len() + 1;
    assert_eq!(envelope, 133);
    let limit = 6 * MAX_WEBVIEW_MESSAGE_BYTES + envelope * (MAX_WEBVIEW_EVENTS + 2);
    let trace = Trace(Arc::new(Mutex::new(Buffer {
        bytes: Vec::with_capacity(limit),
        limit,
    })));
    let writer = trace.clone();
    let ordinary = tracing_subscriber::fmt::layer()
        .with_writer(std::io::stderr)
        .with_span_events(FmtSpan::FULL)
        .with_filter(filter_fn(|metadata| {
            *metadata.level() <= tracing::Level::INFO
        }));
    let diagnostic = tracing_subscriber::fmt::layer()
        .json()
        .without_time()
        .with_ansi(false)
        .with_current_span(false)
        .with_span_list(false)
        .with_span_events(FmtSpan::NONE)
        .with_writer(move || Capture(writer.clone()))
        .with_filter(filter_fn(|metadata| {
            metadata.target() == TRACE_TARGET && *metadata.level() <= tracing::Level::DEBUG
        }));
    let subscriber = tracing_subscriber::registry()
        .with(ordinary)
        .with(diagnostic);
    (trace, tracing::subscriber::set_default(subscriber))
}

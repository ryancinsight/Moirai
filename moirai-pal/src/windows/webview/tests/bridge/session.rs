//! Native fixture ownership and exact computed endpoint observations.

use super::*;
use crate::webview_limits::MAX_WEBVIEW_EVENTS;
use std::{collections::VecDeque, path::PathBuf};

pub(super) const PAYLOAD_ENV: &str = "MOIRAI_WEBVIEW_BRIDGE_FIXTURE";
const URI: &str = "https://metis.invalid/index.html";

pub(super) struct Session {
    pub(super) host: WebViewHost,
    trace: startup::Trace,
    deadline: Instant,
    incoming: VecDeque<Notice>,
}

impl Session {
    pub(super) fn new(host: WebViewHost, trace: startup::Trace, deadline: Instant) -> Self {
        Self {
            host,
            trace,
            deadline,
            incoming: VecDeque::new(),
        }
    }

    pub(super) fn send(&mut self, command: &Command) {
        let json = serde_json::to_string(command).expect("actual fixture command");
        self.host
            .post_json(&json)
            .expect("actual Rust-to-browser post");
    }

    pub(super) fn next(&mut self) -> Notice {
        loop {
            self.assert_observations();
            if let Some(value) = self.incoming.pop_front() {
                return value;
            }
            assert!(
                Instant::now() < self.deadline,
                "actual Rust bridge deadline; {:?}",
                self.trace.observations()
            );
            for event in self.host.poll_events().expect("actual native message pump") {
                if let WebViewHostEvent::WebView(WebViewEvent::Message { json, source }) = event {
                    assert_eq!(source, URI);
                    assert!(
                        self.incoming.len() < MAX_WEBVIEW_EVENTS,
                        "bounded actual fixture replies"
                    );
                    self.incoming
                        .push_back(serde_json::from_str(&json).expect("actual typed Rust reply"));
                }
            }
            std::thread::yield_now();
        }
    }

    pub(super) fn ready(&mut self, expected_mount: u8, expected_old: u32) {
        match self.next() {
            Notice::Ready { mount, old_calls } => {
                assert_eq!((mount, old_calls), (expected_mount, expected_old))
            }
            actual => panic!("actual Rust listener readiness: {actual:?}"),
        }
    }

    pub(super) fn close_listener(&mut self, mount: u8, calls: u32, old: u32) {
        self.send(&Command::Close { mount });
        match self.next() {
            Notice::Closed {
                mount: actual,
                calls: actual_calls,
                old_calls,
                retained,
            } => {
                assert_eq!(
                    (actual, actual_calls, old_calls, retained),
                    (mount, calls, old, 0)
                );
            }
            actual => panic!("actual explicit Rust listener close: {actual:?}"),
        }
    }

    fn assert_observations(&self) {
        for record in self.trace.observations() {
            assert!(
                !matches!(
                    record["kind"].as_str(),
                    Some("native_error" | "state_error" | "process_failed" | "quota_exhausted")
                ),
                "{record}"
            );
            if record["kind"] == "cdp_event" {
                assert_ne!(
                    record["name"], "Runtime.exceptionThrown",
                    "actual closed-thunk/runtime error: {record}"
                );
                if record["name"] == "Log.entryAdded" {
                    assert_ne!(record["parameters"]["entry"]["level"], "error", "{record}");
                }
            }
        }
    }

    pub(super) fn finish(mut self) {
        self.assert_observations();
        assert!(self.incoming.is_empty());
        self.host.close().expect("actual browser/profile shutdown");
        let events = self
            .host
            .poll_events()
            .expect("actual final destruction drain");
        assert_eq!(
            events
                .iter()
                .filter(|e| matches!(e, WebViewHostEvent::Window(WindowEvent::Destroyed)))
                .count(),
            1
        );
        self.assert_observations();
        let observations = self.trace.observations();
        let closed = observations.last().expect("actual observer closure");
        assert_eq!(closed["kind"], "closed");
        assert_eq!(closed["closed"], true);
        assert_eq!(closed["cleanup_failed"], false);
        assert_eq!(closed["failed"], false);
        assert_eq!(closed["exhausted"], false);
        assert_eq!(closed["omitted"], "0");
        drop(self.host);
    }
}

pub(super) fn package() -> TestPackage {
    let directory = PathBuf::from(
        std::env::var_os(PAYLOAD_ENV).expect("actual producer output selected by CI"),
    )
    .canonicalize()
    .expect("existing original compiled fixture payload");
    let mut names: Vec<_> = std::fs::read_dir(&directory)
        .expect("actual original payload directory")
        .map(|entry| entry.expect("actual payload entry").file_name())
        .collect();
    names.sort();
    let expected = [
        "bootstrap.js",
        "favicon.svg",
        "index.html",
        "webview_bridge_fixture.js",
        "webview_bridge_fixture_bg.wasm",
    ];
    assert_eq!(names, expected.map(std::ffi::OsString::from));
    let page = std::fs::read(directory.join("index.html")).expect("actual producer HTML");
    let package = TestPackage::create_with_script(&page);
    for name in expected {
        let original = directory.join(name);
        assert!(
            original
                .symlink_metadata()
                .expect("actual original metadata")
                .file_type()
                .is_file()
        );
        let copied = package.directory.join(name);
        if name != "index.html" {
            std::fs::copy(&original, &copied).expect("copy actual generated payload");
        }
        assert_eq!(
            std::fs::read(&copied).expect("actual extracted payload"),
            std::fs::read(&original).expect("actual original payload")
        );
    }
    package
}

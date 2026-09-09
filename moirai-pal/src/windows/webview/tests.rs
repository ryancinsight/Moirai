//! Value tests for the WebView2 policy boundary.

use std::{
    io,
    path::{Path, PathBuf},
    time::Duration,
};

use super::super::window::{NativeWindow, WindowConfig, WindowVisibility};
use super::{
    WebViewHost,
    config::{MAX_WEBVIEW_MESSAGE_BYTES, WebViewConfig, validate_message},
    event::WebViewEvent,
    event::WebViewHostEvent,
    state::WebViewState,
};

#[test]
fn file_policy_allows_only_the_packaged_directory() {
    let config = WebViewConfig::new("file:///C:/metis/index.html").expect("valid entry");
    assert!(config.allows("file:///C:/metis/index.html"));
    assert!(config.allows("file:///C:/metis/assets/app.js"));
    assert!(!config.allows("https://example.test/app.js"));
    assert!(!config.allows("file:///C:/metis/../secret.txt"));
    assert!(!config.allows("file:///C:/metis/%2e%2e/secret.txt"));
    assert!(!config.allows("file:///C:/metis-other/app.js"));
}

#[test]
fn configuration_and_messages_are_bounded() {
    assert_eq!(
        WebViewConfig::with_wait("file:///C:/metis/index.html", Duration::from_secs(31))
            .expect_err("wait bound")
            .kind(),
        io::ErrorKind::InvalidInput
    );
    assert!(validate_message(&vec![b'x'; MAX_WEBVIEW_MESSAGE_BYTES]).is_ok());
    assert!(validate_message(&vec![b'x'; MAX_WEBVIEW_MESSAGE_BYTES + 1]).is_err());
    assert!(validate_message(b"{\0}").is_err());
}

#[test]
fn file_configuration_rejects_non_file_and_malformed_uris() {
    for uri in [
        "",
        "https://example.test/index.html",
        "file://localhost/C:/metis/index.html",
        "file:///C:/metis/./index.html",
        "file:///C:/metis/%zz/index.html",
    ] {
        assert!(
            WebViewConfig::new(uri).is_err(),
            "URI should be rejected: {uri}"
        );
    }
}

#[test]
fn event_queue_reports_overflow_without_growing() {
    let mut state = WebViewState::new().expect("queue allocation");
    for index in 0..=super::config::MAX_WEBVIEW_EVENTS {
        state.push(WebViewEvent::NavigationStarting {
            uri: format!("file:///C:/metis/{index}.html"),
            allowed: true,
        });
    }
    let (events, overflowed) = state.drain();
    assert_eq!(events.len(), super::config::MAX_WEBVIEW_EVENTS);
    assert!(overflowed);
    let (events, overflowed) = state.drain();
    assert!(events.is_empty());
    assert!(!overflowed);
}

#[test]
#[ignore = "requires an installed WebView2 runtime"]
fn installed_runtime_loads_packaged_page_and_bridge() {
    let package = TestPackage::create();
    let config = WebViewConfig::new(package.uri()).expect("packaged URI");
    let window_config =
        WindowConfig::with_visibility("Moirai WebView2 test", 320, 240, WindowVisibility::Hidden)
            .expect("window configuration");
    let window = NativeWindow::new(&window_config).expect("native window");
    let mut host = WebViewHost::new(window, config).expect("installed WebView2 runtime");
    let mut events = host.poll_events().expect("initial WebView2 events");
    if !events.iter().any(|event| {
        matches!(
            event,
            WebViewHostEvent::WebView(WebViewEvent::Message { json, .. })
                if json.contains("\"ready\"")
        )
    }) {
        events.extend(
            host.wait_events(Duration::from_secs(1))
                .expect("bridge event"),
        );
    }
    assert!(events.iter().any(|event| {
        matches!(
            event,
            WebViewHostEvent::WebView(WebViewEvent::NavigationCompleted { success: true, .. })
        )
    }));
    assert!(events.iter().any(|event| {
        matches!(
            event,
            WebViewHostEvent::WebView(WebViewEvent::Message { json, .. })
                if json.contains("\"ready\"")
        )
    }));
    assert_eq!(
        host.navigate("https://example.test/blocked")
            .expect_err("external navigation must be denied")
            .kind(),
        io::ErrorKind::PermissionDenied
    );
    let denied = host.poll_events().expect("denied navigation event");
    assert!(denied.iter().any(|event| {
        matches!(
            event,
            WebViewHostEvent::WebView(WebViewEvent::NavigationStarting { allowed: false, .. })
        )
    }));
    host.close().expect("close WebView2 host");
    assert!(host.is_closed());
}

struct TestPackage {
    directory: PathBuf,
    entry: PathBuf,
}

impl TestPackage {
    fn create() -> Self {
        let directory =
            std::env::temp_dir().join(format!("moirai-webview2-{}", std::process::id()));
        std::fs::create_dir(&directory).expect("create unique package directory");
        let entry = directory.join("index.html");
        std::fs::write(
            &entry,
            br#"<!doctype html><meta charset="utf-8"><script>
window.chrome.webview.postMessage({"ready":true});
</script>"#,
        )
        .expect("write package entry");
        Self { directory, entry }
    }

    fn uri(&self) -> String {
        let entry = self
            .entry
            .canonicalize()
            .expect("temporary package entry is canonical");
        file_uri(&entry)
    }
}

impl Drop for TestPackage {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.directory);
    }
}

fn file_uri(path: &Path) -> String {
    let path = path
        .to_str()
        .expect("temporary package path is valid UTF-8")
        .replace('\\', "/");
    let path = path.strip_prefix("//?/").unwrap_or(&path);
    let mut uri = String::from("file:///");
    for byte in path.bytes() {
        if matches!(byte, b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'.' | b'_' | b'~' | b'/' | b':')
        {
            uri.push(char::from(byte));
        } else {
            const HEX: &[u8; 16] = b"0123456789ABCDEF";
            uri.push('%');
            uri.push(char::from(HEX[usize::from(byte >> 4)]));
            uri.push(char::from(HEX[usize::from(byte & 0x0f)]));
        }
    }
    uri
}

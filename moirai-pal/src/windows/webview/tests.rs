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
    event::WebViewPermission,
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
fn permission_kind_mapping_preserves_known_and_unknown_values() {
    assert_eq!(
        WebViewPermission::from_raw(1),
        WebViewPermission::Microphone
    );
    assert_eq!(
        WebViewPermission::from_raw(3),
        WebViewPermission::Geolocation
    );
    assert_eq!(
        WebViewPermission::from_raw(12),
        WebViewPermission::WindowManagement
    );
    assert_eq!(
        WebViewPermission::from_raw(99),
        WebViewPermission::Unknown(99)
    );
    assert_eq!(WebViewPermission::Geolocation.to_string(), "geolocation");
    assert_eq!(WebViewPermission::Unknown(99).to_string(), "unknown (99)");
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

#[test]
#[ignore = "requires an installed WebView2 runtime"]
fn installed_runtime_denies_geolocation_permission() {
    let package = TestPackage::create_with_script(
        br#"<!doctype html><meta charset="utf-8"><script>
window.chrome.webview.postMessage({"ready":true});
navigator.geolocation.getCurrentPosition(() => {}, () => {});
</script>"#,
    );
    let config = WebViewConfig::new(package.uri()).expect("packaged URI");
    let window_config = WindowConfig::with_visibility(
        "Moirai WebView2 permission test",
        320,
        240,
        WindowVisibility::Hidden,
    )
    .expect("window configuration");
    let window = NativeWindow::new(&window_config).expect("native window");
    let mut host = WebViewHost::new(window, config).expect("installed WebView2 runtime");
    let mut events = host.poll_events().expect("initial WebView2 events");
    if !events.iter().any(|event| {
        matches!(
            event,
            WebViewHostEvent::WebView(WebViewEvent::PermissionDenied {
                permission: WebViewPermission::Geolocation,
                ..
            })
        )
    }) {
        events.extend(
            host.wait_events(Duration::from_secs(1))
                .expect("permission event"),
        );
    }
    assert!(events.iter().any(|event| {
        matches!(
            event,
            WebViewHostEvent::WebView(WebViewEvent::PermissionDenied {
                permission: WebViewPermission::Geolocation,
                user_initiated: false,
                ..
            })
        )
    }));
    host.close().expect("close WebView2 host");
}

#[test]
#[ignore = "requires an installed WebView2 runtime"]
fn installed_runtime_captures_rendered_preview() {
    let package = TestPackage::create_with_script(
        br#"<!doctype html><meta charset="utf-8"><style>body{background:#123456}</style><body>preview</body>"#,
    );
    let config = WebViewConfig::new(package.uri()).expect("packaged URI");
    let window_config = WindowConfig::with_visibility(
        "Moirai WebView2 preview test",
        320,
        240,
        WindowVisibility::Hidden,
    )
    .expect("window configuration");
    let window = NativeWindow::new(&window_config).expect("native window");
    let host = WebViewHost::new(window, config).expect("installed WebView2 host");
    let png = host
        .capture_preview_png()
        .expect("WebView2 preview capture");
    assert!(png.len() > 32);
    assert_eq!(&png[..8], b"\x89PNG\r\n\x1a\n");
}

#[test]
fn folder_host_must_lie_under_a_reserved_domain() {
    let folder = std::env::temp_dir();
    let wait = Duration::from_secs(1);
    for host in ["app.metis.example", "a.test", "ui.localhost", "x-1.invalid"] {
        let config = WebViewConfig::folder(host, &folder, "index.html", wait)
            .unwrap_or_else(|error| panic!("{host} rejected: {error}"));
        assert_eq!(config.start_uri(), format!("https://{host}/index.html"));
    }
    for host in [
        "example",
        "app.com",
        "App.example",
        "-a.test",
        "a-.test",
        "a..test",
        "a.test.",
        "a_b.test",
        &format!("{}.test", "a".repeat(64)),
    ] {
        assert_eq!(
            WebViewConfig::folder(host, &folder, "index.html", wait)
                .expect_err("host outside the reserved domains")
                .kind(),
            io::ErrorKind::InvalidInput,
            "{host}"
        );
    }
}

#[test]
fn folder_source_requires_an_absolute_directory_and_clean_entry() {
    let wait = Duration::from_secs(1);
    let folder = std::env::temp_dir();
    assert!(WebViewConfig::folder("a.test", "relative/dir", "index.html", wait).is_err());
    let file = folder.join(format!("moirai-webview-file-{}", std::process::id()));
    std::fs::write(&file, b"not a directory").expect("write probe file");
    assert!(WebViewConfig::folder("a.test", &file, "index.html", wait).is_err());
    std::fs::remove_file(&file).expect("remove probe file");
    for entry in [
        "",
        "../index.html",
        "a//b.html",
        "./index.html",
        "i.html?x",
        "i.html#x",
        "a\\b.html",
        "%2e%2e/x.html",
    ] {
        assert!(
            WebViewConfig::folder("a.test", &folder, entry, wait).is_err(),
            "accepted entry {entry:?}"
        );
    }
    assert!(
        WebViewConfig::folder("a.test", &folder, "index.html", Duration::from_secs(31)).is_err()
    );
    let nested =
        WebViewConfig::folder("a.test", &folder, "app/index.html", wait).expect("nested entry");
    assert_eq!(nested.start_uri(), "https://a.test/app/index.html");
}

#[test]
fn folder_policy_confines_navigation_to_the_mapped_host() {
    let config = WebViewConfig::folder(
        "app.metis.example",
        std::env::temp_dir(),
        "index.html",
        Duration::from_secs(1),
    )
    .expect("valid mapping");
    assert!(config.allows("https://app.metis.example/index.html"));
    assert!(config.allows("https://app.metis.example/assets/logo.svg"));
    for uri in [
        "https://app.metis.example/",
        "https://app.metis.example/../secret",
        "https://app.metis.example.evil.test/index.html",
        "https://evil.test/index.html",
        "http://app.metis.example/index.html",
        "file:///C:/metis/index.html",
    ] {
        assert!(!config.allows(uri), "allowed {uri}");
    }
    let file = WebViewConfig::new("file:///C:/metis/index.html").expect("file entry");
    assert!(!file.allows("https://app.metis.example/index.html"));
}

#[test]
#[ignore = "requires an installed WebView2 runtime"]
fn installed_runtime_loads_module_page_from_mapped_folder() {
    let folder =
        std::env::temp_dir().join(format!("moirai-webview2-folder-{}", std::process::id()));
    std::fs::create_dir_all(folder.join("lib")).expect("create unique folder");
    // Canonicalization yields the verbatim `\\?\` form, which WebView2 does
    // not normalize when it appends a request path.
    let folder = folder.canonicalize().expect("canonical folder");
    std::fs::write(
        folder.join("index.html"),
        br#"<!doctype html><meta charset="utf-8"><script type="module" src="main.js"></script>"#,
    )
    .expect("write entry");
    std::fs::write(
        folder.join("main.js"),
        br#"import { origin } from "./lib/origin.js";
window.chrome.webview.postMessage({"module": origin});"#,
    )
    .expect("write module");
    std::fs::write(
        folder.join("lib").join("origin.js"),
        b"export const origin = location.origin;",
    )
    .expect("write imported module");
    let config = WebViewConfig::folder(
        "moirai.test",
        &folder,
        "index.html",
        Duration::from_secs(10),
    )
    .expect("mapped folder");
    let window_config = WindowConfig::with_visibility(
        "Moirai WebView2 folder test",
        320,
        240,
        WindowVisibility::Hidden,
    )
    .expect("window configuration");
    let window = NativeWindow::new(&window_config).expect("native window");
    let mut host = WebViewHost::new(window, config).expect("installed WebView2 runtime");
    let mut events = host.poll_events().expect("initial WebView2 events");
    let loaded = |events: &[WebViewHostEvent]| {
        events.iter().any(|event| {
            matches!(
                event,
                WebViewHostEvent::WebView(WebViewEvent::Message { json, .. })
                    if json.contains("\"module\":\"https://moirai.test\"")
            )
        })
    };
    if !loaded(&events) {
        events.extend(
            host.wait_events(Duration::from_secs(2))
                .expect("module event"),
        );
    }
    assert!(
        loaded(&events),
        "module import did not report the mapped origin: {events:?}"
    );
    assert_eq!(
        host.navigate("https://example.test/elsewhere")
            .expect_err("navigation outside the mapped host")
            .kind(),
        io::ErrorKind::PermissionDenied
    );
    host.close().expect("close WebView2 host");
    std::fs::remove_dir_all(&folder).expect("remove mapped folder");
}

struct TestPackage {
    directory: PathBuf,
    entry: PathBuf,
}

impl TestPackage {
    fn create() -> Self {
        Self::create_with_script(
            br#"<!doctype html><meta charset="utf-8"><script>
window.chrome.webview.postMessage({"ready":true});
</script>"#,
        )
    }

    fn create_with_script(script: &[u8]) -> Self {
        let directory =
            std::env::temp_dir().join(format!("moirai-webview2-{}", std::process::id()));
        std::fs::create_dir(&directory).expect("create unique package directory");
        let entry = directory.join("index.html");
        std::fs::write(&entry, script).expect("write package entry");
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

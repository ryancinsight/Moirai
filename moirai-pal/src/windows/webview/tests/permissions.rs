//! Real per-request native DENY evidence on an exclusively owned fresh profile.

use std::time::{Duration, Instant};

use super::{
    NativeWindow, TestPackage, WebViewEvent, WebViewHost, WebViewHostEvent, WebViewPermission,
    WindowConfig, WindowVisibility,
};
use crate::webview_limits::MAX_WEBVIEW_WAIT_MILLISECONDS;

const PERMISSION_ORIGIN: &str = "https://metis.invalid/";
const SECOND_DOCUMENT: &str = "https://metis.invalid/second.html";

#[test]
#[ignore = "requires an installed WebView2 runtime"]
fn installed_runtime_denies_each_repeated_permission_without_saved_decisions() {
    let _trace = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_test_writer()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .finish(),
    );
    let _test = tracing::info_span!("webview_native_repeated_permission_denial").entered();
    let (package, config, window) = tracing::info_span!("arrange").in_scope(|| {
        let package = TestPackage::create_with_script(request_document(1).as_bytes());
        std::fs::write(package.directory.join("second.html"), request_document(2))
            .expect("write second actual request document");
        let config = package.fresh_config();
        let window = NativeWindow::new(
            &WindowConfig::with_visibility(
                "Moirai WebView2 repeated permission test",
                320,
                240,
                WindowVisibility::Visible,
            )
            .expect("visible native window configuration"),
        )
        .expect("native window");
        (package, config, window)
    });
    let (mut host, events) = tracing::info_span!("act").in_scope(|| {
        let mut host =
            WebViewHost::new(window, config).expect("installed WebView2 host with fresh profile");
        let deadline =
            Instant::now() + Duration::from_millis(u64::from(MAX_WEBVIEW_WAIT_MILLISECONDS));
        let mut events = host.poll_events().expect("initial native events");
        let mut second_started = false;
        while denial_counts(&events) != (2, 2) {
            // Same-document geolocation calls can share a browser permission
            // request even when no decision is persisted. The actual first
            // callback outcome closes that document's request phase; a native
            // navigation starts another navigator under the SAME profile,
            // origin and controller. No timer or synthetic denial is used.
            if !second_started && denial_counts(&events) == (1, 1) {
                host.navigate(SECOND_DOCUMENT)
                    .expect("navigate to second packaged request document");
                second_started = true;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            assert!(
                !remaining.is_zero(),
                "two independent permission denials did not complete: {events:?}"
            );
            events.extend(host.wait_events(remaining).expect("real permission event"));
        }
        (host, events)
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(denial_counts(&events), (2, 2));
        let messages = events
            .iter()
            .filter_map(|event| match event {
                WebViewHostEvent::WebView(WebViewEvent::Message { source, json }) => {
                    Some((source.as_str(), json.as_str()))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(
            messages,
            [
                (
                    "https://metis.invalid/index.html",
                    r#"{"sequence":1,"outcome":"error","code":1}"#
                ),
                (
                    SECOND_DOCUMENT,
                    r#"{"sequence":2,"outcome":"error","code":1}"#
                )
            ]
        );
        host.close()
            .expect("close native controller and profile environment");
        assert!(host.is_closed());
        assert!(package.profile.is_dir(), "profile remains caller-owned");
    });
    drop(host);
    assert_eq!(
        package
            .cleanup()
            .expect("actual profile resources are released"),
        ()
    );
    assert!(!package.directory.exists() && !package.profile.exists());
}

fn denial_counts(events: &[WebViewHostEvent]) -> (usize, usize) {
    let mut native = 0;
    let mut browser = 0;
    for event in events {
        match event {
            WebViewHostEvent::WebView(WebViewEvent::PermissionDenied {
                uri,
                permission: WebViewPermission::Geolocation,
                user_initiated: false,
            }) if uri == PERMISSION_ORIGIN => native += 1,
            WebViewHostEvent::WebView(WebViewEvent::Message { json, .. })
                if json == r#"{"sequence":1,"outcome":"error","code":1}"#
                    || json == r#"{"sequence":2,"outcome":"error","code":1}"# =>
            {
                browser += 1
            }
            _ => {}
        }
    }
    (native, browser)
}

fn request_document(sequence: u8) -> String {
    format!(
        r#"<!doctype html><meta charset="utf-8"><script>
const sequence = {sequence};
navigator.geolocation.getCurrentPosition(
  () => window.chrome.webview.postMessage({{sequence, outcome: "granted"}}),
  (error) => window.chrome.webview.postMessage({{sequence, outcome: "error", code: error.code}})
);
</script>"#
    )
}

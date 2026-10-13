//! Real WM_NCDESTROY survives close and drains once from its owning window.

use super::super::{NativeWindow, WindowConfig, WindowEvent, WindowVisibility};
use std::time::Duration;

#[test]
fn actual_destruction_drains_once_after_close_without_consuming_a_sibling() {
    let _trace = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_test_writer()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .finish(),
    );
    let _test = tracing::info_span!("native_destruction_ownership").entered();
    for visibility in [WindowVisibility::Hidden, WindowVisibility::Visible] {
        let (mut first, mut sibling) = tracing::info_span!("arrange", ?visibility).in_scope(|| {
            let config = WindowConfig::with_visibility("Owned destruction", 320, 240, visibility)
                .expect("valid actual window configuration");
            let mut first = NativeWindow::new(&config).expect("first actual HWND");
            let mut sibling = NativeWindow::new(&config).expect("sibling actual HWND");
            first.poll_events().expect("initial first events");
            sibling.poll_events().expect("initial sibling events");
            (first, sibling)
        });
        let (first_events, sibling_events) =
            tracing::info_span!("act", ?visibility).in_scope(|| {
                first.close().expect("actual synchronous first destruction");
                (
                    first
                        .wait_events(Duration::ZERO)
                        .expect("drain first final events"),
                    sibling
                        .poll_events()
                        .expect("sibling remains independently live"),
                )
            });
        let _assert = tracing::info_span!("assert", ?visibility).entered();
        assert_eq!(
            first_events
                .iter()
                .filter(|event| matches!(event, WindowEvent::Destroyed))
                .count(),
            1
        );
        assert_eq!(
            sibling_events
                .iter()
                .filter(|event| matches!(event, WindowEvent::Destroyed))
                .count(),
            0
        );
        assert_eq!(
            first.poll_events().expect("repeat drain"),
            Vec::<WindowEvent>::new()
        );
        first.close().expect("repeat close is idempotent");
        assert_eq!(
            first
                .wait_events(Duration::ZERO)
                .expect("repeat closed wait"),
            Vec::<WindowEvent>::new()
        );
        sibling
            .close()
            .expect("actual synchronous sibling destruction");
        let final_events = sibling.poll_events().expect("sibling final events");
        assert_eq!(
            final_events
                .iter()
                .filter(|event| matches!(event, WindowEvent::Destroyed))
                .count(),
            1
        );
    }
}

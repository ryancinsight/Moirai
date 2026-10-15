//! The documented overlapping navigation sequence cannot admit an old document.

use super::*;

fn trace() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_test_writer()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .finish(),
    )
}

#[test]
fn overlapping_navigation_ids_preserve_the_current_selected_document_phase() {
    let _trace = trace();
    let _test = tracing::info_span!("startup_navigation_overlap").entered();
    let mut state = tracing::info_span!("arrange").in_scope(|| {
        State::new(Selection {
            components: crate::webview_startup::Components::Full,
            start_uri: "https://startup.invalid/index.html".to_owned(),
            selectors: vec!["#status".to_owned()],
        })
    });
    tracing::info_span!("act").in_scope(|| {
        state.navigation_started(1, "https://startup.invalid/index.html");
        state.navigation_started(2, "https://startup.invalid/other.html");
    });
    tracing::info_span!("assert").in_scope(|| {
        assert!(!state.navigation_completed(1, true));
        assert!(!state.dom_loaded(1));
        assert_eq!(state.navigation, Some(2));
        assert_eq!(state.document, Document::Outside);
        assert_eq!(state.slot.snapshot().generation, 3);
    });
    tracing::info_span!("act")
        .in_scope(|| state.navigation_started(3, "https://startup.invalid/index.html"));
    tracing::info_span!("assert").in_scope(|| {
        assert!(!state.navigation_completed(2, false));
        assert!(!state.dom_loaded(2));
        assert_eq!(state.document, Document::Loading);
        assert_eq!(state.navigation, Some(3));
        assert!(state.dom_loaded(3));
        assert_eq!(state.document, Document::Available);
        assert!(state.navigation_completed(3, true));
        assert_eq!(state.document, Document::Available);
        assert_eq!(state.slot.snapshot().generation, 4);
    });
    tracing::info_span!("act").in_scope(|| state.invalidate_nodes());
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(state.document, Document::Available);
        assert_eq!(state.navigation, Some(3));
        assert_eq!(state.slot.snapshot().generation, 5);
        assert_eq!(state.last, vec![None]);
    });
}

#[test]
fn command_count_overflow_rejects_atomically_without_wrapping() {
    let _trace = trace();
    let _test = tracing::info_span!("startup_command_count_boundary").entered();
    let mut state = tracing::info_span!("arrange").in_scope(|| {
        let mut state = State::new(Selection {
            components: crate::webview_startup::Components::Full,
            start_uri: "https://startup.invalid/index.html".to_owned(),
            selectors: vec!["#状態-🦀".to_owned()],
        });
        state.dom_calls_attempted = u64::MAX - 1;
        state.document_calls_attempted = u64::MAX - 1;
        state.activation_calls_attempted = 2;
        state
    });
    tracing::info_span!("act").in_scope(|| {
        state
            .command_attempt("DOM.getDocument", true)
            .expect("last representable attempt")
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(state.dom_calls_attempted, u64::MAX);
        assert_eq!(state.document_calls_attempted, u64::MAX);
        assert_eq!(state.description_calls_attempted, 0);
        assert_eq!(state.activation_calls_attempted, 2);
    });
    let error = tracing::info_span!("act").in_scope(|| {
        state
            .command_attempt("DOM.describeNode", true)
            .expect_err("no wrapped or partial attempt")
    });
    tracing::info_span!("assert").in_scope(|| {
        assert!(error.message().contains("command count exhausted"));
        assert_eq!(state.dom_calls_attempted, u64::MAX);
        assert_eq!(state.document_calls_attempted, u64::MAX);
        assert_eq!(state.description_calls_attempted, 0);
        assert_eq!(state.activation_calls_attempted, 2);
    });
}

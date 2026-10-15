//! Computed stage decisions enforce exact document ownership and quota stops.

use super::*;
use crate::webview_startup::{RecordAdmission, Selection};

mod admission;
mod deferred;

fn trace() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_test_writer()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .finish(),
    )
}

fn state() -> (State, RequestId) {
    let mut state = State::new(Selection {
        components: crate::webview_startup::Components::Full,
        start_uri: "https://startup.invalid/index.html".to_owned(),
        selectors: vec!["#status".to_owned(), "#result".to_owned()],
    });
    let Start::Begun(id) = state.slot.start().expect("empty slot") else {
        panic!("one request")
    };
    (state, id)
}

#[test]
fn document_reply_requires_exact_selected_uri_before_querying_any_node() {
    let _trace = trace();
    let _test = tracing::info_span!("startup_dom_document_identity").entered();
    for uri in [
        Some("https://startup.invalid/index.html"),
        Some("https://startup.invalid/other.html"),
        None,
    ] {
        let (mut state, id, reply) = tracing::info_span!("arrange").in_scope(|| {
            let (state, id) = state();
            let mut root = json!({"nodeId":17});
            if let Some(uri) = uri {
                root["documentURL"] = json!(uri);
            }
            (state, id, json!({"root":root}))
        });
        let command = tracing::info_span!("act")
            .in_scope(|| advance(&mut state, id, Stage::Document, &reply));
        tracing::info_span!("assert").in_scope(|| {
            if uri == Some(state.selection.start_uri.as_str()) {
                let command = command
                    .expect("owned document")
                    .expect("first actual selector command");
                assert_eq!(command.id, id);
                assert_eq!(command.method, "DOM.querySelector");
                assert_eq!(
                    command.parameters,
                    json!({"nodeId":17,"selector":"#status"})
                );
            } else {
                let error = match command {
                    Err(error) => error,
                    Ok(_) => panic!("unowned document rejects"),
                };
                assert_eq!(
                    error.message(),
                    if uri.is_some() {
                        "DOM documentURL differs from selected startup URI"
                    } else {
                        "DOM document omits its documentURL"
                    }
                );
            }
            assert!(state.slot.is_current(id));
            assert_eq!(state.budget.snapshot().bytes, 0);
            assert_eq!(state.samples, 0);
        });
    }
}

#[test]
fn exhaustion_during_a_selector_reply_finishes_before_the_next_native_command() {
    let _trace = trace();
    let _test = tracing::info_span!("startup_dom_quota_transition").entered();
    let (mut state, id) = tracing::info_span!("arrange").in_scope(|| {
        let (mut state, id) = state();
        let value = json!("x".repeat(crate::webview_limits::MAX_WEBVIEW_MESSAGE_BYTES - 2));
        let RecordAdmission::Admitted(encoded) = state.budget.record(&value).expect("full quota")
        else {
            panic!("inclusive full record")
        };
        assert_eq!(
            encoded.len(),
            crate::webview_limits::MAX_WEBVIEW_MESSAGE_BYTES
        );
        (state, id)
    });
    let command = tracing::info_span!("act").in_scope(|| {
        advance(
            &mut state,
            id,
            Stage::Html { index: 0, root: 17 },
            &json!({"outerHTML":"<p id=\"status\">actual value</p>"}),
        )
    });
    tracing::info_span!("assert").in_scope(|| {
        assert!(command.expect("completed bounded transition").is_none());
        assert_eq!(
            state.budget.snapshot(),
            crate::webview_startup::BudgetSnapshot {
                records: 1,
                bytes: crate::webview_limits::MAX_WEBVIEW_MESSAGE_BYTES,
                omitted: 1,
                exhausted: true
            }
        );
        assert_eq!(
            state.slot.snapshot(),
            crate::webview_startup::RequestSnapshot {
                generation: 1,
                pending: false,
                closed: false
            }
        );
        assert_eq!(state.last, vec![None, None]);
        assert_eq!(state.samples, 0);
    });
}

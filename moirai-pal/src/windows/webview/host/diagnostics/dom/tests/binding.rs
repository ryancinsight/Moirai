//! Live document identity, fresh values and generation revocation stay coupled.

use super::*;

fn complete_sample(state: &mut State, id: RequestId, document: Command, text: &str) {
    let key = if document.method == "DOM.getDocument" {
        "root"
    } else {
        "node"
    };
    let reply = json!({(key):{"nodeId":17,"documentURL":state.selection.start_uri}});
    let query = advance(state, id, document.stage, &reply)
        .expect("owned document")
        .expect("first query");
    assert_eq!(query.parameters, json!({"nodeId":17,"selector":"#status"}));
    let html = advance(state, id, query.stage, &json!({"nodeId":29}))
        .expect("actual first match")
        .expect("HTML read");
    assert_eq!(html.parameters, json!({"nodeId":29}));
    let second = advance(state, id, html.stage, &json!({"outerHTML":text}))
        .expect("actual HTML")
        .expect("second query");
    assert_eq!(second.parameters, json!({"nodeId":17,"selector":"#result"}));
    assert!(
        advance(state, id, second.stage, &json!({"nodeId":0}))
            .expect("actual missing selector")
            .is_none()
    );
}

#[test]
fn later_samples_describe_the_bound_document_and_still_read_fresh_values() {
    let _trace = trace();
    let _test = tracing::info_span!("startup_document_binding_reuse").entered();
    let (mut state, id) = tracing::info_span!("arrange").in_scope(state);
    tracing::info_span!("act").in_scope(|| {
        let document = document_command(&state, id).expect("fresh document command");
        assert_eq!(document.method, "DOM.getDocument");
        assert_eq!(document.parameters, json!({"depth":0,"pierce":false}));
        complete_sample(&mut state, id, document, "<p id=\"status\">患者-🦀-one</p>");
    });
    let root = state.root.expect("admitted native root identity");
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(root.node, 17);
        assert_eq!(root.generation, state.slot.snapshot().generation);
        assert_eq!(state.samples, 1);
    });
    tracing::info_span!("act").in_scope(|| {
        let Start::Begun(next) = state.slot.start().expect("next sample") else {
            panic!("settled sample admits one owner")
        };
        let document = document_command(&state, next).expect("bound document command");
        assert_eq!(document.method, "DOM.describeNode");
        assert_eq!(
            document.parameters,
            json!({"nodeId":17,"depth":0,"pierce":false})
        );
        complete_sample(
            &mut state,
            next,
            document,
            "<p id=\"status\">患者-🦀-two</p>",
        );
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(state.root, Some(root));
        assert_eq!(state.samples, 2);
        assert_eq!(
            state.last,
            [
                Some(json!("<p id=\"status\">患者-🦀-two</p>")),
                Some(Value::Null)
            ]
        );
        assert!(!state.slot.snapshot().pending);
    });
}

#[test]
fn description_requires_current_uri_and_the_exact_bound_identity() {
    let _trace = trace();
    let _test = tracing::info_span!("startup_described_document_admission").entered();
    for (node, uri, expected) in [
        (
            json!(17),
            Some(json!("https://startup.invalid/other.html")),
            "documentURL differs",
        ),
        (json!(17), None, "omits its documentURL"),
        (json!(17), Some(json!(false)), "omits its documentURL"),
        (
            json!(0),
            Some(json!("https://startup.invalid/index.html")),
            "has no root node",
        ),
        (
            json!(18),
            Some(json!("https://startup.invalid/index.html")),
            "document identity differs",
        ),
    ] {
        let (mut state, id) = tracing::info_span!("arrange").in_scope(state);
        let initial = document_command(&state, id).expect("initial document");
        complete_sample(&mut state, id, initial, "<p>owned initial value</p>");
        let root = state.root;
        let Start::Begun(next) = state.slot.start().expect("next sample") else {
            panic!("completed initial chain")
        };
        let document = document_command(&state, next).expect("retained root");
        let mut reply = json!({"node":{"nodeId":node}});
        if let Some(uri) = uri {
            reply["node"]["documentURL"] = uri;
        }
        let error = tracing::info_span!("act").in_scope(|| {
            advance(&mut state, next, document.stage, &reply)
                .err()
                .expect("unowned description rejects before querying")
        });
        tracing::info_span!("assert").in_scope(|| {
            assert!(error.message().contains(expected), "{error}");
            assert_eq!(
                state.root, root,
                "unverified identity never replaces the owner"
            );
            assert_eq!(state.samples, 1);
            assert!(state.slot.is_current(next));
        });
    }
}

#[test]
fn invalidation_navigation_and_close_revoke_a_bound_root() {
    let _trace = trace();
    let _test = tracing::info_span!("startup_document_binding_revocation").entered();
    let (mut state, id) = tracing::info_span!("arrange").in_scope(state);
    let initial = document_command(&state, id).expect("initial document");
    complete_sample(&mut state, id, initial, "<p>before native update</p>");
    let Start::Begun(pending) = state.slot.start().expect("one pending sample") else {
        panic!("initial settled")
    };
    tracing::info_span!("act").in_scope(|| state.invalidate_nodes());
    tracing::info_span!("assert").in_scope(|| {
        assert!(state.root.is_none());
        assert!(!state.slot.is_current(pending));
        assert!(
            state.slot.snapshot().pending,
            "stale callback retains its actual slot"
        );
    });
    state.slot.finish(pending);
    let Start::Begun(next) = state.slot.start().expect("settled old owner") else {
        panic!("actual settlement")
    };
    let initial = document_command(&state, next).expect("new generation root");
    assert_eq!(initial.method, "DOM.getDocument");
    complete_sample(&mut state, next, initial, "<p>after native update</p>");
    tracing::info_span!("act")
        .in_scope(|| state.navigation_started(7, "https://startup.invalid/index.html"));
    tracing::info_span!("assert").in_scope(|| assert!(state.root.is_none()));
    let Start::Begun(next) = state.slot.start().expect("new navigation") else {
        panic!("no pending old sample")
    };
    let initial = document_command(&state, next).expect("new document command");
    complete_sample(&mut state, next, initial, "<p>new actual document</p>");
    tracing::info_span!("act").in_scope(|| state.close_admission());
    tracing::info_span!("assert").in_scope(|| {
        assert!(state.root.is_none());
        assert!(matches!(
            state.slot.start().expect("closed admission"),
            Start::Closed
        ));
    });
}

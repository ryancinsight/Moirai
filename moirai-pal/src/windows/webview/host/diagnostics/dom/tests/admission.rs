//! A wait turn may settle every stage without admitting another chain.

use super::*;

#[test]
fn pending_only_polls_finish_fresh_html_without_rearming_the_sampler() {
    let _trace = trace();
    let _test = tracing::info_span!("startup_sample_admission").entered();
    let mut state = tracing::info_span!("arrange").in_scope(|| {
        let mut state = State::new(Selection {
            components: crate::webview_startup::Components::Full,
            late_input_after_key: None,
            start_uri: "https://startup.invalid/index.html".to_owned(),
            selectors: vec!["#status".to_owned(), "#missing".to_owned()],
        });
        state.navigation_started(7, "https://startup.invalid/index.html");
        assert!(state.dom_loaded(7));
        state
    });
    tracing::info_span!("assert").in_scope(|| {
        assert!(next_command(&mut state, false).is_none());
        assert!(!state.slot.snapshot().pending);
    });
    let document = tracing::info_span!("act")
        .in_scope(|| next_command(&mut state, true).expect("one fresh chain"));
    assert_eq!(document.method, "DOM.getDocument");
    let id = document.id;
    state.reply = Some((
        id,
        document.stage,
        Ok(json!({"root":{"nodeId":17,"documentURL":"https://startup.invalid/index.html"}})),
    ));
    let query = tracing::info_span!("act")
        .in_scope(|| next_command(&mut state, false).expect("pending document advances"));
    assert_eq!(query.parameters, json!({"nodeId":17,"selector":"#status"}));
    state.reply = Some((id, query.stage, Ok(json!({"nodeId":29}))));
    let html = tracing::info_span!("act")
        .in_scope(|| next_command(&mut state, false).expect("pending selector advances"));
    assert_eq!(html.method, "DOM.getOuterHTML");
    assert_eq!(html.parameters, json!({"nodeId":29}));
    state.reply = Some((
        id,
        html.stage,
        Ok(json!({"outerHTML":"<p id=\"status\">患者-🦀-one</p>"})),
    ));
    let missing = tracing::info_span!("act")
        .in_scope(|| next_command(&mut state, false).expect("fresh HTML admits next selector"));
    assert_eq!(
        missing.parameters,
        json!({"nodeId":17,"selector":"#missing"})
    );
    state.reply = Some((id, missing.stage, Ok(json!({"nodeId":0}))));
    tracing::info_span!("act").in_scope(|| assert!(next_command(&mut state, false).is_none()));
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(state.samples, 1);
        assert_eq!(
            state.last,
            [
                Some(json!("<p id=\"status\">患者-🦀-one</p>")),
                Some(Value::Null)
            ]
        );
        assert!(!state.slot.snapshot().pending);
        assert!(
            next_command(&mut state, false).is_none(),
            "internal completion cannot self-rearm"
        );
        assert!(!state.slot.snapshot().pending);
    });
    tracing::info_span!("act").in_scope(|| {
        let next = next_command(&mut state, true).expect("next caller turn admits fresh work");
        assert_eq!(next.method, "DOM.getDocument");
        assert_ne!(next.id, id);
        assert!(state.slot.is_current(next.id));
    });
}

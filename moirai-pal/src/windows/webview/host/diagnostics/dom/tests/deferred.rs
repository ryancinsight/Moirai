//! Unissued commands retain their exact identity; actual in-flight work does not replay.

use super::*;

#[test]
fn deferred_fresh_and_followup_commands_resume_without_a_fresh_sample() {
    let _trace = trace();
    let _test = tracing::info_span!("startup_deferred_command_identity").entered();
    let mut state = tracing::info_span!("arrange").in_scope(|| {
        let (mut state, id) = state();
        state.slot.finish(id);
        state.document = Document::Available;
        state.selection.selectors[0] = "#状態-🦀".into();
        state
    });
    let id = tracing::info_span!("act").in_scope(|| {
        let command = next_command(&mut state, true).expect("one fresh command");
        let id = command.id;
        defer(&mut state, command);
        id
    });
    tracing::info_span!("assert").in_scope(|| {
        assert!(state.slot.is_current(id));
        assert_eq!(state.dom_calls_attempted, 0);
        assert_eq!(state.samples, 0);
        assert!(state.reply.is_none());
    });
    let command = tracing::info_span!("act")
        .in_scope(|| next_command(&mut state, false).expect("deferred command, no new sample"));
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(command.id, id);
        assert_eq!(command.method, "DOM.getDocument");
        assert_eq!(command.parameters, json!({"depth":0,"pierce":false}));
        assert!(state.prepared.is_none());
        assert!(next_command(&mut state, false).is_none());
    });
    let command = tracing::info_span!("act").in_scope(|| {
        state.reply = Some((
            id,
            Stage::Document,
            Ok(json!({"root":{"nodeId":31,"documentURL":state.selection.start_uri}})),
        ));
        let command = next_command(&mut state, false).expect("settled reply produces selector");
        defer(&mut state, command);
        next_command(&mut state, false).expect("deferred next stage")
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(command.id, id);
        assert_eq!(command.stage, Stage::Query { index: 0, root: 31 });
        assert_eq!(command.method, "DOM.querySelector");
        assert_eq!(
            command.parameters,
            json!({"nodeId":31,"selector":"#状態-🦀"})
        );
        assert!(state.prepared.is_none());
        assert!(state.slot.is_current(id));
        assert_eq!(state.dom_calls_attempted, 0);
    });
}

#[test]
fn cancellation_retires_only_unissued_ownership_and_never_reuses_identity() {
    let _trace = trace();
    let _test = tracing::info_span!("startup_deferred_cancellation").entered();
    for cause in ["navigation", "closed", "quota", "failed"] {
        let (mut state, id) = tracing::info_span!("arrange", cause).in_scope(|| {
            let (mut state, old) = state();
            state.slot.finish(old);
            state.document = Document::Available;
            let command = next_command(&mut state, true).expect("fresh command");
            let id = command.id;
            defer(&mut state, command);
            (state, id)
        });
        tracing::info_span!("act").in_scope(|| {
            match cause {
                "navigation" => state.invalidate_nodes(),
                "closed" => state.slot.close(),
                "quota" => {
                    for index in 0..crate::webview_limits::MAX_WEBVIEW_EVENTS {
                        state.record(&json!({"index":index}));
                    }
                    state.record(&json!({"overflow":"actual quota transition"}));
                }
                "failed" => state.failed = true,
                _ => unreachable!("declared rejection causes"),
            }
            assert!(next_command(&mut state, false).is_none());
        });
        tracing::info_span!("assert").in_scope(|| {
            assert!(state.prepared.is_none());
            assert!(!state.slot.snapshot().pending);
            assert!(!state.slot.is_current(id));
            assert_eq!(state.dom_calls_attempted, 0);
            assert_eq!(state.samples, 0);
            if cause == "navigation" {
                let next = next_command(&mut state, true).expect("new generation is admissible");
                assert_ne!(next.id, id);
                assert_eq!(state.slot.snapshot().generation, 2);
            }
        });
    }
    let (mut state, id) = tracing::info_span!("arrange").in_scope(state);
    tracing::info_span!("act").in_scope(|| state.invalidate_nodes());
    tracing::info_span!("assert").in_scope(|| {
        assert!(state.prepared.is_none());
        assert!(
            state.slot.snapshot().pending,
            "actual in-flight ownership stays pending"
        );
        assert!(!state.slot.is_current(id));
        assert!(matches!(
            state.slot.start().expect("pending stale callback"),
            Start::Busy
        ));
    });
}

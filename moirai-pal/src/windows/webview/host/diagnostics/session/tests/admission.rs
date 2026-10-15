//! Ready app events and expired budgets cannot admit diagnostic SDK work.
//!
//! On aa532, collect_events drains WM_SIZE then issues/advances CDP before
//! returning that ready event. macOS cannot execute this installed Windows
//! regression; baseline RED and candidate GREEN remain real Windows evidence.

use super::*;
use std::cell::Cell;

#[test]
#[ignore = "requires an installed WebView2 runtime and child-selected native observations"]
fn installed_runtime_prioritizes_native_events_and_expired_admission() {
    let (trace, _subscriber) = crate::windows::webview::tests::startup::tracing();
    let _test = tracing::info_span!("installed_startup_admission_priority").entered();
    let mut fixture = tracing::info_span!("arrange").in_scope(Fixture::new);
    fixture.resume(true);
    fixture.settle(1);
    tracing::info_span!("act", phase = "fresh").in_scope(|| fresh_priority(&mut fixture));
    let (id, root) = fixture.await_document();
    tracing::info_span!("act", phase = "continuation")
        .in_scope(|| continuation_priority(&mut fixture, id, root));
    tracing::info_span!("assert").in_scope(|| {
        fixture.settle(2);
        let counts = fixture.counts();
        assert_eq!(counts.samples, 2);
        assert_eq!(counts.document, 2);
        // One document, three queries and two actual selected elements per chain.
        let selectors = u64::try_from(SELECTORS.len()).expect("bounded selector count");
        assert_eq!(counts.dom, 2 * (1 + selectors + 2));
        assert_eq!(counts.activation, 2);
        fixture.finish();
        let observations = trace.observations();
        let closed: Vec<_> = observations
            .iter()
            .filter(|value| value["kind"] == "closed")
            .collect();
        assert_eq!(closed.len(), 1);
        assert_eq!(closed[0]["failed"], false);
        assert_eq!(closed[0]["cleanup_failed"], false);
        assert_eq!(closed[0]["pending"], false);
        assert_eq!(closed[0]["omitted"], "0");
    });
}

fn fresh_priority(fixture: &mut Fixture) {
    let before = fixture.counts();
    let size = fixture.resize();
    let events = fixture
        .host
        .poll_events()
        .expect("deliver ready native resize");
    assert_resize(&events, size);
    assert_eq!(
        fixture.counts(),
        before,
        "ready application event precedes new SDK work"
    );
    fixture.poll_expired(true);
    assert_eq!(
        fixture.counts(),
        before,
        "expired fresh admission cannot enter SDK"
    );
    let denied = fixture.owner();
    assert_eq!(
        denied, None,
        "early denial leaves the idle sampler untouched"
    );
    assert!(!fixture.startup().state.borrow().slot.snapshot().pending);
    fixture.poll_expired(false);
    assert_eq!(
        fixture.owner(),
        denied,
        "repeat denial retains complete owned preparation"
    );
    assert_eq!(fixture.counts(), before);
    let (prepared, size) = boundary_priority(fixture);
    assert_eq!(
        fixture.counts(),
        before,
        "final SDK guard denies actual entry"
    );
    let events = fixture
        .host
        .poll_events()
        .expect("deliver actual event produced during preparation");
    assert_resize(&events, size);
    assert_eq!(fixture.counts(), before);
    assert_eq!(fixture.owner(), Some(prepared.clone()));
    fixture.resume(false);
    let now = fixture.counts();
    assert_eq!(now.document, before.document + 1);
    assert_eq!(now.dom, before.dom + 1);
    assert!(
        fixture
            .startup()
            .state
            .borrow()
            .slot
            .is_current(prepared.id())
    );
}

fn boundary_priority(fixture: &Fixture) -> (Owner, (u32, u32)) {
    let resized = Cell::new(None);
    fixture.startup().tick(true, || {
        // The first predicate runs before slot/argument preparation. At the
        // final predicate the actual command owns its slot but has not entered
        // the SDK. Produce a real event at this seam, rather than replacing
        // either the SDK or native readiness with a test double.
        let prepared = fixture.startup().state.borrow().slot.snapshot().pending;
        if prepared && resized.get().is_none() {
            resized.set(Some(fixture.resize()));
        }
        !fixture.host.window.has_pending_events()
            && !fixture.host.state.borrow().has_pending_events()
            && Instant::now() < fixture.deadline
    });
    let size = resized
        .get()
        .expect("actual event at prepared SDK admission");
    let owner = fixture.owner().expect("unissued command remains owned");
    let Owner::Prepared {
        id,
        stage,
        method,
        parameters,
    } = &owner
    else {
        panic!("no callback or reply exists before SDK admission")
    };
    assert!(fixture.startup().state.borrow().slot.is_current(*id));
    assert_eq!(*stage, Stage::Document);
    assert_eq!(*method, "DOM.getDocument");
    assert_eq!(*parameters, json!({"depth":0,"pierce":false}));
    (owner, size)
}

fn continuation_priority(fixture: &mut Fixture, id: RequestId, root: i64) {
    let before = fixture.counts();
    let size = fixture.resize();
    let events = fixture
        .host
        .poll_events()
        .expect("deliver resize before continuation");
    assert_resize(&events, size);
    assert_eq!(
        fixture.counts(),
        before,
        "ready event precedes continuation SDK work"
    );
    fixture.poll_expired(false);
    let denied = fixture
        .owner()
        .expect("actual reply or prepared successor survives denial");
    assert_eq!(denied.id(), id);
    match &denied {
        Owner::Reply { stage, value, .. } => {
            assert_eq!(*stage, Stage::Document);
            assert_eq!(value["root"]["nodeId"], root);
            assert_eq!(value["root"]["documentURL"], START_URI);
        }
        Owner::Prepared {
            stage,
            method,
            parameters,
            ..
        } => {
            assert_eq!(*stage, Stage::Query { index: 0, root });
            assert_eq!(*method, "DOM.querySelector");
            assert_eq!(*parameters, json!({"nodeId":root,"selector":SELECTORS[0]}));
        }
    }
    fixture.poll_expired(false);
    assert_eq!(fixture.owner(), Some(denied));
    assert_eq!(
        fixture.counts(),
        before,
        "expired continuation cannot enter SDK"
    );
    fixture.resume(false);
    assert!(fixture.startup().state.borrow().slot.is_current(id));
    assert_eq!(fixture.counts().document, before.document);
    assert_eq!(fixture.counts().dom, before.dom + 1);
}

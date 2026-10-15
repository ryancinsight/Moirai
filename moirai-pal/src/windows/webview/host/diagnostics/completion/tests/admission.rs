//! Admission uses the last actual predicate, before counting or entering the SDK.

use super::*;
use std::{
    cell::Cell,
    time::{Duration, Instant},
};

#[test]
fn denied_admission_and_expired_deadline_do_not_count_an_sdk_attempt() {
    let _trace = trace();
    let _test = tracing::info_span!("startup_sdk_entry_admission").entered();
    let (state, id, _callback) = tracing::info_span!("arrange").in_scope(request);
    let guard_calls = Cell::new(0);
    let deadline = Instant::now()
        .checked_sub(Duration::from_secs(1))
        .expect("actual past instant");
    let admitted = tracing::info_span!("act").in_scope(|| {
        admit(&Rc::downgrade(&state), "DOM.getDocument", Some(id), || {
            guard_calls.set(guard_calls.get() + 1);
            Instant::now() < deadline
        })
        .expect("expired admission is deferred")
    });
    tracing::info_span!("assert").in_scope(|| {
        assert!(!admitted);
        assert_eq!(guard_calls.get(), 1);
        let state = state.borrow();
        assert_eq!(state.dom_calls_attempted, 0);
        assert_eq!(state.activation_calls_attempted, 0);
        assert_eq!(state.document_calls_attempted, 0);
        assert!(state.slot.is_current(id));
    });
    let activation = tracing::info_span!("act").in_scope(|| {
        admit(&Rc::downgrade(&state), "Runtime.enable", None, || {
            Instant::now() < deadline
        })
        .expect("activation uses the same original deadline")
    });
    tracing::info_span!("assert").in_scope(|| {
        assert!(!activation);
        assert_eq!(state.borrow().activation_calls_attempted, 0);
    });
    for stop in ["stale", "closed"] {
        let (state, id, _callback) = tracing::info_span!("arrange", stop).in_scope(request);
        if stop == "stale" {
            state
                .borrow_mut()
                .slot
                .invalidate()
                .expect("actual generation advance");
        } else {
            state.borrow_mut().slot.close();
        }
        let admitted = tracing::info_span!("act").in_scope(|| {
            admit(
                &Rc::downgrade(&state),
                "DOM.querySelector",
                Some(id),
                || true,
            )
            .expect("cancelled command is deferred to its owner")
        });
        tracing::info_span!("assert").in_scope(|| {
            assert!(!admitted);
            assert_eq!(state.borrow().dom_calls_attempted, 0);
            assert!(state.borrow().slot.snapshot().pending);
        });
    }
}

#[test]
fn expired_owner_and_closed_activation_preserve_distinct_errors() {
    let _trace = trace();
    let _test = tracing::info_span!("startup_sdk_owner_errors").entered();
    let (state, _, _callback) = tracing::info_span!("arrange").in_scope(request);
    let weak = Rc::downgrade(&state);
    tracing::info_span!("act").in_scope(|| state.borrow_mut().slot.close());
    let closed = tracing::info_span!("act").in_scope(|| {
        admit(&weak, "Log.enable", None, || true).expect_err("closed activation is not a timeout")
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(
            closed.message(),
            "WebView2 activation observation owner closed"
        );
        assert_eq!(state.borrow().activation_calls_attempted, 0);
    });
    tracing::info_span!("act").in_scope(|| drop(state));
    let expired = tracing::info_span!("act").in_scope(|| {
        admit(&weak, "Runtime.enable", None, || true).expect_err("expired owner is not a timeout")
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(
            expired.message(),
            "WebView2 command observation owner expired"
        );
        assert_eq!(weak.strong_count(), 0);
    });
}

#[test]
fn current_admission_counts_each_domain_and_overflow_is_transactional() {
    let _trace = trace();
    let _test = tracing::info_span!("startup_sdk_admission_positive_boundary").entered();
    let (state, id, _callback) = tracing::info_span!("arrange").in_scope(request);
    for (method, request, expected) in [
        ("DOM.getDocument", Some(id), (1, 0, 1, 0)),
        ("DOM.querySelector", Some(id), (2, 0, 1, 0)),
        ("Runtime.enable", None, (2, 1, 1, 0)),
        ("DOM.describeNode", Some(id), (3, 1, 1, 1)),
    ] {
        let admitted = tracing::info_span!("act", method).in_scope(|| {
            admit(&Rc::downgrade(&state), method, request, || true)
                .expect("current admission policy")
        });
        tracing::info_span!("assert").in_scope(|| {
            assert!(admitted, "an always-deny policy cannot pass this case");
            let state = state.borrow();
            assert_eq!(
                (
                    state.dom_calls_attempted,
                    state.activation_calls_attempted,
                    state.document_calls_attempted,
                    state.description_calls_attempted,
                ),
                expected
            );
            assert!(state.slot.is_current(id));
            assert!(state.reply.is_none());
            assert_eq!(state.samples, 0);
        });
    }
    let before = tracing::info_span!("arrange").in_scope(|| {
        let mut state = state.borrow_mut();
        state.dom_calls_attempted = u64::MAX;
        (
            state.dom_calls_attempted,
            state.activation_calls_attempted,
            state.document_calls_attempted,
            state.description_calls_attempted,
        )
    });
    let overflow = tracing::info_span!("act").in_scope(|| {
        admit(&Rc::downgrade(&state), "DOM.getDocument", Some(id), || true)
            .expect_err("no partial or wrapped admission")
    });
    tracing::info_span!("assert").in_scope(|| {
        assert!(overflow.message().contains("command count exhausted"));
        let state = state.borrow();
        assert_eq!(
            (
                state.dom_calls_attempted,
                state.activation_calls_attempted,
                state.document_calls_attempted,
                state.description_calls_attempted,
            ),
            before
        );
        assert!(state.slot.is_current(id));
        assert_eq!(state.samples, 0);
    });
}

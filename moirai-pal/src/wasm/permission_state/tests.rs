use super::{Delivery, PermissionState, Phase, WebPermissionKind, deadline};
use proptest::prelude::*;
use std::{io::ErrorKind, time::Duration};

const KINDS: [WebPermissionKind; 4] = [
    WebPermissionKind::Geolocation,
    WebPermissionKind::Camera,
    WebPermissionKind::Microphone,
    WebPermissionKind::Notifications,
];

pub(super) fn tracing() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_test_writer()
            .without_time()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .finish(),
    )
}

#[test]
fn four_kinds_admit_once_and_report_actual_settlement_ownership() {
    let _trace = tracing();
    let _test = tracing::info_span!("permission_kind_admission").entered();
    let mut state = tracing::info_span!("arrange").in_scope(PermissionState::default);
    let tokens = tracing::info_span!("act")
        .in_scope(|| KINDS.map(|kind| state.begin(kind).expect("idle kind")));
    state.watch_started(tokens[0]);
    let snapshot = tracing::info_span!("assert").in_scope(|| state.resources());
    assert_eq!(snapshot.active_requests, KINDS.len());
    assert_eq!(snapshot.unsettled_guards, KINDS.len());
    assert_eq!(snapshot.geolocation_watches, 1);
    assert_eq!(snapshot.geolocation_dispatchers, 0);
    assert_eq!(snapshot.media_leases, 0);
    assert_eq!(snapshot.owned_media_tracks, 0);
    for kind in KINDS {
        let error = state.begin(kind).expect_err("occupied kind");
        assert_eq!(error.kind(), ErrorKind::AlreadyExists);
        assert_eq!(
            error.to_string(),
            format!("Browser {kind:?} request still owns its document slot")
        );
    }
    state.watch_cleared(tokens[0]);
    for token in tokens {
        assert_eq!(state.settle(token), Delivery::Publish);
        state.consumed(token);
    }
    let snapshot = state.resources();
    assert_eq!(snapshot.active_requests, 0);
    assert_eq!(snapshot.unsettled_guards, 0);
    assert_eq!(snapshot.geolocation_watches, 0);
}

#[test]
fn cancellation_retains_all_guards_until_real_settlement() {
    let _trace = tracing();
    let _test = tracing::info_span!("permission_cancelled_settlement").entered();
    for kind in KINDS {
        let mut state = PermissionState::default();
        let token =
            tracing::info_span!("arrange", ?kind).in_scope(|| state.begin(kind).expect("kind"));
        if kind == WebPermissionKind::Geolocation {
            state.watch_started(token);
            state.watch_cleared(token);
        }
        tracing::info_span!("act", ?kind).in_scope(|| state.cancel(token));
        let snapshot = state.resources();
        assert_eq!(snapshot.active_requests, 0);
        assert_eq!(snapshot.geolocation_watches, 0);
        assert_eq!(snapshot.unsettled_guards, 1);
        assert_eq!(
            state
                .begin(kind)
                .expect_err("pending remount remains busy")
                .kind(),
            ErrorKind::AlreadyExists
        );
        assert_eq!(state.settle(token), Delivery::Discard);
        let new = state.begin(kind).expect("settled slot");
        assert!(new.generation > token.generation);
        assert_eq!(state.resources().active_requests, 1);
        state.cancel(new);
        assert_eq!(state.settle(new), Delivery::Discard);
    }
}

#[test]
fn old_watch_callbacks_cannot_clear_or_settle_a_new_generation() {
    let _trace = tracing();
    let _test = tracing::info_span!("permission_watch_generation").entered();
    let mut state = PermissionState::default();
    let old = state
        .begin(WebPermissionKind::Geolocation)
        .expect("old watch");
    state.watch_started(old);
    state.watch_cleared(old);
    state.cancel(old);
    assert_eq!(state.settle(old), Delivery::Discard);
    let new = state
        .begin(WebPermissionKind::Geolocation)
        .expect("new watch");
    state.watch_started(new);
    state.watch_cleared(old);
    assert_eq!(state.settle(old), Delivery::Discard);
    assert_eq!(state.resources().active_requests, 1);
    assert_eq!(state.resources().geolocation_watches, 1);
    state.watch_cleared(new);
    assert_eq!(state.settle(new), Delivery::Publish);
    assert_eq!(state.settle(new), Delivery::Discard, "duplicate callback");
    state.consumed(new);
    assert_eq!(state.resources().active_requests, 0);
}

#[test]
fn media_leases_account_for_every_owned_track_and_release_exactly() {
    let _trace = tracing();
    let _test = tracing::info_span!("permission_media_ownership").entered();
    let mut state = PermissionState::default();
    // Each row is an admitted stream's actual track-count input to the state.
    // The expected total is the analytical sum; this is not native device evidence.
    let counts = [1, 2, 3];
    let mut total = 0;
    for (index, tracks) in counts.into_iter().enumerate() {
        tracing::info_span!("act", tracks)
            .in_scope(|| state.media_acquired(tracks).expect("track accounting"));
        total += tracks;
        assert_eq!(state.resources().media_leases, index + 1);
        assert_eq!(state.resources().owned_media_tracks, total);
    }
    for (index, tracks) in counts.into_iter().enumerate() {
        state.media_released(tracks);
        total -= tracks;
        assert_eq!(state.resources().media_leases, counts.len() - index - 1);
        assert_eq!(state.resources().owned_media_tracks, total);
    }
}

#[test]
fn generation_and_ownership_overflow_reject_without_partial_mutation() {
    let _trace = tracing();
    let _test = tracing::info_span!("permission_arithmetic_bounds").entered();
    let mut state = PermissionState::default();
    let kind = WebPermissionKind::Notifications;
    state.slots[kind.index()].generation = u64::MAX - 1;
    let token = state.begin(kind).expect("last representable generation");
    assert_eq!(token.generation, u64::MAX);
    assert_eq!(state.settle(token), Delivery::Publish);
    state.consumed(token);
    let error = state.begin(kind).expect_err("generation must not wrap");
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert_eq!(error.to_string(), "Permission generation exhausted");
    assert_eq!(state.slots[kind.index()].phase, Phase::Idle);
    state.media_tracks = usize::MAX;
    let before = state.resources();
    let error = state
        .media_acquired(1)
        .expect_err("track sum must not wrap");
    assert_eq!(error.kind(), ErrorKind::InvalidData);
    assert_eq!(error.to_string(), "Media ownership count overflowed");
    assert_eq!(state.resources(), before);
}

#[test]
fn caller_deadline_caps_at_existing_provider_law_and_rejects_zero() {
    let _trace = tracing();
    let _test = tracing::info_span!("permission_deadline_bound").entered();
    let maximum = Duration::from_millis(u64::from(
        crate::webview_limits::MAX_WEBVIEW_WAIT_MILLISECONDS,
    ));
    for wait in [
        Duration::from_nanos(1),
        maximum,
        maximum + Duration::from_nanos(1),
        Duration::MAX,
    ] {
        assert_eq!(
            deadline(wait).expect("positive deadline"),
            wait.min(maximum)
        );
    }
    let error = deadline(Duration::ZERO).expect_err("zero deadline");
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert_eq!(error.to_string(), "Permission deadline must be positive");
}

proptest! {
    #[test]
    fn complete_and_cancel_cycles_keep_slots_and_guards_bounded(kind in 0..KINDS.len(), cancellations in proptest::collection::vec(any::<bool>(), 1..100)) {
        let _trace = tracing();
        let _test = tracing::info_span!("permission_lifecycle_property").entered();
        let mut state = PermissionState::default();
        let kind = KINDS[kind];
        let mut last_generation = 0;
        for cancelled in cancellations {
            let token = state.begin(kind).expect("completed previous cycle");
            prop_assert!(token.generation > last_generation);
            last_generation = token.generation;
            if kind == WebPermissionKind::Geolocation {
                state.watch_started(token);
                state.watch_cleared(token);
            }
            if cancelled { state.cancel(token); }
            prop_assert_eq!(state.settle(token), if cancelled { Delivery::Discard } else { Delivery::Publish });
            if !cancelled { state.consumed(token); }
            let snapshot = state.resources();
            prop_assert_eq!(snapshot.active_requests, 0);
            prop_assert_eq!(snapshot.unsettled_guards, 0);
            prop_assert_eq!(snapshot.geolocation_watches, 0);
        }
    }
}

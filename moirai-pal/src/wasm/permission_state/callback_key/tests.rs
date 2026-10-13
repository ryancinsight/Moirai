use super::{CallbackKey, CallbackSequence};
use crate::permission_state::tests::tracing;
use proptest::prelude::*;
use std::io::ErrorKind;

#[test]
fn decimal_binding_preserves_the_full_identity_domain_and_rejects_alternate_grammar() {
    let _trace = tracing();
    let _test = tracing::info_span!("permission_callback_wire_identity").entered();
    // Adjacent values at ECMAScript's exact-integer boundary must stay distinct.
    for value in [1, (1u64 << 53) - 1, 1u64 << 53, (1u64 << 53) + 1, u64::MAX] {
        let key = CallbackKey(value);
        let wire = tracing::info_span!("act", value).in_scope(|| key.encode());
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(CallbackKey::decode(&wire), Some(key));
            assert_eq!(wire, value.to_string());
        });
    }
    for wire in [
        "",
        "0",
        "01",
        "+1",
        "-1",
        " 1",
        "1 ",
        "1.0",
        "1e1",
        "١",
        "18446744073709551616",
    ] {
        assert_eq!(CallbackKey::decode(wire), None, "{wire:?}");
    }
}

#[test]
fn identities_never_reuse_the_last_key_or_wrap_at_exhaustion() {
    let _trace = tracing();
    let _test = tracing::info_span!("permission_callback_sequence_bound").entered();
    let mut sequence = CallbackSequence(u64::MAX - 1);
    assert_eq!(
        sequence.next().expect("last identity"),
        CallbackKey(u64::MAX)
    );
    let error = sequence.next().expect_err("identity must not wrap");
    assert_eq!(error.kind(), ErrorKind::InvalidInput);
    assert_eq!(error.to_string(), "Geolocation callback identity exhausted");
    assert_eq!(sequence.0, u64::MAX);
}

#[test]
fn callback_sequence_is_independent_of_recreated_document_state() {
    use crate::permission_state::{Delivery, PermissionState, WebPermissionKind};

    let _trace = tracing();
    let _test = tracing::info_span!("permission_callback_remount_identity").entered();
    let mut sequence = tracing::info_span!("arrange").in_scope(CallbackSequence::default);
    for expected in 1..=4 {
        let mut document = PermissionState::default();
        let (token, key) = tracing::info_span!("act").in_scope(|| {
            (
                document
                    .begin(WebPermissionKind::Geolocation)
                    .expect("new document slot"),
                sequence.next().expect("realm identity"),
            )
        });
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(key, CallbackKey(expected));
            assert_eq!(document.settle(token), Delivery::Publish);
            document.consumed(token);
            assert_eq!(document.resources().active_requests, 0);
            assert_eq!(document.resources().unsettled_guards, 0);
        });
    }
}

proptest! {
    #[test]
    fn canonical_keys_round_trip_exactly(value in 1u64..=u64::MAX) {
        let _trace = tracing();
        let _test = tracing::info_span!("permission_callback_identity_property").entered();
        let key = CallbackKey(value);
        prop_assert_eq!(CallbackKey::decode(&key.encode()), Some(key));
    }
}

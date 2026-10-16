use super::*;
use crate::webview_limits::{MAX_WEBVIEW_EVENTS, MAX_WEBVIEW_MESSAGE_BYTES};

#[test]
fn checkpoints_and_suffix_records_share_the_inclusive_count_quota() {
    let _trace = trace();
    let _test = tracing::info_span!("late_input_count_quota").entered();
    let mut late =
        tracing::info_span!("arrange").in_scope(|| LateInput::new(NonZeroU8::new(68).expect("D")));
    let (bytes, full, exhausted, silent, progress, closed) =
        tracing::info_span!("act").in_scope(|| {
            let initial = late
                .checkpoint(0, None, true)
                .expect("initial")
                .expect("record");
            let anchor = late
                .observe(&key(1, 68, 0))
                .expect("anchor")
                .expect("record");
            let mut bytes = initial.len() + anchor.len();
            for _ in 2..MAX_WEBVIEW_EVENTS {
                let text = late
                    .observe(&json!({"kind":"controller_input", "event":{"name":"GotFocus"}}))
                    .expect("suffix")
                    .expect("inclusive retained record");
                bytes += text.len();
            }
            let full = late.budget.snapshot();
            let exhausted = late
                .observe(&key(2, 83, 0))
                .expect("first loss")
                .expect("quota terminal");
            let silent = late.observe(&key(3, 83, 1)).expect("sticky loss");
            let progress = late
                .checkpoint(3, Some(51), false)
                .expect("budgeted progress");
            let closed = late.close(false).expect("close").expect("closed terminal");
            (bytes, full, exhausted, silent, progress, closed)
        });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(full.records, MAX_WEBVIEW_EVENTS);
        assert_eq!(full.bytes, bytes);
        assert!(!full.exhausted);
        let exhausted: Value = serde_json::from_str(&exhausted).expect("quota JSON");
        let closed: Value = serde_json::from_str(&closed).expect("closed JSON");
        assert_eq!(exhausted["kind"], "late_input_quota_exhausted");
        assert_eq!(exhausted["records"], MAX_WEBVIEW_EVENTS);
        assert_eq!(exhausted["bytes"], bytes);
        assert_eq!(exhausted["omitted"], "1");
        assert_eq!(silent, None);
        assert_eq!(progress, None);
        assert_eq!(closed["records"], MAX_WEBVIEW_EVENTS);
        assert_eq!(closed["omitted"], "3");
        assert_eq!(closed["complete"], false);
        assert!(late.budget.snapshot().exhausted);
    });
}

#[test]
fn the_byte_quota_admits_exactly_65536_and_never_resets_after_exhaustion() {
    let _trace = trace();
    let _test = tracing::info_span!("late_input_byte_quota").entered();
    let mut late =
        tracing::info_span!("arrange").in_scope(|| LateInput::new(NonZeroU8::new(68).expect("D")));
    let (anchor, padding, full, quota, smaller, closed) =
        tracing::info_span!("act").in_scope(|| {
            let anchor = late
                .observe(&key(1, 68, 0))
                .expect("anchor")
                .expect("record");
            // A JSON string adds exactly two quote bytes to this ASCII payload.
            let padding = Value::String("x".repeat(MAX_WEBVIEW_MESSAGE_BYTES - anchor.len() - 2));
            let padding = late
                .observe(&padding)
                .expect("exact remainder")
                .expect("inclusive byte record");
            let full = late.budget.snapshot();
            let quota = late
                .observe(&json!(false))
                .expect("excess")
                .expect("quota terminal");
            let smaller = late.observe(&json!(0)).expect("no selective readmission");
            let closed = late.close(false).expect("close").expect("closed terminal");
            (anchor, padding, full, quota, smaller, closed)
        });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(anchor.len() + padding.len(), MAX_WEBVIEW_MESSAGE_BYTES);
        assert_eq!(full.bytes, MAX_WEBVIEW_MESSAGE_BYTES);
        assert_eq!(full.records, 2);
        assert!(!full.exhausted);
        let quota: Value = serde_json::from_str(&quota).expect("quota JSON");
        let closed: Value = serde_json::from_str(&closed).expect("closed JSON");
        assert_eq!(quota["bytes"], MAX_WEBVIEW_MESSAGE_BYTES);
        assert_eq!(quota["omitted"], "1");
        assert_eq!(smaller, None);
        assert_eq!(closed["bytes"], MAX_WEBVIEW_MESSAGE_BYTES);
        assert_eq!(closed["records"], 2);
        assert_eq!(closed["omitted"], "2");
        assert_eq!(closed["complete"], false);
    });
}

#[test]
fn exhausted_main_quota_does_not_prevent_late_retention() {
    let _trace = trace();
    let _test = tracing::info_span!("late_input_main_independence").entered();
    let (mut main, mut late) = tracing::info_span!("arrange").in_scope(|| {
        (
            TraceBudget::default(),
            LateInput::new(NonZeroU8::new(68).expect("D")),
        )
    });
    let (main_full, observed, main_after) = tracing::info_span!("act").in_scope(|| {
        for _ in 0..MAX_WEBVIEW_EVENTS {
            assert!(matches!(
                main.record(&Value::Null).expect("main record"),
                RecordAdmission::Admitted(_)
            ));
        }
        assert_eq!(
            main.record(&Value::Null).expect("main quota"),
            RecordAdmission::Exhausted { first: true }
        );
        let main_full = main.snapshot();
        let actual = key(7, 68, 0);
        let observed = late
            .observe(&actual)
            .expect("independent late owner")
            .expect("retained anchor");
        (
            main_full,
            serde_json::from_str::<Value>(&observed).expect("JSON"),
            main.snapshot(),
        )
    });
    tracing::info_span!("assert").in_scope(|| {
        assert!(main_full.exhausted);
        assert_eq!(main_full.records, MAX_WEBVIEW_EVENTS);
        assert_eq!(main_full.omitted, 1);
        assert_eq!(main_full, main_after);
        assert_eq!(observed, key(7, 68, 0));
        assert_eq!(late.budget.snapshot().records, 1);
        assert!(!late.budget.snapshot().exhausted);
    });
}

#[test]
fn exhausted_late_quota_does_not_prevent_main_retention() {
    let _trace = trace();
    let _test = tracing::info_span!("late_input_inverse_independence").entered();
    let (mut main, mut late, excessive) = tracing::info_span!("arrange").in_scope(|| {
        (
            TraceBudget::default(),
            LateInput::new(NonZeroU8::new(68).expect("D")),
            Value::String("x".repeat(MAX_WEBVIEW_MESSAGE_BYTES)),
        )
    });
    let (late_full, actual, retained) = tracing::info_span!("act").in_scope(|| {
        late.observe(&key(1, 68, 0))
            .expect("anchor")
            .expect("record");
        late.observe(&excessive)
            .expect("quota")
            .expect("quota terminal");
        let full = late.budget.snapshot();
        let actual = key(2, 83, 0);
        let retained = main.record(&actual).expect("main record");
        (full, actual, retained)
    });
    tracing::info_span!("assert").in_scope(|| {
        assert!(late_full.exhausted);
        assert_eq!(late_full.records, 1);
        assert_eq!(late.budget.snapshot(), late_full);
        let RecordAdmission::Admitted(encoded) = retained else {
            panic!("independent main quota");
        };
        assert_eq!(
            serde_json::from_str::<Value>(&encoded).expect("JSON"),
            actual
        );
        assert_eq!(main.snapshot().records, 1);
        assert_eq!(main.snapshot().bytes, encoded.len());
        assert!(!main.snapshot().exhausted);
    });
}

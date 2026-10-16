use super::*;
use std::error::Error;

#[test]
fn invalid_anchor_decimals_return_typed_failure_and_caller_retains_it() {
    let _trace = trace();
    let _test = tracing::info_span!("late_input_invalid_anchor").entered();
    let cases = tracing::info_span!("arrange").in_scope(|| {
        [
            ("sequence", json!(null)),
            ("sequence", json!(1)),
            ("sequence", json!("+1")),
            ("sequence", json!("01")),
            ("sequence", json!(" 1")),
            ("sequence", json!("1 ")),
            ("sequence", json!("-1")),
            ("sequence", json!(format!("{}0", u64::MAX))),
            ("elapsed_ns", json!(false)),
            ("elapsed_ns", json!("+17")),
            ("elapsed_ns", json!("017")),
            ("elapsed_ns", json!("-17")),
            ("elapsed_ns", json!(format!("{}0", u128::MAX))),
        ]
    });
    for (field, invalid) in cases {
        let (mut late, value) = tracing::info_span!("arrange", field, ?invalid).in_scope(|| {
            let mut value = key(1, 68, 0);
            value[field] = invalid;
            (LateInput::new(NonZeroU8::new(68).expect("D")), value)
        });
        let (error_kind, checkpoint, closed) = tracing::info_span!("act").in_scope(|| {
            let error = late
                .observe(&value)
                .expect_err("canonical decimal required");
            let kind = error.kind();
            late.fail(error);
            let checkpoint = late
                .checkpoint(1, None, false)
                .expect("error checkpoint")
                .expect("record");
            let closed = late.close(false).expect("close").expect("terminal");
            (kind, checkpoint, closed)
        });
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(error_kind, "invalid_canonical_decimal");
            assert_eq!(late.anchor, None);
            let checkpoint: Value = serde_json::from_str(&checkpoint).expect("checkpoint JSON");
            let closed: Value = serde_json::from_str(&closed).expect("closed JSON");
            assert_eq!(checkpoint["armed"], false);
            assert_eq!(checkpoint["failed"], true);
            assert_eq!(checkpoint["error"], "invalid_canonical_decimal");
            assert_eq!(closed["error"], "invalid_canonical_decimal");
            assert_eq!(closed["complete"], false);
            assert_eq!(late.budget.snapshot().records, 1);
        });
    }
}

#[test]
fn shared_counter_overflow_preserves_both_counters_atomically() {
    let _trace = trace();
    let _test = tracing::info_span!("late_input_shared_counter_exhaustion").entered();
    let mut late = tracing::info_span!("arrange").in_scope(|| {
        let mut late = LateInput::new(NonZeroU8::new(68).expect("D"));
        late.shared_omitted = u64::MAX - 1;
        late.shared_contended = u64::MAX - 1;
        late
    });
    let (last, first_error, second_error, after_first, after_second) = tracing::info_span!("act")
        .in_scope(|| {
            late.shared_loss(1, 1).expect("last representable counts");
            let last = (
                late.shared_omitted,
                late.shared_contended,
                late.failed,
                late.pending,
            );
            let first = late.shared_loss(1, 0).expect_err("omission exhaustion");
            let after_first = (
                late.shared_omitted,
                late.shared_contended,
                late.failed,
                late.pending,
            );
            // The first addition succeeds, but must not commit when the second fails.
            late.shared_omitted = 10;
            let second = late.shared_loss(5, 1).expect_err("contention exhaustion");
            let after_second = (
                late.shared_omitted,
                late.shared_contended,
                late.failed,
                late.pending,
            );
            (last, first.kind(), second.kind(), after_first, after_second)
        });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(last, (u64::MAX, u64::MAX, true, true));
        assert_eq!(first_error, "shared_omission_count_exhausted");
        assert_eq!(second_error, "shared_contention_count_exhausted");
        assert_eq!(after_first, last);
        assert_eq!(after_second, (10, u64::MAX, true, true));
        assert_eq!(late.budget.snapshot().records, 0);
    });
}

#[test]
fn actual_native_budget_error_is_retained_and_later_failures_do_not_replace_it() {
    let _trace = trace();
    let _test = tracing::info_span!("late_input_error_source").entered();
    let (mut late, error, expected) = tracing::info_span!("arrange").in_scope(|| {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("Cargo.toml")
            .join("late_input");
        let error = std::fs::read(path).expect_err("a real regular file cannot be a directory");
        let expected = (error.kind(), error.raw_os_error(), error.to_string());
        (
            LateInput::new(NonZeroU8::new(68).expect("D")),
            error,
            expected,
        )
    });
    let closed = tracing::info_span!("act").in_scope(|| {
        late.observe(&key(1, 68, 0))
            .expect("anchor")
            .expect("record");
        late.fail(LateInputFailure::Budget(error));
        late.fail(LateInputFailure::SharedOmission);
        late.close(true).expect("close").expect("terminal")
    });
    tracing::info_span!("assert").in_scope(|| {
        let retained = late.error.as_ref().expect("first failure");
        let source = retained
            .source()
            .expect("owned native cause")
            .downcast_ref::<io::Error>()
            .expect("original IO error");
        assert_eq!(
            (source.kind(), source.raw_os_error(), source.to_string()),
            expected
        );
        assert_eq!(retained.kind(), "trace_budget_error");
        let closed: Value = serde_json::from_str(&closed).expect("closed JSON");
        assert_eq!(closed["error"], "trace_budget_error");
        assert_eq!(closed["armed"], true);
        assert_eq!(closed["failed"], true);
        assert_eq!(closed["cleanup_failed"], true);
        assert_eq!(closed["complete"], false);
    });
}

#[test]
fn sequence_exhaustion_is_retained_by_the_actual_checkpoint() {
    let _trace = trace();
    let _test = tracing::info_span!("late_input_sequence_exhaustion").entered();
    let (mut late, anchor) = tracing::info_span!("arrange").in_scope(|| {
        let mut anchor = key(u64::MAX, 68, 0);
        anchor["elapsed_ns"] = json!(u128::MAX.to_string());
        (LateInput::new(NonZeroU8::new(68).expect("D")), anchor)
    });
    let (retained, checkpoint, closed) = tracing::info_span!("act").in_scope(|| {
        let retained = late
            .observe(&anchor)
            .expect("canonical maximum")
            .expect("anchor");
        let checkpoint = late
            .checkpoint(u64::MAX, Some(u128::MAX), false)
            .expect("checkpoint")
            .expect("record");
        let closed = late.close(false).expect("close").expect("terminal");
        (retained, checkpoint, closed)
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(
            serde_json::from_str::<Value>(&retained).expect("anchor JSON"),
            anchor
        );
        let checkpoint: Value = serde_json::from_str(&checkpoint).expect("checkpoint JSON");
        let closed: Value = serde_json::from_str(&closed).expect("closed JSON");
        assert_eq!(checkpoint["drained_through_sequence"], u64::MAX.to_string());
        assert_eq!(checkpoint["elapsed_ns"], u128::MAX.to_string());
        assert_eq!(checkpoint["failed"], true);
        assert_eq!(closed["complete"], false);
        assert_eq!(late.anchor, Some((u64::MAX, u128::MAX)));
    });
}

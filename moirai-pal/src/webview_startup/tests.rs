use super::{
    BudgetSnapshot, Finish, RecordAdmission, RequestSlot, RequestSnapshot, Start, TRACE_TARGET,
    TraceBudget,
};
use crate::webview_limits::{MAX_WEBVIEW_EVENTS, MAX_WEBVIEW_MESSAGE_BYTES};
use serde_json::{Value, json};
use std::io::ErrorKind;

pub(super) fn trace() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_test_writer()
            .without_time()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .finish(),
    )
}

fn admitted(result: std::io::Result<RecordAdmission>) -> String {
    match result.expect("bounded encoder") {
        RecordAdmission::Admitted(encoded) => encoded,
        RecordAdmission::Exhausted { first } => panic!("unexpected exhaustion: first={first}"),
    }
}

#[test]
fn bounded_record_matches_the_complete_reference_json_bytes() {
    let _trace = trace();
    let _test = tracing::info_span!("startup_record_value").entered();
    let (value, expected, mut budget) = tracing::info_span!("arrange").in_scope(|| {
        let value = json!({"status":"患者🦀\n\"\\", "present":true, "count":17, "missing":null});
        let expected = serde_json::to_string(&value).expect("reference JSON");
        (value, expected, TraceBudget::default())
    });
    let encoded = tracing::info_span!("act").in_scope(|| admitted(budget.record(&value)));
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(encoded, expected);
        assert_eq!(
            serde_json::from_str::<Value>(&encoded).expect("complete JSON"),
            value
        );
        assert_eq!(
            budget.snapshot(),
            BudgetSnapshot {
                records: 1,
                bytes: expected.len(),
                omitted: 0,
                exhausted: false
            }
        );
    });
}

#[test]
fn record_byte_bound_is_inclusive_without_partial_excess_admission() {
    let _trace = trace();
    let _test = tracing::info_span!("startup_record_boundary").entered();
    for bytes in [
        MAX_WEBVIEW_MESSAGE_BYTES - 1,
        MAX_WEBVIEW_MESSAGE_BYTES,
        MAX_WEBVIEW_MESSAGE_BYTES + 1,
    ] {
        let (value, mut budget) = tracing::info_span!("arrange", bytes)
            .in_scope(|| (Value::String("x".repeat(bytes - 2)), TraceBudget::default()));
        let result =
            tracing::info_span!("act").in_scope(|| budget.record(&value).expect("encoding"));
        tracing::info_span!("assert").in_scope(|| {
            if bytes <= MAX_WEBVIEW_MESSAGE_BYTES {
                let RecordAdmission::Admitted(encoded) = result else {
                    panic!("inclusive byte bound")
                };
                assert_eq!(encoded.len(), bytes);
                assert_eq!(
                    serde_json::from_str::<Value>(&encoded).expect("untruncated JSON"),
                    value
                );
                assert_eq!(
                    budget.snapshot(),
                    BudgetSnapshot {
                        records: 1,
                        bytes,
                        omitted: 0,
                        exhausted: false
                    }
                );
            } else {
                assert_eq!(result, RecordAdmission::Exhausted { first: true });
                assert_eq!(
                    budget.snapshot(),
                    BudgetSnapshot {
                        records: 0,
                        bytes: 0,
                        omitted: 1,
                        exhausted: true
                    }
                );
            }
        });
    }
}

#[test]
fn aggregate_quota_and_sticky_omissions_preserve_preceding_records() {
    let _trace = trace();
    let _test = tracing::info_span!("startup_aggregate_quota").entered();
    let (half, mut budget) = tracing::info_span!("arrange").in_scope(|| {
        (
            Value::String("x".repeat(MAX_WEBVIEW_MESSAGE_BYTES / 2 - 2)),
            TraceBudget::default(),
        )
    });
    let encoded = tracing::info_span!("act").in_scope(|| {
        [
            admitted(budget.record(&half)),
            admitted(budget.record(&half)),
        ]
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(
            encoded.iter().map(String::len).sum::<usize>(),
            MAX_WEBVIEW_MESSAGE_BYTES
        );
        assert_eq!(
            budget.snapshot(),
            BudgetSnapshot {
                records: 2,
                bytes: MAX_WEBVIEW_MESSAGE_BYTES,
                omitted: 0,
                exhausted: false
            }
        );
    });
    let omitted = tracing::info_span!("act").in_scope(|| {
        [
            budget.record(&Value::Null).expect("first omission"),
            budget.record(&json!(0)).expect("sticky omission"),
        ]
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(
            omitted,
            [
                RecordAdmission::Exhausted { first: true },
                RecordAdmission::Exhausted { first: false }
            ]
        );
        assert_eq!(
            budget.snapshot(),
            BudgetSnapshot {
                records: 2,
                bytes: MAX_WEBVIEW_MESSAGE_BYTES,
                omitted: 2,
                exhausted: true
            }
        );
    });
    let mut partial = tracing::info_span!("arrange").in_scope(TraceBudget::default);
    let first = tracing::info_span!("act").in_scope(|| admitted(partial.record(&json!({"a":1}))));
    let rejected = tracing::info_span!("act").in_scope(|| {
        partial.record(&json!({"prefix":"retained only within the attempted writer", "z":"x".repeat(MAX_WEBVIEW_MESSAGE_BYTES)})).expect("quota rejection")
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(rejected, RecordAdmission::Exhausted { first: true });
        assert_eq!(
            partial.snapshot(),
            BudgetSnapshot {
                records: 1,
                bytes: first.len(),
                omitted: 1,
                exhausted: true
            }
        );
        assert_eq!(
            serde_json::from_str::<Value>(&first).expect("preceding record"),
            json!({"a":1})
        );
    });
}

#[test]
fn record_count_exhausts_at_the_existing_native_event_limit() {
    let _trace = trace();
    let _test = tracing::info_span!("startup_record_count").entered();
    let mut budget = tracing::info_span!("arrange").in_scope(TraceBudget::default);
    for count in 1..=MAX_WEBVIEW_EVENTS {
        let encoded =
            tracing::info_span!("act", count).in_scope(|| admitted(budget.record(&Value::Null)));
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(encoded, "null");
            assert_eq!(
                budget.snapshot(),
                BudgetSnapshot {
                    records: count,
                    bytes: count * 4,
                    omitted: 0,
                    exhausted: false
                }
            );
        });
    }
    let omitted = tracing::info_span!("act")
        .in_scope(|| budget.record(&Value::Null).expect("count boundary"));
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(omitted, RecordAdmission::Exhausted { first: true });
        assert_eq!(
            budget.snapshot(),
            BudgetSnapshot {
                records: MAX_WEBVIEW_EVENTS,
                bytes: MAX_WEBVIEW_EVENTS * 4,
                omitted: 1,
                exhausted: true
            }
        );
    });
}

#[test]
fn encoded_outer_envelope_bound_fits_the_unchanged_consumer_trace_cap() {
    let _trace = trace();
    let _test = tracing::info_span!("startup_trace_envelope").entered();
    let envelope = |encoded: &str| {
        json!({"level":"DEBUG", "fields":{"event":"webview_startup_observation", "observation":encoded}, "target":TRACE_TARGET}).to_string() + "\n"
    };
    let (overhead, inputs) = tracing::info_span!("arrange").in_scope(|| {
        (
            envelope("").len(),
            [
                Value::Null,
                json!({"escaped":"\"\\\n\r\t患者🦀"}),
                json!([true, 17, "abc"]),
            ],
        )
    });
    for value in inputs {
        let (encoded, output) = tracing::info_span!("act").in_scope(|| {
            let encoded = serde_json::to_string(&value).expect("reference encoding");
            let output = envelope(&encoded);
            (encoded, output)
        });
        tracing::info_span!("assert").in_scope(|| {
            assert!(output.len() <= 2 * encoded.len() + overhead);
            assert_eq!(serde_json::from_str::<Value>(&output).expect("outer JSON")["fields"]["observation"], encoded);
        });
    }
    tracing::info_span!("assert").in_scope(|| {
        // Metis webview_acceptance::contract.MAX_TRACE_BYTES is 1 MiB.
        // Two terminal metadata records each obey the existing 64 KiB law.
        let total = 6 * MAX_WEBVIEW_MESSAGE_BYTES + overhead * (MAX_WEBVIEW_EVENTS + 2);
        assert!(
            total < 1024 * 1024,
            "derived maximum: {total}; envelope: {overhead}"
        );
        assert_eq!(TRACE_TARGET, "moirai_pal::windows::webview::startup");
    });
}

#[test]
fn invalidation_retains_the_owner_until_its_actual_completion() {
    let _trace = trace();
    let _test = tracing::info_span!("startup_request_invalidation").entered();
    let mut slot = tracing::info_span!("arrange").in_scope(RequestSlot::default);
    let first = tracing::info_span!("act").in_scope(|| {
        let Start::Begun(id) = slot.start().expect("first request") else {
            panic!("empty slot")
        };
        assert!(slot.is_current(id));
        slot.invalidate().expect("next navigation");
        id
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(
            slot.snapshot(),
            RequestSnapshot {
                generation: 2,
                pending: true,
                closed: false
            }
        );
        assert!(!slot.is_current(first));
        assert_eq!(slot.start().expect("owner remains in flight"), Start::Busy);
    });
    let (finish, second) = tracing::info_span!("act").in_scope(|| {
        let finish = slot.finish(first);
        let Start::Begun(id) = slot.start().expect("new request after terminal completion") else {
            panic!("released slot")
        };
        (finish, id)
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(finish, Finish::Stale);
        assert_ne!(first, second);
        assert!(slot.is_current(second));
        assert_eq!(slot.finish(first), Finish::Unknown);
        assert!(slot.snapshot().pending);
        assert_eq!(slot.finish(second), Finish::Current);
        assert_eq!(slot.finish(second), Finish::Unknown);
        assert_eq!(
            slot.snapshot(),
            RequestSnapshot {
                generation: 2,
                pending: false,
                closed: false
            }
        );
    });
}

#[test]
fn close_is_idempotent_and_retains_a_late_callback_owner() {
    let _trace = trace();
    let _test = tracing::info_span!("startup_request_close").entered();
    let mut slot = tracing::info_span!("arrange").in_scope(RequestSlot::default);
    let request = tracing::info_span!("act").in_scope(|| {
        let Start::Begun(id) = slot.start().expect("request") else {
            panic!("empty slot")
        };
        slot.close();
        slot.close();
        id
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(
            slot.snapshot(),
            RequestSnapshot {
                generation: 1,
                pending: true,
                closed: true
            }
        );
        assert!(!slot.is_current(request));
        assert_eq!(slot.start().expect("closed slot"), Start::Closed);
        let error = slot.invalidate().expect_err("closed generation");
        assert_eq!(error.kind(), ErrorKind::NotConnected);
        assert_eq!(error.to_string(), "Startup request slot is closed");
        assert_eq!(slot.finish(request), Finish::Stale);
        assert_eq!(
            slot.snapshot(),
            RequestSnapshot {
                generation: 1,
                pending: false,
                closed: true
            }
        );
        assert_eq!(
            slot.start().expect("closed after completion"),
            Start::Closed
        );
    });
}

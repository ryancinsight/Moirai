use super::*;
use crate::webview_startup::tests::trace;

fn key(sequence: u64, virtual_key: u8, kind: i32) -> Value {
    json!({"kind":"controller_input","sequence":sequence.to_string(),
        "elapsed_ns":(u128::from(sequence)*17).to_string(),"focus_hwnd":"123",
        "foreground_hwnd":"456","owner_keys":[0,-32768,-32768],
        "asynchronous_keys":[0,-32767,-32767],"event":{
            "name":"AcceleratorKeyPressed","virtual_key":virtual_key,
            "event_kind":kind,"lparam":-1071702015,"handled":false}})
}

#[test]
fn down_anchor_retains_exact_suffix_without_rearming_or_filtering() {
    let _trace = trace();
    let _test = tracing::info_span!("late_input_suffix").entered();
    let (mut late, records) = tracing::info_span!("arrange").in_scope(|| {
        let records = [
            key(1, 83, 0),
            key(2, 68, 1),
            key(3, 68, 3),
            key(4, 68, 2),
            key(5, 68, 3),
            key(6, 70, 0),
            key(7, 68, 0),
        ];
        (LateInput::new(NonZeroU8::new(68).expect("D")), records)
    });
    let observed = tracing::info_span!("act").in_scope(|| {
        records
            .iter()
            .filter_map(|value| {
                late.observe(value)
                    .expect("retention")
                    .map(|text| serde_json::from_str::<Value>(&text).expect("JSON"))
            })
            .collect::<Vec<_>>()
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(observed, records[3..]);
        assert_eq!(late.anchor, Some((4, 68)));
        assert_eq!(late.budget.snapshot().records, 4);
        assert_eq!(observed[2]["event"]["virtual_key"], 70);
        assert_eq!(observed[0]["owner_keys"], json!([0, -32768, -32768]));
    });
}

#[test]
fn checkpoints_are_budgeted_and_terminal_close_is_truthful_and_once() {
    let _trace = trace();
    let _test = tracing::info_span!("late_input_checkpoints").entered();
    let mut late =
        tracing::info_span!("arrange").in_scope(|| LateInput::new(NonZeroU8::new(68).expect("D")));
    let (initial, empty, loss, closed, again) = tracing::info_span!("act").in_scope(|| {
        let initial = late
            .checkpoint(0, None, true)
            .expect("initial")
            .expect("one initial");
        let empty = late.checkpoint(1, None, false).expect("no heartbeat");
        late.shared_loss(3, 2).expect("actual loss");
        let loss = late
            .checkpoint(4, None, false)
            .expect("loss checkpoint")
            .expect("retained loss");
        let closed = late.close(true).expect("actual close").expect("once");
        let again = late.close(false).expect("idempotent close");
        (initial, empty, loss, closed, again)
    });
    tracing::info_span!("assert").in_scope(|| {
        assert!(empty.is_none() && again.is_none());
        assert_eq!(late.budget.snapshot().records, 2);
        let initial: Value = serde_json::from_str(&initial).expect("initial JSON");
        let loss: Value = serde_json::from_str(&loss).expect("loss JSON");
        let closed: Value = serde_json::from_str(&closed).expect("closed JSON");
        assert_eq!(initial["armed"], false);
        assert_eq!(initial["elapsed_ns"], Value::Null);
        assert_eq!(loss["shared_capacity_omitted"], "3");
        assert_eq!(loss["shared_contended"], "2");
        assert_eq!(loss["elapsed_ns"], Value::Null);
        assert_eq!(closed["kind"], "late_input_closed");
        assert_eq!(closed["cleanup_failed"], true);
        assert_eq!(closed["complete"], false);
        assert_eq!(closed["armed"], false);
    });
}

mod failures;
mod quota;
mod representation;
mod retention;

use super::*;

fn event(sequence: u64, name: &str) -> Value {
    let mut value = key(sequence, 68, 0);
    value["event"] = json!({"name":name});
    value
}

#[test]
fn only_the_actual_selected_down_or_system_down_arms_the_anchor() {
    let _trace = trace();
    let _test = tracing::info_span!("late_input_anchor_domain").entered();
    for kind in [0, 2] {
        let (mut late, ignored, anchor) = tracing::info_span!("arrange", kind).in_scope(|| {
            let mut wrong_source = key(4, 68, 0);
            wrong_source["kind"] = json!("other");
            (
                LateInput::new(NonZeroU8::new(68).expect("D")),
                [
                    key(1, 83, 0),
                    key(2, 68, 1),
                    key(3, 68, 3),
                    wrong_source,
                    event(5, "GotFocus"),
                    event(6, "MoveFocusRequested"),
                ],
                key(7, 68, kind),
            )
        });
        let (ignored_outputs, retained) = tracing::info_span!("act").in_scope(|| {
            let outputs = ignored
                .iter()
                .map(|value| late.observe(value).expect("ignored event"))
                .collect::<Vec<_>>();
            let retained = late
                .observe(&anchor)
                .expect("actual down")
                .expect("retained anchor");
            (outputs, retained)
        });
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(ignored_outputs, vec![None; ignored.len()]);
            assert_eq!(
                serde_json::from_str::<Value>(&retained).expect("JSON"),
                anchor
            );
            assert_eq!(late.anchor, Some((7, 119)));
            assert_eq!(late.budget.snapshot().records, 1);
            assert!(!late.failed);
        });
    }
}

#[test]
fn every_post_anchor_event_is_retained_verbatim_with_one_sticky_anchor() {
    let _trace = trace();
    let _test = tracing::info_span!("late_input_complete_suffix").entered();
    let (mut late, records) = tracing::info_span!("arrange").in_scope(|| {
        let mut moved = event(4, "MoveFocusRequested");
        moved["event"]["reason"] = json!(2);
        moved["event"]["handled"] = json!(false);
        let mut error = event(6, "read_error");
        error["event"]["operation"] = json!("AcceleratorKeyPressed");
        error["event"]["hresult"] = json!(i32::MIN);
        (
            LateInput::new(NonZeroU8::new(68).expect("D")),
            [
                key(1, 68, 0),
                key(2, 83, 0),
                event(3, "LostFocus"),
                moved,
                key(5, 70, 3),
                error,
                key(7, 68, 2),
                event(8, "GotFocus"),
            ],
        )
    });
    let actual = tracing::info_span!("act").in_scope(|| {
        records
            .iter()
            .map(|value| {
                let text = late
                    .observe(value)
                    .expect("suffix")
                    .expect("retained event");
                serde_json::from_str::<Value>(&text).expect("exact JSON")
            })
            .collect::<Vec<_>>()
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(actual, records);
        assert_eq!(late.anchor, Some((1, 17)));
        assert_eq!(late.budget.snapshot().records, records.len());
        assert!(late.failed);
        assert_eq!(actual[1]["event"]["virtual_key"], 83);
        assert_eq!(actual[4]["event"]["virtual_key"], 70);
        assert_eq!(actual[5]["event"]["hresult"], i32::MIN);
    });
}

#[test]
fn a_pre_anchor_read_error_and_shared_loss_survive_without_arming() {
    let _trace = trace();
    let _test = tracing::info_span!("late_input_unarmed_failures").entered();
    let (mut late, error) = tracing::info_span!("arrange").in_scope(|| {
        let mut error = event(9, "read_error");
        error["event"]["operation"] = json!("MoveFocusRequested");
        error["event"]["hresult"] = json!(-2_147_467_259_i32);
        (LateInput::new(NonZeroU8::new(68).expect("D")), error)
    });
    let (read, checkpoint, closed) = tracing::info_span!("act").in_scope(|| {
        let read = late
            .observe(&error)
            .expect("read failure")
            .expect("retained error");
        late.shared_loss(3, 5).expect("loss counts");
        let checkpoint = late
            .checkpoint(9, Some(153), false)
            .expect("checkpoint")
            .expect("pending loss");
        let closed = late.close(false).expect("close").expect("terminal");
        (read, checkpoint, closed)
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(
            serde_json::from_str::<Value>(&read).expect("read JSON"),
            error
        );
        let checkpoint: Value = serde_json::from_str(&checkpoint).expect("checkpoint JSON");
        let closed: Value = serde_json::from_str(&closed).expect("closed JSON");
        assert_eq!(checkpoint["armed"], false);
        assert_eq!(checkpoint["failed"], true);
        assert_eq!(checkpoint["shared_capacity_omitted"], "3");
        assert_eq!(checkpoint["shared_contended"], "5");
        assert_eq!(closed["complete"], false);
        assert_eq!(late.anchor, None);
        assert_eq!(late.budget.snapshot().records, 2);
    });
}

#[test]
fn successful_close_is_once_and_post_close_observation_cannot_spend_quota() {
    let _trace = trace();
    let _test = tracing::info_span!("late_input_closed_owner").entered();
    let mut late =
        tracing::info_span!("arrange").in_scope(|| LateInput::new(NonZeroU8::new(68).expect("D")));
    let (closed, before, after, outputs) = tracing::info_span!("act").in_scope(|| {
        late.observe(&key(1, 68, 0))
            .expect("anchor")
            .expect("record");
        late.checkpoint(1, Some(17), false)
            .expect("progress")
            .expect("checkpoint");
        let closed = late.close(false).expect("close").expect("first close");
        let before = late.budget.snapshot();
        let outputs = [
            late.observe(&key(2, 83, 0)).expect("closed observation"),
            late.checkpoint(2, Some(34), true)
                .expect("closed checkpoint"),
            late.close(true).expect("second close"),
        ];
        (closed, before, late.budget.snapshot(), outputs)
    });
    tracing::info_span!("assert").in_scope(|| {
        let closed: Value = serde_json::from_str(&closed).expect("closed JSON");
        assert_eq!(closed["armed"], true);
        assert_eq!(closed["complete"], true);
        assert_eq!(closed["cleanup_failed"], false);
        assert_eq!(closed["records"], 2);
        assert_eq!(closed["drained_through_sequence"], "1");
        assert_eq!(closed["elapsed_ns"], "17");
        assert_eq!(outputs, [None, None, None]);
        assert_eq!(after, before);
    });
}

#[test]
fn shared_loss_before_and_after_arming_accumulates_without_inventing_elapsed_time() {
    let _trace = trace();
    let _test = tracing::info_span!("late_input_shared_loss_timeline").entered();
    let mut late =
        tracing::info_span!("arrange").in_scope(|| LateInput::new(NonZeroU8::new(68).expect("D")));
    let (before, after, closed) = tracing::info_span!("act").in_scope(|| {
        late.shared_loss(2, 3).expect("unarmed loss");
        let before = late
            .checkpoint(0, None, false)
            .expect("unknown-time checkpoint")
            .expect("retained loss");
        late.observe(&key(10, 68, 0))
            .expect("anchor")
            .expect("retained record");
        late.shared_loss(5, 7).expect("armed loss");
        let after = late
            .checkpoint(10, Some(170), false)
            .expect("copied snapshot time")
            .expect("retained loss");
        let closed = late.close(false).expect("close").expect("terminal");
        (before, after, closed)
    });
    tracing::info_span!("assert").in_scope(|| {
        let before: Value = serde_json::from_str(&before).expect("before JSON");
        let after: Value = serde_json::from_str(&after).expect("after JSON");
        let closed: Value = serde_json::from_str(&closed).expect("closed JSON");
        assert_eq!(before["armed"], false);
        assert_eq!(before["elapsed_ns"], Value::Null);
        assert_eq!(before["shared_capacity_omitted"], "2");
        assert_eq!(before["shared_contended"], "3");
        assert_eq!(after["armed"], true);
        assert_eq!(after["elapsed_ns"], "170");
        assert_eq!(after["shared_capacity_omitted"], "7");
        assert_eq!(after["shared_contended"], "10");
        assert_eq!(closed["failed"], true);
        assert_eq!(closed["complete"], false);
        assert_eq!(late.budget.snapshot().records, 3);
        assert_eq!(late.anchor, Some((10, 170)));
    });
}

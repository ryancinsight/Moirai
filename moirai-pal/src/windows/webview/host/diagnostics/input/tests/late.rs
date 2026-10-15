//! Real controller snapshots retain the exact independently selected suffix.

use super::{observe_controller_input, tracing};
use crate::webview_startup::SELECTION_ENV;
use serde_json::Value;

#[test]
#[ignore = "requires installed WebView2, physical input and child-selected late observations"]
fn installed_runtime_retains_selected_late_controller_input() {
    let (_outer, _subscriber) = tracing();
    let _test = tracing::info_span!("native_late_controller_input").entered();
    tracing::info_span!("arrange").in_scope(|| {
        let selected: Value = serde_json::from_str(
            &std::env::var(SELECTION_ENV).expect("explicit child-selected diagnostics"),
        )
        .expect("actual selection JSON");
        assert_eq!(selected["late_input_after_key"], 65);
    });
    // The shared owner performs the original six physical strokes, checks
    // their SDK metadata and Handled values, then closes and cleans its profile.
    let trace = tracing::info_span!("act").in_scope(|| observe_controller_input(false));
    tracing::info_span!("assert").in_scope(|| {
        let main = trace.observations();
        let late = trace.late_observations();
        assert_suffix(&main, &late);
        assert_metadata(&main, &late);
        assert!(!trace.late_bytes().is_empty());
    });
}

fn assert_suffix(main: &[Value], late: &[Value]) {
    let canonical: Vec<_> = main
        .iter()
        .filter(|value| value["kind"] == "controller_input")
        .collect();
    let anchor = canonical
        .iter()
        .position(|value| {
            value["event"]["name"] == "AcceleratorKeyPressed"
                && value["event"]["virtual_key"] == 65
                && value["event"]["event_kind"] == 0
        })
        .expect("actual physical A key down reaches the controller");
    let copied: Vec<_> = late
        .iter()
        .filter(|value| value["kind"] == "controller_input")
        .collect();
    assert_eq!(
        copied,
        canonical[anchor..],
        "same copied SDK/native values in original sequence order"
    );
    let keys: Vec<_> = copied
        .iter()
        .filter(|value| value["event"]["name"] == "AcceleratorKeyPressed")
        .collect();
    assert_eq!(keys.len(), 5);
    for (value, (key, release)) in
        keys.iter()
            .zip([(65, false), (65, true), (17, true), (27, false), (27, true)])
    {
        assert_eq!(value["event"]["virtual_key"], key);
        assert_eq!(value["event"]["event_kind"], i32::from(release));
        assert_eq!(value["event"]["handled"], false);
        let lparam = i32::try_from(
            value["event"]["lparam"]
                .as_i64()
                .expect("actual signed SDK lParam"),
        )
        .expect("Win32 signed keyboard bits")
        .cast_unsigned();
        assert_eq!(lparam >> 31 != 0, release);
        for field in ["focus_hwnd", "foreground_hwnd", "sequence", "elapsed_ns"] {
            let text = value[field].as_str().expect("canonical native decimal");
            assert_eq!(
                text.parse::<u128>().expect("native value").to_string(),
                text
            );
        }
        for field in ["owner_keys", "asynchronous_keys"] {
            let readings = value[field]
                .as_array()
                .expect("actual native key-state array");
            assert_eq!(readings.len(), 3);
            for reading in readings {
                let actual = reading.as_i64().expect("actual signed native state");
                assert_eq!(
                    i64::from(i16::try_from(actual).expect("native SHORT")),
                    actual
                );
            }
        }
    }
}

fn assert_metadata(main: &[Value], late: &[Value]) {
    let initial = late.first().expect("actual initial late checkpoint");
    assert_eq!(initial["kind"], "late_input_checkpoint");
    assert_eq!(initial["after_virtual_key"], 65);
    assert_eq!(initial["armed"], false);
    assert_eq!(initial["anchor_sequence"], Value::Null);
    assert_eq!(initial["drained_through_sequence"], "0");
    let anchor = late
        .iter()
        .find(|value| value["kind"] == "controller_input")
        .expect("retained anchor");
    let closed = late.last().expect("actual closed late owner");
    assert_eq!(closed["kind"], "late_input_closed");
    assert_eq!(
        late.iter()
            .filter(|value| value["kind"] == "late_input_closed")
            .count(),
        1
    );
    assert_eq!(closed["armed"], true);
    assert_eq!(closed["anchor_sequence"], anchor["sequence"]);
    assert_eq!(closed["anchor_elapsed_ns"], anchor["elapsed_ns"]);
    assert_eq!(closed["after_virtual_key"], 65);
    assert_eq!(closed["complete"], true);
    assert_eq!(closed["failed"], false);
    assert_eq!(closed["error"], Value::Null);
    assert_eq!(closed["cleanup_failed"], false);
    for field in ["omitted", "shared_capacity_omitted", "shared_contended"] {
        assert_eq!(closed[field], "0");
    }
    assert_eq!(closed["records"], late.len() - 1);
    assert_eq!(
        closed["bytes"],
        late[..late.len() - 1]
            .iter()
            .map(|value| value.to_string().len())
            .sum::<usize>()
    );
    let last = main
        .iter()
        .rev()
        .find(|value| value["kind"] == "controller_input")
        .expect("last actual native callback");
    assert_eq!(closed["drained_through_sequence"], last["sequence"]);
    for value in late {
        assert_ne!(value["kind"], "late_input_quota_exhausted");
        assert_ne!(value["event"]["name"], "read_error");
    }
}

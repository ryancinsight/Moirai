//! Stream framing and domain binding use exact declared data, not a UIA substitute.

use super::*;

fn progress() -> Progress {
    Progress {
        schema: 1,
        window: "34817".into(),
        mode: "startup".into(),
        replacement: 1,
        field: "reference".into(),
        expected: "患者-🦀-one".encode_utf16().collect(),
        actual: "demo".encode_utf16().collect(),
        queries: 99,
    }
}

#[test]
fn partial_utf8_and_progress_lines_preserve_original_stdout() {
    let (_trace, _subscriber) = crate::windows::webview::tests::startup::tracing();
    let bytes = tracing::info_span!("arrange").in_scope(|| {
        format!(
            "libtest 患者-🦀\n{PREFIX}{}\r\noriginal completion\n",
            serde_json::to_string(&progress()).expect("progress JSON")
        )
        .into_bytes()
    });
    for width in 1..=bytes.len() {
        let mut channel = Channel::default();
        let mut count = 0;
        tracing::info_span!("act", width).in_scope(|| {
            for chunk in bytes.chunks(width) {
                if let Some(progress) = channel.append(chunk).expect("split stream") {
                    progress
                        .validate(34817, ObservationMode::Startup)
                        .expect("exact bound progress");
                    count += 1;
                }
            }
        });
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(count, 1);
            assert_eq!(channel.bytes, bytes);
            assert_eq!(channel.scanned, bytes.len());
        });
    }
}

#[test]
fn duplicate_incomplete_and_over_budget_progress_reject() {
    let (_trace, _subscriber) = crate::windows::webview::tests::startup::tracing();
    let record = format!(
        "{PREFIX}{}\n",
        serde_json::to_string(&progress()).expect("progress JSON")
    );
    let mut channel = Channel::default();
    assert!(
        channel
            .append(record.as_bytes())
            .expect("first record")
            .is_some()
    );
    assert!(channel.append(record.as_bytes()).is_err());
    assert!(
        validate_terminal(
            record.trim_end().as_bytes(),
            34817,
            ObservationMode::Startup
        )
        .is_err()
    );
    let mut channel = Channel::default();
    let exact = vec![b'\n'; MAX_WEBVIEW_MESSAGE_BYTES];
    assert!(
        channel
            .append(&exact)
            .expect("inclusive byte bound")
            .is_none()
    );
    assert!(channel.append(b"\n").is_err());
    assert_eq!(channel.bytes, exact);
}

#[test]
fn progress_binds_owner_mode_scope_field_and_requested_units() {
    let (_trace, _subscriber) = crate::windows::webview::tests::startup::tracing();
    let original = serde_json::to_value(progress()).expect("progress representation");
    for (field, replacement) in [
        ("schema", serde_json::json!(2)),
        ("window", serde_json::json!("34818")),
        ("mode", serde_json::json!("selected")),
        ("replacement", serde_json::json!(0)),
        ("replacement", serde_json::json!(17)),
        ("field", serde_json::json!("dose")),
        ("expected", serde_json::json!([100, 101, 109, 111])),
        ("queries", serde_json::json!(0)),
        ("actual", original["expected"].clone()),
    ] {
        let mut value = original.clone();
        value[field] = replacement;
        let candidate: Progress = serde_json::from_value(value).expect("typed mutation");
        assert!(
            candidate.validate(34817, ObservationMode::Startup).is_err(),
            "{field}"
        );
    }
    let mut extra = original;
    extra["unexpected"] = true.into();
    assert!(serde_json::from_value::<Progress>(extra).is_err());
}

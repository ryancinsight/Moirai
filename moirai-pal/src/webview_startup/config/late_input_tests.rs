use super::{Components, Selection};
use crate::webview_startup::tests::trace;
use serde_json::{Value, json};
use std::io::ErrorKind;

fn selection(components: Option<&str>, key: Option<Value>) -> String {
    let mut value = json!({"start_uri":"x", "selectors":["#状態-🦀"]});
    if let Some(components) = components {
        value["components"] = json!(components);
    }
    if let Some(key) = key {
        value["late_input_after_key"] = key;
    }
    value.to_string()
}

#[test]
fn late_input_is_disabled_when_absent_or_null_for_every_original_composition() {
    let _trace = trace();
    let _test = tracing::info_span!("startup_late_input_default_off").entered();
    for (name, expected) in [
        (None, Components::Full),
        (Some("full"), Components::Full),
        (Some("controller"), Components::Controller),
        (Some("native"), Components::Native),
        (Some("events"), Components::Events),
        (Some("domain"), Components::Domain),
        (Some("snapshot"), Components::Snapshot),
        (Some("startup"), Components::Startup),
    ] {
        for key in [None, Some(Value::Null)] {
            let text =
                tracing::info_span!("arrange", ?name, ?key).in_scope(|| selection(name, key));
            let actual = tracing::info_span!("act")
                .in_scope(|| Selection::decode(&text, "x").expect("disabled late input"));
            tracing::info_span!("assert").in_scope(|| {
                assert_eq!(actual.start_uri, "x");
                assert_eq!(actual.selectors, ["#状態-🦀"]);
                assert_eq!(actual.components, expected);
                assert_eq!(actual.late_input_after_key, None);
            });
        }
    }
}

#[test]
fn late_input_preserves_exact_positive_key_values_with_controller_observations() {
    let _trace = trace();
    let _test = tracing::info_span!("startup_late_input_values").entered();
    for (name, expected) in [
        (None, Components::Full),
        (Some("full"), Components::Full),
        (Some("controller"), Components::Controller),
    ] {
        for key in [1, 68, 255] {
            let text = tracing::info_span!("arrange", ?name, key)
                .in_scope(|| selection(name, Some(json!(key))));
            let actual = tracing::info_span!("act")
                .in_scope(|| Selection::decode(&text, "x").expect("admitted exact key"));
            tracing::info_span!("assert").in_scope(|| {
                assert_eq!(actual.start_uri, "x");
                assert_eq!(actual.selectors, ["#状態-🦀"]);
                assert_eq!(actual.components, expected);
                assert_eq!(actual.late_input_after_key, Some(key));
            });
        }
    }
}

#[test]
fn late_input_rejects_zero_and_every_composition_without_controller_observations() {
    let _trace = trace();
    let _test = tracing::info_span!("startup_late_input_controller_domain").entered();
    for name in [None, Some("full"), Some("controller")] {
        let text =
            tracing::info_span!("arrange", ?name).in_scope(|| selection(name, Some(json!(0))));
        let error = tracing::info_span!("act")
            .in_scope(|| Selection::decode(&text, "x").expect_err("nonzero key required"));
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(error.kind(), ErrorKind::InvalidInput);
            assert_eq!(error.to_string(), "Startup late input key must be nonzero");
        });
    }
    for name in ["native", "events", "domain", "snapshot", "startup"] {
        for key in [1, 68, 255] {
            let text = tracing::info_span!("arrange", name, key)
                .in_scope(|| selection(Some(name), Some(json!(key))));
            let error = tracing::info_span!("act").in_scope(|| {
                Selection::decode(&text, "x").expect_err("controller observations required")
            });
            tracing::info_span!("assert").in_scope(|| {
                assert_eq!(error.kind(), ErrorKind::InvalidInput);
                assert_eq!(
                    error.to_string(),
                    "Startup late input requires controller observations"
                );
            });
        }
    }
}

#[test]
fn late_input_rejects_malformed_values_duplicates_and_unknown_fields() {
    let _trace = trace();
    let _test = tracing::info_span!("startup_late_input_closed_schema").entered();
    let cases = tracing::info_span!("arrange").in_scope(|| {
        [
            (r#""late_input_after_key":true"#, "invalid type"),
            (r#""late_input_after_key":"68""#, "invalid type"),
            (r#""late_input_after_key":1.5"#, "invalid type"),
            (r#""late_input_after_key":68.0"#, "invalid type"),
            (r#""late_input_after_key":-1"#, "invalid value"),
            (r#""late_input_after_key":256"#, "invalid value"),
            (r#""late_input_after_key":65535"#, "invalid value"),
            (
                r#""late_input_after_key":1,"late_input_after_key":68"#,
                "duplicate field",
            ),
            (
                r#""late_input_after_key":null,"late_input_after_key":null"#,
                "duplicate field",
            ),
            (
                r#""late_input_after_key":null,"late_input_after_key":1"#,
                "duplicate field",
            ),
            (
                r#""late_input_after_key":68,"late_input_after_key":null"#,
                "duplicate field",
            ),
            (r#""late_input_after_key":68,"other":1"#, "unknown field"),
        ]
    });
    for (field, expected) in cases {
        let text = tracing::info_span!("arrange", field)
            .in_scope(|| format!(r#"{{"start_uri":"x","selectors":["p"],{field}}}"#));
        let error = tracing::info_span!("act")
            .in_scope(|| Selection::decode(&text, "x").expect_err("closed key domain"));
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(error.kind(), ErrorKind::InvalidInput);
            assert!(error.to_string().contains(expected), "{error}");
            let source = error
                .get_ref()
                .and_then(|source| source.downcast_ref::<serde_json::Error>())
                .expect("original structured JSON failure");
            assert_eq!(source.line(), 1);
            assert!(source.column() > 0);
        });
    }
}

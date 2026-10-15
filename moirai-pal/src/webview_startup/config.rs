//! Closed selection data belongs to the fixture, not the native mechanism.

use super::components::Components;
use crate::webview_limits::{
    MAX_WEBVIEW_EVENTS, MAX_WEBVIEW_MESSAGE_BYTES, MAX_WEBVIEW_MESSAGE_UNITS, validate_message,
};
use serde::Deserialize;
use std::{collections::BTreeSet, io};

/// One already-admitted host URI and the fixture's requested DOM selectors.
#[derive(Debug, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct Selection {
    pub(crate) start_uri: String,
    pub(crate) selectors: Vec<String>,
    #[serde(default)]
    pub(crate) components: Components,
    /// Absent or null disables this private selection; positive values require
    /// controller observations. The byte representation admits at most 255.
    #[serde(default)]
    pub(crate) late_input_after_key: Option<u8>,
}

impl Selection {
    /// Admits the closed configuration before retained native observations.
    ///
    /// URI syntax belongs to the existing host configuration; exact equality
    /// binds this selection to that owner's already-admitted start URI. Serde's
    /// struct decoder rejects duplicate fields as well as unknown fields.
    pub(crate) fn decode(text: &str, expected_start_uri: &str) -> io::Result<Self> {
        validate_message(text.as_bytes())?;
        // Each Unicode scalar uses at least as many UTF8 bytes as UTF16 units.
        // The present equal limits make byte admission prove both bounds; the
        // constant branch eliminates a redundant scan, while preserving the
        // unit check if the canonical limits later diverge.
        if MAX_WEBVIEW_MESSAGE_BYTES > MAX_WEBVIEW_MESSAGE_UNITS
            && text.encode_utf16().count() > MAX_WEBVIEW_MESSAGE_UNITS
        {
            return Err(invalid("Startup selection exceeds the UTF16 unit bound"));
        }
        let selection: Self = serde_json::from_str(text)
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
        if selection.start_uri != expected_start_uri {
            return Err(invalid(
                "Startup selection URI differs from the configured URI",
            ));
        }
        if selection.selectors.is_empty() || selection.selectors.len() > MAX_WEBVIEW_EVENTS {
            return Err(invalid(
                "Startup selector count exceeds the admitted domain",
            ));
        }
        let mut unique = BTreeSet::new();
        for selector in &selection.selectors {
            if selector.is_empty() || selector.contains('\0') {
                return Err(invalid(
                    "Startup selectors must be nonempty and contain no NUL",
                ));
            }
            if !unique.insert(selector.as_str()) {
                return Err(invalid("Startup selectors must be unique"));
            }
        }
        if selection.late_input_after_key == Some(0) {
            return Err(invalid("Startup late input key must be nonzero"));
        }
        if selection.late_input_after_key.is_some() && !selection.components.controller() {
            return Err(invalid(
                "Startup late input requires controller observations",
            ));
        }
        Ok(selection)
    }
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}

#[cfg(test)]
mod late_input_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        webview_limits::MAX_WEBVIEW_MESSAGE_BYTES,
        webview_startup::{SELECTION_ENV, tests::trace},
    };
    use serde_json::json;
    use std::io::ErrorKind;

    fn selection(selectors: Vec<String>) -> String {
        json!({"start_uri":"https://example.invalid/index.html", "selectors":selectors}).to_string()
    }
    #[test]
    fn selection_preserves_exact_uri_and_unicode_selector_values() {
        let _trace = trace();
        let _test = tracing::info_span!("startup_selection_values").entered();
        let (selectors, text) = tracing::info_span!("arrange").in_scope(|| {
            let selectors = vec!["#状態-🦀".to_owned(), "[role=\"status\"]".to_owned()];
            let text = selection(selectors.clone());
            (selectors, text)
        });
        let value = tracing::info_span!("act").in_scope(|| {
            Selection::decode(&text, "https://example.invalid/index.html").expect("selected URI")
        });
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(value.start_uri, "https://example.invalid/index.html");
            assert_eq!(value.selectors, selectors);
            assert_eq!(value.components, Components::Full);
            assert_eq!(value.late_input_after_key, None);
            assert_eq!(SELECTION_ENV, "MOIRAI_WEBVIEW_STARTUP_OBSERVATION");
        });
    }

    #[test]
    fn selection_admits_exact_named_observer_components() {
        let _trace = trace();
        let _test = tracing::info_span!("startup_component_selection").entered();
        for (components, expected, controller, native, cdp, dom, first, later) in [
            (
                "full",
                Components::Full,
                true,
                true,
                true,
                false,
                true,
                true,
            ),
            (
                "controller",
                Components::Controller,
                true,
                false,
                false,
                false,
                false,
                false,
            ),
            (
                "native",
                Components::Native,
                false,
                true,
                false,
                false,
                false,
                false,
            ),
            (
                "events",
                Components::Events,
                false,
                true,
                true,
                false,
                false,
                false,
            ),
            (
                "domain",
                Components::Domain,
                false,
                true,
                true,
                true,
                false,
                false,
            ),
            (
                "snapshot",
                Components::Snapshot,
                false,
                true,
                true,
                false,
                true,
                false,
            ),
            (
                "startup",
                Components::Startup,
                false,
                true,
                true,
                false,
                true,
                true,
            ),
        ] {
            let text = tracing::info_span!("arrange", components).in_scope(|| {
                json!({"start_uri":"x", "selectors":["#状態-🦀"], "components":components})
                    .to_string()
            });
            let actual = tracing::info_span!("act")
                .in_scope(|| Selection::decode(&text, "x").expect("real component selection"));
            tracing::info_span!("assert").in_scope(|| {
                assert_eq!(actual.start_uri, "x");
                assert_eq!(actual.selectors, ["#状態-🦀"]);
                assert_eq!(actual.components, expected);
                assert_eq!(actual.late_input_after_key, None);
                assert_eq!(actual.components.as_str(), components);
                assert_eq!(actual.components.controller(), controller);
                assert_eq!(actual.components.native(), native);
                assert_eq!(actual.components.cdp(), cdp);
                assert_eq!(actual.components.enables_dom(), dom);
                assert_eq!(actual.components.admits_sample(0), first);
                assert_eq!(actual.components.admits_sample(1), later);
                assert_eq!(actual.components.admits_sample(u64::MAX), later);
            });
        }
    }

    #[test]
    fn selection_admits_native_and_event_controls_without_changing_selectors() {
        let _trace = trace();
        let _test = tracing::info_span!("startup_passive_controls").entered();
        for components in ["native", "events"] {
            let text = tracing::info_span!("arrange", components).in_scope(|| {
                json!({"start_uri":"x", "selectors":["#状態-🦀"], "components":components})
                    .to_string()
            });
            let selected = tracing::info_span!("act").in_scope(|| {
                Selection::decode(&text, "x").expect("real passive component control")
            });
            tracing::info_span!("assert").in_scope(|| {
                assert_eq!(selected.start_uri, "x");
                assert_eq!(selected.selectors, ["#状態-🦀"]);
                assert_eq!(selected.components.as_str(), components);
                assert!(!selected.components.controller());
            });
        }
    }

    #[test]
    fn selection_admits_domain_and_single_snapshot_controls() {
        let _trace = trace();
        let _test = tracing::info_span!("startup_sampling_controls").entered();
        for components in ["domain", "snapshot"] {
            let text = tracing::info_span!("arrange", components).in_scope(|| {
                json!({"start_uri":"x", "selectors":["#状態-🦀"], "components":components})
                    .to_string()
            });
            let selected = tracing::info_span!("act").in_scope(|| {
                Selection::decode(&text, "x").expect("real activation/sampling control")
            });
            tracing::info_span!("assert").in_scope(|| {
                assert_eq!(selected.components.as_str(), components);
                assert_eq!(selected.selectors, ["#状態-🦀"]);
            });
        }
    }

    #[test]
    fn selection_rejects_unknown_duplicate_and_malformed_fields() {
        let _trace = trace();
        let _test = tracing::info_span!("startup_selection_closed_schema").entered();
        let inputs = tracing::info_span!("arrange").in_scope(|| {
            [
                (
                    r#"{"start_uri":"x","selectors":["p"],"other":1}"#,
                    "unknown field",
                ),
                (
                    r#"{"start_uri":"x","start_uri":"x","selectors":["p"]}"#,
                    "duplicate field",
                ),
                (
                    r#"{"start_uri":"x","selectors":["p"],"selectors":["p"]}"#,
                    "duplicate field",
                ),
                (r#"{"start_uri":"x","selectors":false}"#, "invalid type"),
                (r#"{"start_uri":"x"}"#, "missing field"),
                (r#"{"start_uri":"x","selectors":["p"],"components":"other"}"#, "unknown variant"),
                (r#"{"start_uri":"x","selectors":["p"],"components":null}"#, "expected value"),
                (r#"{"start_uri":"x","selectors":["p"],"components":"full","components":"controller"}"#, "duplicate field"),
                (
                    r#"{"start_uri":"x","selectors":["p"]} true"#,
                    "trailing characters",
                ),
            ]
        });
        for (text, expected) in inputs {
            let error = tracing::info_span!("act", text)
                .in_scope(|| Selection::decode(text, "x").expect_err("closed schema"));
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

    #[test]
    fn selector_domain_admits_inclusive_count_and_rejects_each_invalid_dimension() {
        let _trace = trace();
        let _test = tracing::info_span!("startup_selector_domain").entered();
        let selectors = tracing::info_span!("arrange").in_scope(|| {
            (0..MAX_WEBVIEW_EVENTS)
                .map(|index| format!("#status-{index}"))
                .collect::<Vec<_>>()
        });
        let selected = tracing::info_span!("act").in_scope(|| {
            Selection::decode(
                &selection(selectors.clone()),
                "https://example.invalid/index.html",
            )
            .expect("inclusive selector count")
        });
        tracing::info_span!("assert").in_scope(|| assert_eq!(selected.selectors, selectors));
        let cases = tracing::info_span!("arrange").in_scope(|| {
            let mut excessive = selectors;
            excessive.push("#extra".to_owned());
            [
                (Vec::new(), "selector count"),
                (excessive, "selector count"),
                (vec![String::new()], "nonempty"),
                (vec!["#one".to_owned(), "#one".to_owned()], "unique"),
                (vec!["#a\0b".to_owned()], "contain no NUL"),
            ]
        });
        for (selectors, expected) in cases {
            let error = tracing::info_span!("act").in_scope(|| {
                Selection::decode(&selection(selectors), "https://example.invalid/index.html")
                    .expect_err("selector domain")
            });
            tracing::info_span!("assert").in_scope(|| {
                assert_eq!(error.kind(), ErrorKind::InvalidInput);
                assert!(error.to_string().contains(expected), "{error}");
            });
        }
        let error = tracing::info_span!("act").in_scope(|| {
            Selection::decode(&selection(vec!["p".to_owned()]), "https://other.invalid/")
                .expect_err("different owner URI")
        });
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(error.kind(), ErrorKind::InvalidInput);
            assert_eq!(
                error.to_string(),
                "Startup selection URI differs from the configured URI"
            );
        });
    }

    #[test]
    fn selection_representation_admits_exact_limit_and_rejects_excess() {
        let _trace = trace();
        let _test = tracing::info_span!("startup_selection_representation").entered();
        let overhead =
            tracing::info_span!("arrange").in_scope(|| selection(vec![String::new()]).len());
        for length in [
            MAX_WEBVIEW_MESSAGE_BYTES - 1,
            MAX_WEBVIEW_MESSAGE_BYTES,
            MAX_WEBVIEW_MESSAGE_BYTES + 1,
        ] {
            let text = tracing::info_span!("arrange", length)
                .in_scope(|| selection(vec!["a".repeat(length - overhead)]));
            let result = tracing::info_span!("act")
                .in_scope(|| Selection::decode(&text, "https://example.invalid/index.html"));
            tracing::info_span!("assert").in_scope(|| {
                assert_eq!(text.len(), length);
                assert_eq!(text.encode_utf16().count(), length);
                if length <= MAX_WEBVIEW_MESSAGE_BYTES {
                    assert_eq!(
                        result.expect("inclusive bound").selectors,
                        vec!["a".repeat(length - overhead)]
                    );
                } else {
                    let error = result.expect_err("excess bytes");
                    assert_eq!(error.kind(), ErrorKind::InvalidInput);
                    assert_eq!(
                        error.to_string(),
                        "WebView2 JSON message exceeds the bounded byte limit"
                    );
                }
            });
        }
    }
}

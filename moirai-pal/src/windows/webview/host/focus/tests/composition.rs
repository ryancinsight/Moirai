//! Actual installed owners and issued SDK commands identify each intervention.

use super::{ObservationMode, WebViewHost};
use serde_json::Value;

impl ObservationMode {
    pub(super) fn verify_installed(self, host: &WebViewHost, observations: &[Value]) {
        assert_eq!(host.startup.is_some(), self != Self::Ordinary);
        let registered: Vec<_> = observations
            .iter()
            .filter(|v| v["kind"] == "registered")
            .collect();
        if self == Self::Ordinary {
            assert!(registered.is_empty());
            return;
        }
        assert_eq!(registered.len(), 1, "one actual diagnostic owner");
        let (components, native, cdp, input) = match self {
            Self::Selected => ("full", 6, 3, 4),
            Self::Controller => ("controller", 0, 0, 4),
            Self::Startup => ("startup", 6, 3, 0),
            Self::Ordinary => unreachable!("ordinary returned before selected-owner checks"),
        };
        let actual = registered[0];
        assert_eq!(actual["components"], components);
        assert_eq!(actual["native_receivers"], native);
        assert_eq!(actual["cdp_receivers"], cdp);
        assert_eq!(actual["controller_receivers"], input);
    }

    pub(super) fn verify_closed(self, observations: &[Value]) {
        if self == Self::Ordinary {
            assert!(
                observations.is_empty(),
                "ordinary mode installs no observations"
            );
            return;
        }
        let closed = observations
            .last()
            .expect("actual closed diagnostic metadata");
        assert_eq!(closed["kind"], "closed");
        assert_eq!(closed["closed"], true);
        assert_eq!(closed["cleanup_failed"], false);
        let count = |field: &str| {
            closed[field]
                .as_str()
                .expect("exact counter")
                .parse::<u64>()
                .expect("native call count")
        };
        if self == Self::Controller {
            assert_eq!(count("samples"), 0);
            assert_eq!(count("dom_calls_attempted"), 0);
            assert_eq!(count("activation_calls_attempted"), 0);
            assert!(!observations.iter().any(|v| v["kind"] == "dom_status"));
        } else {
            assert_eq!(count("activation_calls_attempted"), 2);
            assert!(count("samples") > 0, "a complete real DOM chain");
            assert!(
                count("dom_calls_attempted") >= 6,
                "all three actual selectors"
            );
            for (selector, expected) in [
                (
                    "#host-status",
                    Value::String("<p id=\"host-status\">native UIA focus</p>".into()),
                ),
                (
                    "#result",
                    Value::String("<output id=\"result\">unsubmitted</output>".into()),
                ),
                ("#missing-observer-fixture", Value::Null),
            ] {
                assert!(
                    observations.iter().any(|v| v["kind"] == "dom_status"
                        && v["selector"] == selector
                        && v["value"] == expected),
                    "actual selected node {selector}"
                );
            }
        }
    }
}

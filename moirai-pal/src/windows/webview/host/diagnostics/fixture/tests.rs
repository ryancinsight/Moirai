//! Actual quota states and the real tracing targets remain independent.

use super::*;
use crate::{webview_limits::MAX_WEBVIEW_EVENTS, windows::webview::tests::startup::tracing};
use std::{
    io::{self, Write},
    sync::{Arc, Mutex},
};

#[derive(Clone, Default)]
struct Capture(Arc<Mutex<Vec<u8>>>);

impl Write for Capture {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let mut retained = self
            .0
            .lock()
            .map_err(|_| io::Error::other("fixture trace lock poisoned"))?;
        // This unit's finite two-record output uses the existing one-owner
        // capture bound; overflow rejects rather than truncating evidence.
        let bound =
            6 * crate::webview_limits::MAX_WEBVIEW_MESSAGE_BYTES + 133 * (MAX_WEBVIEW_EVENTS + 2);
        if bytes.len() > bound.saturating_sub(retained.len()) {
            return Err(io::Error::other("fixture trace capture exceeds its bound"));
        }
        retained.extend_from_slice(bytes);
        io::stderr().write_all(bytes)?;
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        io::stderr().flush()
    }
}

#[test]
fn independent_fixture_emits_exact_payload_and_terminal_metadata_on_info_target() {
    let capture = Capture::default();
    let writer = capture.clone();
    let subscriber = tracing_subscriber::fmt()
        .json()
        .without_time()
        .with_max_level(tracing::Level::DEBUG)
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
        .with_writer(move || writer.clone())
        .finish();
    let _guard = tracing::subscriber::set_default(subscriber);
    let (mut state, payload) = tracing::info_span!("arrange").in_scope(|| {
        let mut state = State::new(Selection {
            start_uri: "https://metis.invalid/index.html".into(),
            selectors: vec!["#result".into()],
            components: Components::Controller,
            late_input_after_key: None,
        });
        state.fixture_input = true;
        let payload = json!({"kind":"fixture_payload_contract", "sequence":"173",
            "units":"患者-🦀-three".encode_utf16().collect::<Vec<_>>()});
        (state, payload)
    });
    tracing::info_span!("act").in_scope(|| {
        assert!(state.record(&payload));
        state.slot.close();
        state.metadata("closed", false);
    });
    let bytes = capture.0.lock().expect("actual emitted trace").clone();
    let records: Vec<_> = std::str::from_utf8(&bytes)
        .expect("actual UTF-8 trace")
        .lines()
        .map(|line| {
            serde_json::from_str::<serde_json::Value>(line).expect("actual JSON tracing record")
        })
        .filter(|record| record["fields"]["event"] == "uia_controller_probe_observation")
        .collect();
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(records.len(), 2);
        for record in &records {
            assert_eq!(record["target"], "moirai_pal::native_uia_fixture");
            assert_eq!(record["level"], "INFO");
        }
        let decoded = |index: usize| {
            serde_json::from_str::<serde_json::Value>(
                records[index]["fields"]["observation"]
                    .as_str()
                    .expect("actual encoded payload"),
            )
            .expect("actual native record body")
        };
        assert_eq!(decoded(0), payload);
        let terminal = decoded(1);
        assert_eq!(terminal["kind"], "closed");
        assert_eq!(terminal["records"], 1);
        assert_eq!(terminal["bytes"], payload.to_string().len());
        assert_eq!(terminal["closed"], true);
        assert_eq!(terminal["exhausted"], false);
        assert_eq!(terminal["failed"], false);
        assert_eq!(terminal["cleanup_failed"], false);
    });
}

#[test]
fn fixture_controller_quota_and_target_cannot_exhaust_or_enter_main_observations() {
    let (trace, _subscriber) = tracing();
    let _scope = tracing::info_span!("controller_probe_isolation").entered();
    let selection = || Selection {
        start_uri: "https://metis.invalid/index.html".into(),
        selectors: vec!["#host-status".into(), "#result".into()],
        components: Components::Controller,
        late_input_after_key: None,
    };
    let (mut main, mut independent) = tracing::info_span!("arrange").in_scope(|| {
        let mut independent = State::new(selection());
        independent.fixture_input = true;
        (State::new(selection()), independent)
    });
    tracing::info_span!("act").in_scope(|| {
        for index in 0..MAX_WEBVIEW_EVENTS {
            assert!(independent.record(&json!({"kind":"fixture_sink_contract","index":index})));
        }
        assert!(!independent.record(&json!({"kind":"fixture_sink_overflow"})));
        independent.slot.close();
        independent.metadata("closed", false);
        assert!(main.record(&json!({"kind":"main_sink_contract","value":"retained"})));
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(independent.budget.snapshot().records, MAX_WEBVIEW_EVENTS);
        assert!(independent.budget.snapshot().exhausted);
        assert_eq!(independent.budget.snapshot().omitted, 1);
        assert_eq!(main.budget.snapshot().records, 1);
        assert!(!main.budget.snapshot().exhausted);
        assert_eq!(
            trace.observations(),
            [json!({"kind":"main_sink_contract","value":"retained"})]
        );
    });
}

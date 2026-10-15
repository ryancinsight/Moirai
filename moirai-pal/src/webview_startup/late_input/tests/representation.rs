use super::*;
use crate::{
    webview_limits::{MAX_WEBVIEW_EVENTS, MAX_WEBVIEW_MESSAGE_BYTES},
    webview_startup::TRACE_TARGET,
};
use std::{
    io::Write,
    sync::{Arc, Mutex},
};
use tracing_subscriber::fmt::MakeWriter;

#[derive(Clone)]
struct Capture(Arc<Mutex<Vec<u8>>>);

impl Write for Capture {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.0
            .lock()
            .expect("trace buffer")
            .extend_from_slice(bytes);
        Ok(bytes.len())
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl<'a> MakeWriter<'a> for Capture {
    type Writer = Self;
    fn make_writer(&'a self) -> Self::Writer {
        self.clone()
    }
}

fn formatted(observation: &str, late: bool) -> Vec<u8> {
    let capture = Capture(Arc::new(Mutex::new(Vec::new())));
    let subscriber = tracing_subscriber::fmt()
        .json()
        .without_time()
        .with_max_level(tracing::Level::DEBUG)
        .with_current_span(false)
        .with_span_list(false)
        .with_writer(capture.clone())
        .finish();
    tracing::subscriber::with_default(subscriber, || {
        if late {
            tracing::info!(target: LATE_INPUT_TARGET, parent: None, event=LATE_INPUT_EVENT, observation=%observation);
        } else {
            tracing::debug!(target: TRACE_TARGET, parent: None, event="webview_startup_observation", observation=%observation);
        }
    });
    capture.0.lock().expect("trace buffer").clone()
}

fn maximum_metadata(kind: &str) -> Value {
    let mut late = LateInput::new(NonZeroU8::new(u8::MAX).expect("maximum admitted key"));
    late.anchor = Some((u64::MAX, u128::MAX));
    late.drained = (u64::MAX, Some(u128::MAX));
    late.shared_omitted = u64::MAX;
    late.shared_contended = u64::MAX;
    late.error = Some(LateInputFailure::SharedContention);
    let mut value = late.metadata(kind, false);
    // Cartesian upper bound: unreachable joint states are permitted. Every
    // field keeps the actual serializer/schema; quota fields use their exact
    // admitted maxima and false is the longest Boolean representation.
    value["records"] = json!(MAX_WEBVIEW_EVENTS);
    value["bytes"] = json!(MAX_WEBVIEW_MESSAGE_BYTES);
    value["omitted"] = json!(u64::MAX.to_string());
    value["armed"] = json!(false);
    value["failed"] = json!(false);
    value
}

#[test]
fn maximum_metadata_uses_finite_error_kinds_and_actual_field_widths() {
    let _trace = trace();
    let _test = tracing::info_span!("late_input_maximum_metadata").entered();
    let (kinds, errors) = tracing::info_span!("arrange").in_scope(|| {
        (
            [
                "late_input_checkpoint",
                "late_input_quota_exhausted",
                "late_input_closed",
            ],
            [
                LateInputFailure::InvalidDecimal,
                LateInputFailure::SharedOmission,
                LateInputFailure::SharedContention,
                LateInputFailure::Budget(io::Error::other("source retained only")),
            ],
        )
    });
    let (records, longest_error) = tracing::info_span!("act").in_scope(|| {
        let records = kinds.map(|kind| maximum_metadata(kind).to_string());
        let longest_error = errors
            .iter()
            .map(LateInputFailure::kind)
            .max_by_key(|kind| kind.len())
            .expect("closed error domain");
        (records, longest_error)
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(longest_error, "shared_contention_count_exhausted");
        assert_eq!(
            records[1].len(),
            records
                .iter()
                .map(String::len)
                .max()
                .expect("metadata kinds")
        );
        assert_eq!(records[1].len(), LATE_INPUT_TERMINAL_BYTES);
        assert_eq!(LATE_INPUT_TERMINAL_BYTES, 534);
        assert!(
            records
                .iter()
                .all(|record| record.len() <= LATE_INPUT_TERMINAL_BYTES)
        );
        let maximum: Value = serde_json::from_str(&records[1]).expect("actual serializer");
        assert_eq!(maximum["anchor_sequence"], u64::MAX.to_string());
        assert_eq!(maximum["anchor_elapsed_ns"], u128::MAX.to_string());
        assert_eq!(maximum["omitted"], u64::MAX.to_string());
        assert_eq!(maximum["records"], MAX_WEBVIEW_EVENTS);
        assert_eq!(maximum["bytes"], MAX_WEBVIEW_MESSAGE_BYTES);
        assert_eq!(maximum["armed"], false);
        assert_eq!(maximum["failed"], false);
        assert_eq!(maximum["cleanup_failed"], false);
        assert_eq!(maximum["complete"], false);
        tracing::info!(
            metadata_bytes = records[1].len(),
            "Actual Cartesian metadata maximum"
        );
    });
}

#[test]
fn actual_info_envelope_and_escaping_keep_both_streams_below_one_mebibyte() {
    let _trace = trace();
    let _test = tracing::info_span!("late_input_physical_trace_bound").entered();
    let (sample, maximum) = tracing::info_span!("arrange").in_scope(|| {
        let mut value = json!({"prefix":"\"\\\n\t\u{0001}患者🦀",
            "event":{"name":"LostFocus"},"padding":""});
        let remaining = MAX_WEBVIEW_MESSAGE_BYTES - value.to_string().len();
        // Each quote adds two encoded bytes and both are escaped again by the
        // physical formatter. One ASCII byte covers an odd remaining length.
        let mut padding = "\"".repeat(remaining / 2);
        if !remaining.is_multiple_of(2) {
            padding.push('a');
        }
        value["padding"] = json!(padding);
        (
            value.to_string(),
            maximum_metadata("late_input_quota_exhausted").to_string(),
        )
    });
    let (main_envelope, late_envelope, physical, maximum_physical) = tracing::info_span!("act")
        .in_scope(|| {
            (
                formatted("", false),
                formatted("", true),
                formatted(&sample, true),
                formatted(&maximum, true),
            )
        });
    tracing::info_span!("assert").in_scope(|| {
        let record: Value = serde_json::from_slice(&physical).expect("physical JSON line");
        assert_eq!(record["fields"]["observation"], sample);
        assert_eq!(record["fields"]["event"], LATE_INPUT_EVENT);
        assert_eq!(record["target"], LATE_INPUT_TARGET);
        assert_eq!(record["level"], "INFO");
        assert_eq!(record.as_object().expect("record").len(), 3);
        assert_eq!(record["fields"].as_object().expect("fields").len(), 2);
        assert_eq!(physical.last(), Some(&b'\n'));
        assert_eq!(sample.len(), MAX_WEBVIEW_MESSAGE_BYTES);
        assert_eq!(late_envelope.len(), 138);
        assert!(physical.len() <= late_envelope.len() + 2 * sample.len());
        assert!(maximum_physical.len() <= late_envelope.len() + 2 * maximum.len());
        let main_bound =
            6 * MAX_WEBVIEW_MESSAGE_BYTES + main_envelope.len() * (MAX_WEBVIEW_EVENTS + 2);
        let late_bound = 2 * MAX_WEBVIEW_MESSAGE_BYTES
            + 4 * maximum.len()
            + late_envelope.len() * (MAX_WEBVIEW_EVENTS + 2);
        assert_eq!(main_bound, 427_530);
        assert_eq!(late_bound, 168_812);
        assert_eq!(main_bound + late_bound, 596_342);
        assert!(main_bound + late_bound < 1024 * 1024);
        tracing::info!(
            main_envelope_bytes = main_envelope.len(),
            late_envelope_bytes = late_envelope.len(),
            metadata_bytes = maximum.len(),
            late_bound,
            combined_bound = main_bound + late_bound,
            "Actual formatter and finite-schema physical bounds"
        );
    });
}

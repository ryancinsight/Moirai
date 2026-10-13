//! Native representation boundaries, computed before any filesystem syscall.

use std::path::Path;

#[cfg(unix)]
use super::validate_representation;
use crate::process::ProcessError;

fn trace() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_test_writer()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .finish(),
    )
}

#[cfg(unix)]
#[test]
fn native_path_budget_is_inclusive_before_the_terminator() {
    let _trace = trace();
    let _test = tracing::info_span!("process_native_path_boundary").entered();
    let (valid, excessive) = tracing::info_span!("arrange").in_scope(|| {
        let capacity = usize::try_from(libc::PATH_MAX).expect("positive native PATH_MAX");
        (
            format!("/{}", "a".repeat(capacity - 2)),
            format!("/{}", "b".repeat(capacity - 1)),
        )
    });
    let actual = tracing::info_span!("act").in_scope(|| {
        [
            validate_representation(Path::new(&valid)),
            validate_representation(Path::new(&excessive)),
        ]
    });
    tracing::info_span!("assert").in_scope(|| {
        let [valid_result, excessive_result] = actual;
        assert_eq!(
            valid_result.expect("maximum native path plus terminator"),
            ()
        );
        assert_eq!(
            excessive_result.expect_err("one byte beyond native path budget"),
            ProcessError::InvalidSpecification
        );
    });
}

#[cfg(windows)]
#[test]
fn native_directory_budget_counts_unicode_separator_and_terminator() {
    use super::{DIRECTORY_CAPACITY, encode};

    let _trace = trace();
    let _test = tracing::info_span!("process_windows_directory_boundary").entered();
    let inputs = tracing::info_span!("arrange").in_scope(|| {
        let prefix = "C:\\Zoë-李-🦀\\";
        let payload = DIRECTORY_CAPACITY - prefix.encode_utf16().count() - 2;
        let valid = format!("{prefix}{}", "a".repeat(payload));
        [
            valid.clone(),
            format!("{valid}\\"),
            format!("{valid}b"),
            format!("{valid}b\\"),
        ]
    });
    let actual = tracing::info_span!("act").in_scope(|| {
        inputs
            .iter()
            .map(|input| encode(Path::new(input)))
            .collect::<Vec<_>>()
    });
    tracing::info_span!("assert").in_scope(|| {
        for (index, result) in actual.into_iter().enumerate() {
            if index < 2 {
                assert_eq!(
                    result.expect("inclusive native path"),
                    format!("{}\0", inputs[index])
                        .encode_utf16()
                        .collect::<Vec<_>>()
                );
            } else {
                assert_eq!(
                    result.expect_err("separator and terminator exceed MAX_PATH"),
                    ProcessError::InvalidSpecification
                );
            }
        }
    });
}

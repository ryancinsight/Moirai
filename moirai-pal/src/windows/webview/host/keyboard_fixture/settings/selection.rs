//! The global experiment is admitted only for its observer-free default arm.

use std::{ffi::OsStr, io};

/// G is selected exactly when its value is recognized and D ∧ ¬O holds.
/// Absence admits all existing arms without changing their SDK call graph.
pub(super) fn selection(
    value: Option<&OsStr>,
    default_policy: bool,
    observer: bool,
) -> io::Result<Option<bool>> {
    let Some(value) = value else { return Ok(None) };
    let requested = match value.to_str() {
        Some("enabled") => true,
        Some("disabled") => false,
        _ => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "unknown global keyboard setting",
            ));
        }
    };
    if !default_policy || observer {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "global keyboard setting requires observer-none/default policy",
        ));
    }
    Ok(Some(requested))
}

#[cfg(test)]
mod tests {
    use std::ffi::OsStr;

    fn trace() -> tracing::subscriber::DefaultGuard {
        tracing::subscriber::set_default(
            tracing_subscriber::fmt()
                .with_test_writer()
                .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
                .finish(),
        )
    }

    #[test]
    fn absent_selection_preserves_every_existing_arm() {
        let _trace = trace();
        let contexts = tracing::info_span!("arrange")
            .in_scope(|| [(false, false), (false, true), (true, false), (true, true)]);
        for (default_policy, observer) in contexts {
            let actual = tracing::info_span!("act", default_policy, observer)
                .in_scope(|| super::selection(None, default_policy, observer));
            tracing::info_span!("assert")
                .in_scope(|| assert_eq!(actual.expect("absent control is valid"), None));
        }
    }

    #[test]
    fn explicit_setting_has_a_closed_value_and_context_domain() {
        let _trace = trace();
        let values = tracing::info_span!("arrange").in_scope(|| {
            [
                ("enabled", Some(true)),
                ("disabled", Some(false)),
                ("", None),
                ("true", None),
                ("false", None),
                ("ENABLED", None),
                (" enabled", None),
            ]
        });
        for (value, expected) in values {
            for (default_policy, observer) in
                [(false, false), (false, true), (true, false), (true, true)]
            {
                let actual =
                    tracing::info_span!("act", value, default_policy, observer).in_scope(|| {
                        super::selection(Some(OsStr::new(value)), default_policy, observer)
                    });
                tracing::info_span!("assert").in_scope(|| {
                    if default_policy
                        && !observer
                        && let Some(expected) = expected
                    {
                        assert_eq!(actual.expect("valid explicit control"), Some(expected));
                    } else {
                        assert_eq!(
                            actual.expect_err("invalid explicit control").kind(),
                            std::io::ErrorKind::InvalidInput
                        );
                    }
                });
            }
        }
    }
}

//! Pure state reconciliation for kqueue's independent read/write filters.

use std::io;

use crate::Interest;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum InterestFilter {
    Readable,
    Writable,
}

impl InterestFilter {
    fn enabled(self, interest: Interest) -> bool {
        match self {
            Self::Readable => interest.readable,
            Self::Writable => interest.writable,
        }
    }

    fn set(self, interest: &mut Interest, enabled: bool) {
        match self {
            Self::Readable => interest.readable = enabled,
            Self::Writable => interest.writable = enabled,
        }
    }
}

pub(crate) enum FilterChange {
    Applied,
    AlreadyAbsent(io::Error),
}

pub(crate) struct TransitionFailure {
    pub(crate) error: io::Error,
    pub(crate) lifecycle_lost: bool,
}

pub(crate) struct InterestTransition {
    pub(crate) actual: Interest,
    pub(crate) failure: Option<TransitionFailure>,
}

/// Apply independent filter changes while preserving an exact postcondition.
///
/// An expected filter that is already absent proves that the descriptor's
/// kernel registration lifecycle no longer matches the sidecar. The remaining
/// sidecar filters are then deleted as one generation before returning; an
/// unchanged filter must never be inherited across descriptor reuse.
pub(crate) fn transition_interest(
    current: Interest,
    desired: Interest,
    mut change: impl FnMut(InterestFilter, bool) -> io::Result<FilterChange>,
) -> InterestTransition {
    let mut actual = current;
    let mut first_error = None;
    for filter in [InterestFilter::Readable, InterestFilter::Writable] {
        let desired_enabled = filter.enabled(desired);
        if filter.enabled(actual) == desired_enabled {
            continue;
        }
        match change(filter, desired_enabled) {
            Ok(FilterChange::Applied) => filter.set(&mut actual, desired_enabled),
            Ok(FilterChange::AlreadyAbsent(error)) => {
                debug_assert!(!desired_enabled, "only deletion can prove absence");
                filter.set(&mut actual, false);
                collapse_generation(&mut actual, &mut change);
                return InterestTransition {
                    actual,
                    failure: Some(TransitionFailure {
                        error,
                        lifecycle_lost: true,
                    }),
                };
            }
            Err(error) => {
                if first_error.is_none() {
                    first_error = Some(error);
                }
            }
        }
    }
    InterestTransition {
        actual,
        failure: first_error.map(|error| TransitionFailure {
            error,
            lifecycle_lost: false,
        }),
    }
}

fn collapse_generation(
    actual: &mut Interest,
    change: &mut impl FnMut(InterestFilter, bool) -> io::Result<FilterChange>,
) {
    for filter in [InterestFilter::Readable, InterestFilter::Writable] {
        if !filter.enabled(*actual) {
            continue;
        }
        match change(filter, false) {
            Ok(FilterChange::Applied | FilterChange::AlreadyAbsent(_)) => {
                filter.set(actual, false);
            }
            Err(_) => {
                // A receipt error leaves the pre-change filter state intact.
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ReceiptErrorDisposition {
    Applied,
    Failed,
}

/// Classify a syscall-level receipt error by its documented postcondition.
///
/// FreeBSD and NetBSD `kevent(2)` explicitly state that every changelist entry
/// is applied before an `EINTR` return. OpenBSD applies the complete changelist
/// before placing return events. Darwin XNU likewise registers and copies an
/// `EV_RECEIPT` before entering its interruptible event scan. This backend
/// submits one receipt with one output slot, so interruption is a definitive
/// applied postcondition, not a reason to replay the change.
///
/// Sources: [FreeBSD `kevent(2)`], [NetBSD `kevent(2)`], [OpenBSD `kevent(2)`],
/// and Darwin XNU [`kevent_internal`].
///
/// [FreeBSD `kevent(2)`]: https://man.freebsd.org/cgi/man.cgi?query=kevent&sektion=2
/// [NetBSD `kevent(2)`]: https://man.netbsd.org/kevent.2
/// [OpenBSD `kevent(2)`]: https://man.openbsd.org/kevent.2
/// [`kevent_internal`]: https://github.com/apple-oss-distributions/xnu/blob/main/bsd/kern/kern_event.c#L7638-L7725
pub(crate) fn classify_receipt_error(kind: io::ErrorKind) -> ReceiptErrorDisposition {
    if kind == io::ErrorKind::Interrupted {
        ReceiptErrorDisposition::Applied
    } else {
        ReceiptErrorDisposition::Failed
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absent_changed_filter_collapses_unchanged_sibling() {
        let mut calls = Vec::new();
        let transition = transition_interest(
            Interest::READ_WRITE,
            Interest::WRITABLE,
            |filter, enabled| {
                calls.push((filter, enabled));
                if calls.len() == 1 {
                    Ok(FilterChange::AlreadyAbsent(io::Error::new(
                        io::ErrorKind::NotFound,
                        "injected lifecycle loss",
                    )))
                } else {
                    Ok(FilterChange::Applied)
                }
            },
        );

        assert_eq!(
            calls,
            [
                (InterestFilter::Readable, false),
                (InterestFilter::Writable, false),
            ]
        );
        assert!(!transition.actual.readable);
        assert!(!transition.actual.writable);
        let failure = transition.failure.expect("lifecycle loss is reported");
        assert_eq!(failure.error.kind(), io::ErrorKind::NotFound);
        assert!(failure.lifecycle_lost);
    }

    #[test]
    fn collapse_failure_retains_exact_sibling_state() {
        let mut calls = 0;
        let transition = transition_interest(
            Interest::READ_WRITE,
            Interest::WRITABLE,
            |_filter, _enabled| {
                calls += 1;
                if calls == 1 {
                    Ok(FilterChange::AlreadyAbsent(io::Error::new(
                        io::ErrorKind::NotFound,
                        "injected lifecycle loss",
                    )))
                } else {
                    Err(io::Error::other("injected collapse failure"))
                }
            },
        );

        assert!(!transition.actual.readable);
        assert!(transition.actual.writable);
        assert_eq!(calls, 2);
        let failure = transition.failure.expect("lifecycle loss is reported");
        assert_eq!(failure.error.kind(), io::ErrorKind::NotFound);
        assert!(failure.lifecycle_lost);
    }

    #[test]
    fn consecutive_interrupted_receipts_are_definitively_applied() {
        let dispositions = [io::ErrorKind::Interrupted; 2].map(classify_receipt_error);
        assert_eq!(
            dispositions,
            [
                ReceiptErrorDisposition::Applied,
                ReceiptErrorDisposition::Applied
            ]
        );
        assert_eq!(
            classify_receipt_error(io::ErrorKind::InvalidInput),
            ReceiptErrorDisposition::Failed
        );
    }
}

use super::{WebMediaLease, WebPermissionError, WebPermissionKind};

/// The three actual values returned by Notification.requestPermission.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebNotificationPermission {
    /// The browser grants notification permission.
    Granted,
    /// The browser denies notification permission.
    Denied,
    /// The browser returns its undecided/default state.
    Default,
}

/// Permission result classification without conflating default or unavailable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebPermissionStatus {
    /// A real position callback, media stream or notification grant occurs.
    Granted,
    /// The actual browser operation rejects or reports a native error.
    Rejected,
    /// Notification.requestPermission explicitly returns denied.
    Denied,
    /// Notification.requestPermission explicitly returns default.
    Default,
    /// The requested browser interface or method is absent.
    Unavailable,
}

/// Actual request outcome; a media grant owns the acquired stream.
pub enum WebPermissionOutcome {
    /// The browser interface/method is absent; this is not a denial.
    Unavailable(WebPermissionKind),
    /// The browser reports its real rejection context.
    Rejected {
        /// Requested mechanism.
        kind: WebPermissionKind,
        /// Native browser error fields.
        error: WebPermissionError,
    },
    /// The first real geolocation position callback succeeds; no coordinates remain.
    GeolocationGranted,
    /// A real acquired stream whose tracks stop on lease drop.
    MediaGranted(WebMediaLease),
    /// The browser's actual notification permission value.
    Notification(WebNotificationPermission),
}

impl WebPermissionOutcome {
    /// Requested mechanism associated with this actual outcome.
    #[must_use]
    pub fn kind(&self) -> WebPermissionKind {
        match self {
            Self::Unavailable(kind) | Self::Rejected { kind, .. } => *kind,
            Self::GeolocationGranted => WebPermissionKind::Geolocation,
            Self::MediaGranted(lease) => lease.kind(),
            Self::Notification(_) => WebPermissionKind::Notifications,
        }
    }

    /// Exact result category, retaining denial/default/unavailable distinctions.
    #[must_use]
    pub const fn status(&self) -> WebPermissionStatus {
        match self {
            Self::Unavailable(_) => WebPermissionStatus::Unavailable,
            Self::Rejected { .. } => WebPermissionStatus::Rejected,
            Self::GeolocationGranted
            | Self::MediaGranted(_)
            | Self::Notification(WebNotificationPermission::Granted) => {
                WebPermissionStatus::Granted
            }
            Self::Notification(WebNotificationPermission::Denied) => WebPermissionStatus::Denied,
            Self::Notification(WebNotificationPermission::Default) => WebPermissionStatus::Default,
        }
    }

    /// Actual browser error, present only for a rejected operation.
    #[must_use]
    pub const fn error(&self) -> Option<&WebPermissionError> {
        match self {
            Self::Rejected { error, .. } => Some(error),
            _ => None,
        }
    }
}

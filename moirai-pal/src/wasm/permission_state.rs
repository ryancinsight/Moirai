//! Per-document permission admission and cancellation state, independent of WebIDL.

use std::{io, time::Duration};

/// Browser permission mechanism requested by an application.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebPermissionKind {
    /// First actual geolocation position or error.
    Geolocation,
    /// A video-only media stream.
    Camera,
    /// An audio-only media stream.
    Microphone,
    /// The browser's three-valued notification permission.
    Notifications,
}

impl WebPermissionKind {
    pub(crate) const fn index(self) -> usize {
        match self {
            Self::Geolocation => 0,
            Self::Camera => 1,
            Self::Microphone => 2,
            Self::Notifications => 3,
        }
    }
}

/// Resources owned by one document's permission provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WebPermissionResources {
    /// Requests whose consumer future has not completed or been dropped.
    pub active_requests: usize,
    /// Registered geolocation watches not yet cleared.
    pub geolocation_watches: usize,
    /// Realm-owned geolocation callback entry points: zero before first use,
    /// then two until document destruction. Queued callbacks can outlive a watch.
    pub geolocation_dispatchers: usize,
    /// Acquired media leases not yet dropped.
    pub media_leases: usize,
    /// Media tracks owned by those leases, including already-ended tracks.
    pub owned_media_tracks: usize,
    /// Requests whose actual browser settlement remains owned, including after
    /// consumer cancellation. A cleared geolocation watch may never callback;
    /// its guard then remains until document destruction. The maximum is four.
    pub unsettled_guards: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
enum Phase {
    #[default]
    Idle,
    ActivePending,
    ActiveSettled,
    DetachedPending,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct Slot {
    generation: u64,
    phase: Phase,
    watch: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Token {
    pub(crate) kind: WebPermissionKind,
    generation: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Delivery {
    Publish,
    Discard,
}

/// Four slots are the cardinality of the closed permission-kind domain.
/// Generations never wrap: a stale callback cannot settle a newer request.
#[derive(Debug, Default)]
pub(crate) struct PermissionState {
    slots: [Slot; 4],
    media_leases: usize,
    media_tracks: usize,
}

impl PermissionState {
    pub(crate) fn begin(&mut self, kind: WebPermissionKind) -> io::Result<Token> {
        let slot = &mut self.slots[kind.index()];
        if slot.phase != Phase::Idle {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!("Browser {kind:?} request still owns its document slot"),
            ));
        }
        slot.generation = slot.generation.checked_add(1).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "Permission generation exhausted",
            )
        })?;
        slot.phase = Phase::ActivePending;
        Ok(Token {
            kind,
            generation: slot.generation,
        })
    }

    fn slot(&mut self, token: Token) -> Option<&mut Slot> {
        let slot = &mut self.slots[token.kind.index()];
        (slot.generation == token.generation).then_some(slot)
    }

    pub(crate) fn watch_started(&mut self, token: Token) {
        if let Some(slot) = self.slot(token) {
            assert_eq!(token.kind, WebPermissionKind::Geolocation);
            assert_eq!(slot.phase, Phase::ActivePending);
            assert!(!slot.watch);
            slot.watch = true;
        }
    }

    pub(crate) fn watch_cleared(&mut self, token: Token) {
        if let Some(slot) = self.slot(token) {
            slot.watch = false;
        }
    }

    pub(crate) fn settle(&mut self, token: Token) -> Delivery {
        let Some(slot) = self.slot(token) else {
            return Delivery::Discard;
        };
        match slot.phase {
            Phase::ActivePending => {
                assert!(!slot.watch, "clear actual watch before settlement");
                slot.phase = Phase::ActiveSettled;
                Delivery::Publish
            }
            Phase::DetachedPending => {
                slot.phase = Phase::Idle;
                Delivery::Discard
            }
            Phase::Idle | Phase::ActiveSettled => Delivery::Discard,
        }
    }

    pub(crate) fn consumed(&mut self, token: Token) {
        if let Some(slot) = self.slot(token) {
            assert_eq!(slot.phase, Phase::ActiveSettled);
            slot.phase = Phase::Idle;
        }
    }

    pub(crate) fn cancel(&mut self, token: Token) {
        if let Some(slot) = self.slot(token) {
            assert!(!slot.watch, "clear actual watch before cancellation");
            slot.phase = match slot.phase {
                Phase::ActivePending => Phase::DetachedPending,
                Phase::ActiveSettled => Phase::Idle,
                phase => phase,
            };
        }
    }

    pub(crate) fn media_acquired(&mut self, tracks: usize) -> io::Result<()> {
        let leases = self.media_leases.checked_add(1);
        let total = self.media_tracks.checked_add(tracks);
        match (leases, total) {
            (Some(leases), Some(total)) => {
                self.media_leases = leases;
                self.media_tracks = total;
                Ok(())
            }
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "Media ownership count overflowed",
            )),
        }
    }

    pub(crate) fn media_released(&mut self, tracks: usize) {
        self.media_leases = self.media_leases.checked_sub(1).expect("owned media lease");
        self.media_tracks = self
            .media_tracks
            .checked_sub(tracks)
            .expect("owned media tracks");
    }

    pub(crate) fn resources(&self) -> WebPermissionResources {
        let active = self
            .slots
            .iter()
            .filter(|s| matches!(s.phase, Phase::ActivePending | Phase::ActiveSettled))
            .count();
        let unsettled = self
            .slots
            .iter()
            .filter(|s| matches!(s.phase, Phase::ActivePending | Phase::DetachedPending))
            .count();
        WebPermissionResources {
            active_requests: active,
            geolocation_watches: usize::from(
                self.slots[WebPermissionKind::Geolocation.index()].watch,
            ),
            // The portable admission model owns no WebIDL callbacks. The actual
            // browser provider adds its realm-owned dispatcher count to this.
            geolocation_dispatchers: 0,
            media_leases: self.media_leases,
            owned_media_tracks: self.media_tracks,
            unsettled_guards: unsettled,
        }
    }
}

#[path = "permission_state/callback_key.rs"]
pub(crate) mod callback_key;

pub(crate) fn deadline(duration: Duration) -> io::Result<Duration> {
    if duration.is_zero() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "Permission deadline must be positive",
        ));
    }
    Ok(duration.min(Duration::from_millis(u64::from(
        crate::webview_limits::MAX_WEBVIEW_WAIT_MILLISECONDS,
    ))))
}

#[cfg(test)]
#[path = "permission_state/tests.rs"]
mod tests;

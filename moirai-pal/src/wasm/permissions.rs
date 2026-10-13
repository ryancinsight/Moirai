//! Real browser permission requests with owned cancellation and late cleanup.
//!
//! Geolocation resolves on its first native callback and clears its watch.
//! Cancellation suppresses delivery while bounded settlement guards remain
//! until actual settlement. `clearWatch` does not retract already-queued
//! callbacks or permission waiting, so two realm-owned dispatchers remain valid
//! until document destruction. A cleared watch may never callback, leaving its
//! kind busy for the rest of that document's lifetime.
//! Media grant leases stop every acquired track when dropped. Coordinates are
//! neither retained nor published by this capability-probe mechanism.

mod error;
mod geolocation;
mod media;
mod notification;
mod outcome;
mod registry;
mod request;

pub use crate::permission_state::{WebPermissionKind, WebPermissionResources};
pub use error::WebPermissionError;
pub use media::WebMediaLease;
pub use outcome::{WebNotificationPermission, WebPermissionOutcome, WebPermissionStatus};
pub use request::WebPermissionRequest;

use registry::Registry;
use std::{io, rc::Rc, time::Duration};
use web_sys::{Document, Window};

/// Permission mechanisms bound to the actual current document.
/// Clones and remounts share the same bounded per-kind admission state.
#[derive(Clone)]
pub struct WebPermissions {
    window: Window,
    document: Document,
    registry: Rc<Registry>,
}

impl WebPermissions {
    /// Acquires the current browser document and its shared permission state.
    ///
    /// # Errors
    /// Returns `Unsupported` outside a Window/Document context.
    pub fn current() -> io::Result<Self> {
        let window = web_sys::window().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                "Browser permissions require a Window",
            )
        })?;
        let document = window.document().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                "Browser permissions require a Document",
            )
        })?;
        let registry = registry::for_document(&document);
        Ok(Self {
            window,
            document,
            registry,
        })
    }

    /// Starts one actual permission request, retaining ownership until settled.
    ///
    /// A positive deadline is capped by the existing 30-second native WebView
    /// operation bound. Drop clears a geolocation watch and cancels publication.
    /// Any unsettled browser request keeps its document slot busy;
    /// remount does not create another request of that kind before settlement.
    ///
    /// # Errors
    /// Returns `AlreadyExists` for an occupied kind, `InvalidInput` for a zero
    /// deadline or exhausted generation, and an error if timer registration or
    /// the bound document is unavailable. Actual browser rejection is a typed
    /// outcome, with its native code/name/message preserved.
    pub fn request(
        &self,
        kind: WebPermissionKind,
        wait: Duration,
    ) -> io::Result<WebPermissionRequest> {
        if self.window.document().as_ref() != Some(&self.document) {
            return Err(io::Error::new(
                io::ErrorKind::NotConnected,
                "Permission document is no longer current",
            ));
        }
        request::start(&self.window, Rc::clone(&self.registry), kind, wait)
    }

    /// Counts actual provider ownership for this document.
    /// Unsettled guards remain visible after consumer cancellation.
    #[must_use]
    pub fn resource_snapshot(&self) -> WebPermissionResources {
        let mut resources = self.registry.state.borrow().resources();
        resources.geolocation_dispatchers = geolocation::dispatcher_count();
        resources
    }
}

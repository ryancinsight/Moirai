//! One EventTarget callback owner, including a counted failed teardown guard.

use std::{
    cell::{Cell, RefCell},
    io,
};

use wasm_bindgen::{JsCast, closure::Closure};
use web_sys::{EventTarget, MessageEvent};

use super::message;
use crate::wasm::browser_error::javascript_error;

thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static FAILED_CLEANUP: RefCell<Option<ListenerParts>> = const { RefCell::new(None) };
}

struct ListenerParts {
    target: EventTarget,
    // Type erasure is the runtime browser callback ABI (interop condition 2),
    // not a throughput policy or an application callback collection.
    callback: Closure<dyn FnMut(MessageEvent)>,
}

/// The sole native-message listener owned by one browser document.
///
/// Successful close detaches before releasing the callback. Drop attempts
/// close; if the browser refuses removal, a counted realm-owned guard retains
/// the callback and refuses new listeners until the realm is destroyed.
#[must_use = "retain the listener until explicit close or document teardown"]
pub struct WebViewMessageListener {
    parts: Option<ListenerParts>,
}

impl WebViewMessageListener {
    pub(super) fn attach<F>(target: EventTarget, mut callback: F) -> io::Result<Self>
    where
        F: FnMut(io::Result<String>) + 'static,
    {
        if retained_cleanup_count() != 0 || ACTIVE.with(|active| active.replace(true)) {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "WebView2 document already owns a message listener or failed cleanup guard",
            ));
        }
        let callback =
            Closure::new(move |event: MessageEvent| callback(message::receive(event.data())));
        if let Err(error) =
            target.add_event_listener_with_callback("message", callback.as_ref().unchecked_ref())
        {
            ACTIVE.with(|active| active.set(false));
            return Err(javascript_error(
                "WebView2 addEventListener",
                io::ErrorKind::Other,
                error,
            ));
        }
        Ok(Self {
            parts: Some(ListenerParts { target, callback }),
        })
    }

    /// Detaches this callback before releasing its Rust closure.
    ///
    /// # Errors
    /// An actual browser removal exception leaves this owner active. Dropping
    /// it retains the callback in the counted guard if removal fails again.
    pub fn close(&mut self) -> io::Result<()> {
        let Some(parts) = self.parts.as_ref() else {
            return Ok(());
        };
        parts
            .target
            .remove_event_listener_with_callback("message", parts.callback.as_ref().unchecked_ref())
            .map_err(|error| {
                javascript_error("WebView2 removeEventListener", io::ErrorKind::Other, error)
            })?;
        self.parts.take();
        ACTIVE.with(|active| active.set(false));
        Ok(())
    }
}

impl Drop for WebViewMessageListener {
    fn drop(&mut self) {
        if let Err(error) = self.close() {
            tracing::error!(error = %error, "WebView2 callback teardown failed; retaining the realm owner");
            if let Some(parts) = self.parts.take() {
                // ACTIVE admits exactly one owner, and FAILED_CLEANUP blocks
                // every subsequent attachment, so this slot cannot be occupied.
                FAILED_CLEANUP.with(|guard| {
                    let previous = guard.borrow_mut().replace(parts);
                    assert!(previous.is_none(), "single WebView2 failed cleanup owner");
                });
                ACTIVE.with(|active| active.set(false));
            }
        }
    }
}

pub(super) fn retained_cleanup_count() -> usize {
    FAILED_CLEANUP.with(|guard| usize::from(guard.borrow().is_some()))
}

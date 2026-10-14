//! One native callback owner, including a counted failed teardown guard.

use std::{
    cell::{Cell, RefCell},
    io,
};

use wasm_bindgen::closure::Closure;
use web_sys::MessageEvent;

use super::{
    endpoint::{EventOperation, NativeMessageTarget},
    message,
};

thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static FAILED_CLEANUP: RefCell<Option<ListenerParts>> = const { RefCell::new(None) };
}

struct ListenerParts {
    target: NativeMessageTarget,
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
    pub(super) fn attach<F>(target: NativeMessageTarget, mut callback: F) -> io::Result<Self>
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
        let parts = ListenerParts { target, callback };
        if let Err(registration) = parts
            .target
            .event(EventOperation::Attach, parts.callback.as_ref())
        {
            // A thrown host operation does not prove that it registered nothing.
            // The exact cached inverse must detach before callback release.
            if let Err(cleanup) = parts.detach() {
                retain_failed_cleanup(parts);
                return Err(io::Error::new(
                    registration.kind(),
                    format!("{registration}; registration cleanup failed: {cleanup}"),
                ));
            }
            ACTIVE.with(|active| active.set(false));
            return Err(registration);
        }
        Ok(Self { parts: Some(parts) })
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
        parts.detach()?;
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
                retain_failed_cleanup(parts);
            }
        }
    }
}

impl ListenerParts {
    fn detach(&self) -> io::Result<()> {
        self.target
            .event(EventOperation::Detach, self.callback.as_ref())
    }
}

fn retain_failed_cleanup(parts: ListenerParts) {
    // ACTIVE admits one owner; this guard blocks all subsequent attachments.
    FAILED_CLEANUP.with(|guard| {
        let previous = guard.borrow_mut().replace(parts);
        assert!(previous.is_none(), "single WebView2 failed cleanup owner");
    });
    ACTIVE.with(|active| active.set(false));
}

pub(super) fn retained_cleanup_count() -> usize {
    FAILED_CLEANUP.with(|guard| usize::from(guard.borrow().is_some()))
}

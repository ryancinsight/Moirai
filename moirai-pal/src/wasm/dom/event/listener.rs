//! Callback owners and allocation-free failed teardown retention in this realm.

use std::{cell::RefCell, io};

use wasm_bindgen::{JsCast, closure::Closure};
use web_sys::{Event, EventTarget};

use super::WebEvent;
use crate::wasm::browser_error::javascript_error;

thread_local! {
    static OWNERS: RefCell<Owners> = const { RefCell::new(Owners {
        active: 0, retained: 0, failure_bound: None, head: None,
    }) };
}

struct Owners {
    active: usize,
    retained: usize,
    failure_bound: Option<usize>,
    head: Option<Box<ListenerParts>>,
}

struct ListenerParts {
    target: EventTarget,
    event_name: String,
    // Type erasure is the browser's dynamic event callback ABI (condition 2),
    // not an application execution policy or callback collection.
    closure: Closure<dyn FnMut(Event)>,
    next: Option<Box<ListenerParts>>,
}

/// An owned DOM callback with fallible browser teardown.
///
/// Drop removes the callback before releasing its closure. If removal throws,
/// its pre-owned node moves to a counted realm guard without allocating. New
/// admission stops at the first failure, so retained guards never exceed the
/// finite set of owners admitted before that failure and cannot accumulate
/// across page remount. Realm destruction is the final failed-owner cleanup.
#[must_use = "retain the listener until explicit close or document teardown"]
pub struct WebEventListener {
    parts: Option<Box<ListenerParts>>,
}

impl WebEventListener {
    /// Number of callbacks currently owned by live listener handles in this realm.
    #[must_use]
    pub fn active_count() -> usize {
        OWNERS.with(|owners| owners.borrow().active)
    }

    /// Number of callbacks retained after a failed Drop teardown in this realm.
    #[must_use]
    pub fn retained_cleanup_count() -> usize {
        OWNERS.with(|owners| owners.borrow().retained)
    }

    /// Removes this callback before releasing the Rust closure.
    ///
    /// # Errors
    /// A contextual browser removal exception leaves this handle active; Drop
    /// retains the callback if removal fails again. Successful repeated close
    /// returns without changing counts.
    pub fn close(&mut self) -> io::Result<()> {
        let Some(parts) = self.parts.as_ref() else {
            return Ok(());
        };
        parts
            .target
            .remove_event_listener_with_callback(
                &parts.event_name,
                parts.closure.as_ref().unchecked_ref(),
            )
            .map_err(|error| {
                OWNERS.with(|owners| {
                    let mut owners = owners.borrow_mut();
                    let admitted = owners.active;
                    owners.failure_bound.get_or_insert(admitted);
                });
                javascript_error("DOM removeEventListener", io::ErrorKind::Other, error)
            })?;
        self.parts.take();
        OWNERS.with(|owners| owners.borrow_mut().active -= 1);
        Ok(())
    }
}

pub(in crate::wasm::dom) fn listen<F>(
    target: EventTarget,
    event_name: &str,
    mut callback: F,
) -> io::Result<WebEventListener>
where
    F: FnMut(WebEvent) + 'static,
{
    OWNERS.with(|owners| {
        let mut owners = owners.borrow_mut();
        if owners.failure_bound.is_some() {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "DOM realm retains failed event listener cleanup",
            ));
        }
        owners.active = owners.active.checked_add(1).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::OutOfMemory,
                "DOM event listener owner count overflow",
            )
        })?;
        Ok(())
    })?;
    let closure = Closure::new(move |event: Event| callback(WebEvent { event }));
    // This node exists before browser registration. A later failure transfers
    // this allocation's ownership instead of allocating or forgetting a closure.
    let parts = Box::new(ListenerParts {
        target,
        event_name: event_name.to_owned(),
        closure,
        next: None,
    });
    if let Err(error) = parts
        .target
        .add_event_listener_with_callback(&parts.event_name, parts.closure.as_ref().unchecked_ref())
    {
        OWNERS.with(|owners| owners.borrow_mut().active -= 1);
        return Err(javascript_error(
            "DOM addEventListener",
            io::ErrorKind::InvalidInput,
            error,
        ));
    }
    Ok(WebEventListener { parts: Some(parts) })
}

impl Drop for WebEventListener {
    fn drop(&mut self) {
        if let Err(error) = self.close() {
            tracing::error!(error = %error, "DOM callback teardown failed; retaining the realm owner");
            if let Some(mut parts) = self.parts.take() {
                OWNERS.with(|owners| {
                    let mut owners = owners.borrow_mut();
                    let admitted = owners.active;
                    let bound = *owners.failure_bound.get_or_insert(admitted);
                    owners.active -= 1;
                    owners.retained += 1;
                    assert!(
                        owners.retained <= bound,
                        "failed DOM guards cannot exceed admitted owners"
                    );
                    parts.next = owners.head.take();
                    owners.head = Some(parts);
                });
            }
        }
    }
}

impl Drop for Owners {
    fn drop(&mut self) {
        // Iteration avoids recursive destruction proportional to admitted
        // callback count when the realm destroys its retained ownership list.
        while let Some(mut parts) = self.head.take() {
            self.head = parts.next.take();
        }
    }
}

#[cfg(test)]
mod tests;

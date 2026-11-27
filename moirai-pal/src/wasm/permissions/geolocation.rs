//! Watch ownership is distinct from callback settlement ownership.
//!
//! [Geolocation](https://www.w3.org/TR/geolocation/), sections 6.4–6.7:
//! `clearWatch` removes a watch identifier, but the position
//! request/acquisition algorithms can already have queued callback tasks or be
//! waiting for permission. Two realm-owned callback dispatchers therefore stay
//! valid until realm destruction. Native `bind` attaches a never-reused decimal
//! key; stale callbacks cannot access a newer registration. No script is authored
//! or evaluated. Cancelled records stay bounded by per-document admission until
//! an actual callback; clearing may suppress that callback indefinitely.

mod callbacks;

use super::{
    error,
    registry::Registry,
    request::{Completion, malformed},
};
use crate::permission_state::{Token, callback_key::CallbackKey};
use std::{cell::Cell, rc::Rc, time::Duration};
use wasm_bindgen::JsValue;
use web_sys::{Geolocation, PositionOptions, Window};

#[cfg(test)]
mod tests;

struct Registration {
    geolocation: Geolocation,
    handle: Cell<Option<i32>>,
    registry: Rc<Registry>,
    token: Token,
}

impl Registration {
    fn clear(&self) {
        if let Some(handle) = self.handle.take() {
            self.geolocation.clear_watch(handle);
            self.registry.state.borrow_mut().watch_cleared(self.token);
        }
    }
}

pub(super) struct Watch {
    registration: Rc<Registration>,
}

impl Drop for Watch {
    fn drop(&mut self) {
        self.registration.clear();
        // The settlement record remains: clearWatch does not retract all tasks.
    }
}

pub(super) fn dispatcher_count() -> usize {
    callbacks::dispatcher_count()
}

/// Acquires the navigator's WebIDL geolocation handle without applying an
/// `instanceof` check. Browsers do not expose a `Geolocation` constructor, so
/// web-sys deliberately defines its runtime type test to return false. The
/// typed Navigator getter is the generated binding for this WebIDL attribute.
fn interface(window: &Window) -> Result<Option<Geolocation>, super::WebPermissionError> {
    let geolocation = window
        .navigator()
        .geolocation()
        .map_err(|value| error::native(&value))?;
    let value: &JsValue = geolocation.as_ref();
    Ok((!value.is_null() && !value.is_undefined()).then_some(geolocation))
}

pub(super) fn start(window: &Window, completion: &Rc<Completion>, wait: Duration) -> Option<Watch> {
    let geolocation = match interface(window) {
        Ok(Some(geolocation)) => geolocation,
        Ok(None) => {
            completion.unavailable();
            return None;
        }
        Err(error) => {
            completion.reject(error);
            return None;
        }
    };
    for method in ["watchPosition", "clearWatch"] {
        match error::method(geolocation.as_ref(), method) {
            Ok(true) => (),
            Ok(false) => {
                completion.unavailable();
                return None;
            }
            Err(error) => {
                completion.reject(error);
                return None;
            }
        }
    }
    let milliseconds = match u32::try_from(wait.as_millis()) {
        Ok(value) => value,
        Err(_) => {
            completion.finish(Err(malformed(
                "Admitted geolocation timeout is not representable",
            )));
            return None;
        }
    };
    let registration = Rc::new(Registration {
        geolocation,
        handle: Cell::new(None),
        registry: Rc::clone(&completion.registry),
        token: completion.token,
    });
    let (key, position, rejected) = match callbacks::register(&registration, completion) {
        Ok(callbacks) => callbacks,
        Err(error) => {
            completion.finish(Err(error));
            return None;
        }
    };
    let options = PositionOptions::new();
    options.set_timeout(milliseconds);
    let result = registration
        .geolocation
        .watch_position_with_error_callback_and_options(&position, Some(&rejected), &options);
    match result {
        Ok(handle) => {
            // The standard queues native callbacks; registration returns first.
            registration.handle.set(Some(handle));
            completion
                .registry
                .state
                .borrow_mut()
                .watch_started(completion.token);
            Some(Watch { registration })
        }
        Err(value) => {
            callbacks::registration_failed(key);
            completion.reject(error::native(&value));
            None
        }
    }
}

//! Bounded browser keyboard metadata for owned event listeners.

use super::{PointerModifiers, WebEvent};
use crate::key_validation::bounded_name;
use std::io;
use wasm_bindgen::JsCast;
use web_sys::KeyboardEvent;

/// Metadata captured from one browser keyboard event.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct KeyboardMetadata {
    key: String,
    code: String,
    repeat: bool,
    modifiers: PointerModifiers,
}

impl KeyboardMetadata {
    /// Returns the bounded browser key value, such as `+` or `ArrowDown`.
    #[must_use]
    pub fn key(&self) -> &str {
        &self.key
    }

    /// Returns the bounded physical key code, such as `Equal` or `Minus`.
    #[must_use]
    pub fn code(&self) -> &str {
        &self.code
    }

    /// Returns whether the browser marked this event as an auto-repeat.
    #[must_use]
    pub const fn is_repeat(&self) -> bool {
        self.repeat
    }

    /// Returns the modifier-key snapshot captured with this event.
    #[must_use]
    pub const fn modifiers(&self) -> PointerModifiers {
        self.modifiers
    }
}

impl WebEvent {
    /// Reads bounded keyboard metadata from a browser keyboard event.
    ///
    /// Non-keyboard events return `Ok(None)`. The key and code values are
    /// bounded before the metadata is retained by the consumer.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::InvalidInput`] when either browser name
    /// exceeds the provider bound.
    pub fn keyboard_metadata(&self) -> io::Result<Option<KeyboardMetadata>> {
        let Some(keyboard) = self.event.dyn_ref::<KeyboardEvent>() else {
            return Ok(None);
        };
        Ok(Some(KeyboardMetadata {
            key: bounded_name(keyboard.key())?,
            code: bounded_name(keyboard.code())?,
            repeat: keyboard.repeat(),
            modifiers: keyboard_modifiers(keyboard),
        }))
    }
}

fn keyboard_modifiers(event: &KeyboardEvent) -> PointerModifiers {
    PointerModifiers {
        ctrl: event.ctrl_key(),
        shift: event.shift_key(),
        alt: event.alt_key(),
        meta: event.meta_key(),
    }
}

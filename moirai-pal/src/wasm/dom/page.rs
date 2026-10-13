//! Read-only page observations and owned document events.
use super::{WebDocument, WebElement, WebEvent, WebEventListener, event};
use std::io;
use wasm_bindgen::JsCast;
use web_sys::Url;

/// Parsed browser URL for configuration and destination observations.
#[derive(Clone)]
pub struct WebLocation {
    url: Url,
}
impl WebLocation {
    /// Reads the current document URL.
    /// # Errors
    /// Returns `Unsupported` without a Window, or a contextual URL error.
    pub fn current() -> io::Result<Self> {
        let window = web_sys::window().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                "Browser location requires a Window",
            )
        })?;
        let href = window
            .location()
            .href()
            .map_err(|_| io::Error::other("Browser location is unreadable"))?;
        let url =
            Url::new(&href).map_err(|_| io::Error::other("Browser location URL is invalid"))?;
        Ok(Self { url })
    }
    /// Browser origin serialization, including `null` for opaque origins.
    #[must_use]
    pub fn origin(&self) -> String {
        self.url.origin()
    }
    /// First URLSearchParams value with browser decoding rules.
    /// # Errors
    /// Rejects a value exceeding the caller's byte bound.
    pub fn parameter(&self, name: &str, max_bytes: usize) -> io::Result<Option<String>> {
        let value = self.url.search_params().get(name);
        if value.as_ref().is_some_and(|value| value.len() > max_bytes) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Browser query value exceeds its byte bound",
            ));
        }
        Ok(value)
    }
    /// Resolves a destination without navigating.
    /// # Errors
    /// Returns `InvalidInput` for an invalid destination URL.
    pub fn resolve(&self, destination: &str) -> io::Result<Self> {
        let url = Url::new_with_base(destination, &self.url.href()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "Browser destination URL is invalid",
            )
        })?;
        Ok(Self { url })
    }
}
impl WebDocument {
    /// Registers an owned document callback with fallible explicit close.
    /// Drop attempts removal and retains a counted realm owner if it fails.
    /// # Errors
    /// Returns a contextual event registration refusal or `WouldBlock` when
    /// this realm retains a failed listener cleanup.
    pub fn add_event_listener<F>(
        &self,
        event_name: &str,
        callback: F,
    ) -> io::Result<WebEventListener>
    where
        F: FnMut(WebEvent) + 'static,
    {
        event::listener::listen(self.document.clone().unchecked_into(), event_name, callback)
    }
}
impl WebElement {
    /// Nearest matching ancestor, including this element.
    /// # Errors
    /// Returns `InvalidInput` for an invalid selector.
    pub fn closest(&self, selector: &str) -> io::Result<Option<Self>> {
        self.element
            .closest(selector)
            .map(|value| value.map(|element| Self { element }))
            .map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "Browser closest selector is invalid",
                )
            })
    }
    /// Reads an attribute without resolving URLs or executing content.
    #[must_use]
    pub fn attribute(&self, name: &str) -> Option<String> {
        self.element.get_attribute(name)
    }

    /// Reads the DOM textContent observation without executing markup.
    #[must_use]
    pub fn text_content(&self) -> Option<String> {
        self.element.text_content()
    }
}

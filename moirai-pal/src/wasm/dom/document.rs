//! Browser document acquisition and element access.
use super::{WebClipboard, WebDocument, WebElement, clipboard};
use std::io;
use wasm_bindgen::JsCast;
use web_sys::Window;

impl WebDocument {
    /// Obtains the document for the current browser window.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::Unsupported`] when no browser window or
    /// document is available.
    pub fn current() -> io::Result<Self> {
        let window = web_sys::window().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                "DOM access requires a browser Window",
            )
        })?;
        Self::from_window(window)
    }

    /// Wraps a browser window's document.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::Unsupported`] when the window has no document.
    pub fn from_window(window: Window) -> io::Result<Self> {
        window
            .document()
            .map(|document| Self { document })
            .ok_or_else(|| {
                io::Error::new(io::ErrorKind::Unsupported, "Browser Window has no Document")
            })
    }

    /// Returns the document body.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::NotFound`] before the document body exists.
    pub fn body(&self) -> io::Result<WebElement> {
        self.document
            .body()
            .map(|body| WebElement {
                element: body.unchecked_into(),
            })
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "Document body is absent"))
    }

    /// Finds an element by its stable identifier.
    #[must_use]
    pub fn get_element_by_id(&self, id: &str) -> Option<WebElement> {
        self.document
            .get_element_by_id(id)
            .map(|element| WebElement { element })
    }

    /// Creates an element with the supplied HTML tag name.
    ///
    /// # Errors
    /// Returns the browser's DOM validation error as [`io::ErrorKind::InvalidInput`].
    pub fn create_element(&self, tag: &str) -> io::Result<WebElement> {
        self.document
            .create_element(tag)
            .map(|element| WebElement { element })
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "Invalid DOM element name"))
    }

    /// Returns the browser's text clipboard provider.
    ///
    /// Clipboard access is available only in a secure browser context and
    /// remains subject to the browser's permission and user-activation
    /// policy. The provider never exposes a filesystem or native clipboard
    /// handle to the application.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::Unsupported`] when the current document has no
    /// navigator clipboard API, including an insecure browser context.
    pub fn clipboard(&self) -> io::Result<WebClipboard> {
        clipboard::from_document(self)
    }
}

//! Same-origin browser history for client-side routes.

use std::io;
use wasm_bindgen::JsCast;
use wasm_bindgen::JsValue;
use wasm_bindgen::closure::Closure;
use web_sys::{Event, History, Window};

pub use crate::history_path::MAX_HISTORY_PATH_BYTES;
use crate::history_path::validate;

/// The page's history, limited to paths on the page's own origin.
///
/// Applications read the current path, push or replace entries as their
/// router moves, and hear back/forward navigation through
/// [`WebHistory::on_navigate`]. No method accepts a URL with a scheme or
/// host, so a route can never navigate the page elsewhere.
#[derive(Debug, Clone)]
pub struct WebHistory {
    window: Window,
    history: History,
}

impl WebHistory {
    /// The current window's history.
    ///
    /// # Errors
    /// Returns `Unsupported` outside a browser window.
    pub fn new() -> io::Result<Self> {
        let window = web_sys::window().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                "Browser history requires a Window execution context",
            )
        })?;
        let history = window
            .history()
            .map_err(|_| io::Error::other("Browser history is unavailable"))?;
        Ok(Self { window, history })
    }

    /// The current path, query and fragment, such as `/study/7?tab=info`.
    ///
    /// # Errors
    /// Returns the error when the location cannot be read.
    pub fn path(&self) -> io::Result<String> {
        let location = self.window.location();
        let read = |value: Result<String, JsValue>| {
            value.map_err(|_| io::Error::other("Browser location is unreadable"))
        };
        Ok(format!(
            "{}{}{}",
            read(location.pathname())?,
            read(location.search())?,
            read(location.hash())?
        ))
    }

    /// Adds a history entry for `path` without loading a page.
    ///
    /// # Errors
    /// Returns `InvalidInput` for a path that is not a printable
    /// same-origin path, or the browser's refusal.
    pub fn push(&self, path: &str) -> io::Result<()> {
        validate(path)?;
        self.history
            .push_state_with_url(&JsValue::NULL, "", Some(path))
            .map_err(|_| io::Error::other("Browser refused the history entry"))
    }

    /// Replaces the current history entry with `path`.
    ///
    /// # Errors
    /// As [`Self::push`].
    pub fn replace(&self, path: &str) -> io::Result<()> {
        validate(path)?;
        self.history
            .replace_state_with_url(&JsValue::NULL, "", Some(path))
            .map_err(|_| io::Error::other("Browser refused the history entry"))
    }

    /// Calls `callback` with the new path after each back or forward
    /// navigation, until the returned listener is dropped. Entries this
    /// application pushes or replaces do not call it.
    ///
    /// # Errors
    /// Returns the browser's refusal to register the listener.
    pub fn on_navigate<F>(&self, mut callback: F) -> io::Result<HistoryListener>
    where
        F: FnMut(String) + 'static,
    {
        let history = self.clone();
        let closure = Closure::wrap(Box::new(move |_event: Event| {
            if let Ok(path) = history.path() {
                callback(path);
            }
        }) as Box<dyn FnMut(Event)>);
        self.window
            .add_event_listener_with_callback("popstate", closure.as_ref().unchecked_ref())
            .map_err(|_| io::Error::other("Browser rejected the history listener"))?;
        Ok(HistoryListener {
            window: self.window.clone(),
            closure,
        })
    }
}

/// A registered back/forward listener; dropping it removes the callback
/// before the closure is released.
pub struct HistoryListener {
    window: Window,
    closure: Closure<dyn FnMut(Event)>,
}

impl Drop for HistoryListener {
    fn drop(&mut self) {
        let _ = self
            .window
            .remove_event_listener_with_callback("popstate", self.closure.as_ref().unchecked_ref());
    }
}

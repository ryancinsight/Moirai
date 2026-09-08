//! Thread-owned Win32 window, event and software-frame provider.
//!
//! The provider is deliberately smaller than a GUI toolkit. It owns one HWND,
//! translates the messages addressed to that HWND into bounded Rust values and
//! paints the last validated ARGB frame. Application state, widget policy,
//! authority and WebView hosting stay above this PAL boundary.

mod config;
mod event;
mod input;
mod native;
mod state;

pub use config::{
    MAX_FRAME_DIMENSION, MAX_FRAME_PIXELS, MAX_PUMP_MESSAGES, MAX_TITLE_UNITS,
    MAX_WAIT_MILLISECONDS, MAX_WINDOW_EVENTS, WindowConfig, WindowVisibility,
};
pub use event::{MouseButton, WindowEvent};
pub use native::NativeWindow;

#[cfg(test)]
mod tests;

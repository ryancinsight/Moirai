//! Thread-owned Win32 window, event and software-frame provider.
//!
//! The provider is deliberately smaller than a GUI toolkit. It owns one HWND,
//! translates the messages addressed to that HWND into bounded Rust values and
//! paints the last validated ARGB frame. Application state, widget policy,
//! authority and WebView hosting stay above this PAL boundary.

mod accessibility;
mod config;
mod event;
mod input;
mod native;
mod placement;
mod present;
mod state;

pub use accessibility::{
    AccessibilityAction, AccessibilityActionRequest, AccessibilityNode, AccessibilityRole,
    AccessibilityTree, MAX_ACCESSIBILITY_ACTIONS, MAX_ACCESSIBILITY_NODES,
    MAX_ACCESSIBILITY_TEXT_BYTES,
};
pub use config::{
    MAX_COMPOSITION_UNITS, MAX_FRAME_DIMENSION, MAX_FRAME_PIXELS, MAX_PUMP_MESSAGES,
    MAX_TITLE_UNITS, MAX_WAIT_MILLISECONDS, MAX_WINDOW_EVENTS, WindowConfig, WindowVisibility,
};
pub use event::{CompositionPhase, ModifierState, MouseButton, WindowEvent};
pub use native::NativeWindow;
pub use placement::{MAX_PLACEMENT_COORDINATE, WindowPlacement};
pub use present::FrameRegion;

#[cfg(test)]
mod tests;

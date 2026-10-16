//! Compiled Rust keyboard observations for the installed WebView2 fixture.
//!
//! Default, EnableS and SkipS policy arms load identical producer bytes. The
//! fixture observes document keydown metadata without changing browser input.

#[cfg(target_arch = "wasm32")]
#[path = "webview_keyboard_fixture/browser.rs"]
mod browser;

#[cfg(target_arch = "wasm32")]
#[path = "webview_keyboard_fixture/protocol.rs"]
mod protocol;

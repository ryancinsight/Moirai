//! WASM payload for the installed-runtime native bridge regression.
//!
//! The library example has no native deployment entry point. Its only runtime
//! is the actual packaged browser realm selected by the Windows fixture.

#[cfg(target_arch = "wasm32")]
#[path = "webview_bridge_fixture/browser.rs"]
mod browser;

#[cfg(target_arch = "wasm32")]
#[path = "webview_bridge_fixture/protocol.rs"]
mod protocol;

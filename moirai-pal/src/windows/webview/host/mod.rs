//! WebView2 COM lifetime, policy and event integration.

mod callbacks;
mod com;
mod diagnostics;
mod error;
mod events;
mod focus;
mod frame;
mod hotkey;
mod lifecycle;
mod shutdown;
mod text;
mod tray;
mod view;

#[cfg(test)]
mod keyboard_fixture;

pub use view::WebViewHost;

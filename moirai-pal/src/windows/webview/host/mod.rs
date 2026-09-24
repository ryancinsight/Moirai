//! WebView2 COM lifetime, policy and event integration.

mod callbacks;
mod com;
mod error;
mod hotkey;
mod lifecycle;
mod text;
mod tray;
mod view;

pub use view::WebViewHost;

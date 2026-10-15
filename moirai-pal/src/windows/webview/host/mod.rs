//! WebView2 COM lifetime, policy and event integration.

mod callbacks;
mod com;
mod container;
mod diagnostics;
mod error;
mod focus;
mod frame;
mod hotkey;
mod lifecycle;
mod shutdown;
mod text;
mod tray;
mod view;

pub use view::WebViewHost;

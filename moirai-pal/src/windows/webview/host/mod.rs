//! WebView2 COM lifetime, policy and event integration.

mod callbacks;
mod com;
mod error;
mod lifecycle;
mod text;
mod view;

pub use view::WebViewHost;

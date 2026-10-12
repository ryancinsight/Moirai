//! Apple application lifecycle and bounded packaged WebKit presentation.
//!
//! AppKit and UIKit are host adapters for the same WebView/resource policy.
//! Native objects remain on the main thread. See the provider's ADR 0068.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

mod resource;

pub use resource::{MAX_RESOURCE_BYTES, ResourceRoot};

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod application;
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod webview;

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use application::run;

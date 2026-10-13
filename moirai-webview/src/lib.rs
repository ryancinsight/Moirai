//! Apple application lifecycle and bounded packaged WebKit presentation.
//!
//! AppKit and UIKit are host adapters for the same WebView/resource policy.
//! Native objects remain on the main thread. See the provider's ADR 0068.

#![deny(missing_docs)]
#![deny(unsafe_op_in_unsafe_fn)]

mod navigation;
mod resource;

pub use navigation::{ExternalLinks, MAX_EXTERNAL_LINK_BYTES, MAX_EXTERNAL_LINKS};
pub use resource::{MAX_RESOURCE_BYTES, ResourceRoot};

/// Bundle metadata key containing the host-authorized exact HTTPS links.
///
/// An absent or empty string array grants no external navigation. The Apple
/// host validates this array before starting its WebView.
pub const EXTERNAL_LINKS_INFO_KEY: &str = "MoiraiExternalLinks";

#[cfg(any(target_os = "macos", target_os = "ios"))]
mod application;
#[cfg(any(target_os = "macos", target_os = "ios"))]
mod webview;

#[cfg(any(target_os = "macos", target_os = "ios"))]
pub use application::run;

//! Native lifecycle adapters; resource and DOM behavior remain shared.

use crate::ResourceRoot;
use std::io;

#[cfg(target_os = "macos")]
mod desktop;
#[cfg(target_os = "ios")]
mod mobile;

/// Runs a packaged HTML/CSS/WASM application on the native main thread.
///
/// macOS supplies native application, editing and window menus. Command-Q
/// quits; Command-W closes the key window; Command-M minimizes it. The host
/// terminates after its last window closes. UIKit owns the iPhone/iPad event
/// loop, rotation, keyboard and application activation. WebKit serves only
/// the validated resources; it does not grant native filesystem/process APIs.
/// An explicit title supports development executables; otherwise the title
/// comes from the application's CFBundleDisplayName metadata.
///
/// # Errors
/// Returns `InvalidInput` off the main thread or a native setup/resource error.
pub fn run(resources: ResourceRoot, title: Option<&str>) -> io::Result<()> {
    let mtm = objc2::MainThreadMarker::new().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "Apple application requires the main thread",
        )
    })?;
    let title = match title {
        Some(title) => title.to_owned(),
        None => objc2_foundation::NSBundle::mainBundle()
            .objectForInfoDictionaryKey(&objc2_foundation::NSString::from_str(
                "CFBundleDisplayName",
            ))
            .and_then(|value| value.downcast::<objc2_foundation::NSString>().ok())
            .map(|value| value.to_string())
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "application requires a title or CFBundleDisplayName",
                )
            })?,
    };
    #[cfg(target_os = "macos")]
    return desktop::run(resources, &title, mtm);
    #[cfg(target_os = "ios")]
    mobile::run(resources, &title, mtm)
}

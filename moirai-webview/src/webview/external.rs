//! Host-owned bundle policy and platform URL handoff; no page-native bridge.

use crate::{EXTERNAL_LINKS_INFO_KEY, ExternalLinks, MAX_EXTERNAL_LINK_BYTES, MAX_EXTERNAL_LINKS};
use objc2::{MainThreadMarker, runtime::AnyObject};
use objc2_foundation::{NSArray, NSBundle, NSString, NSURL};
use std::io;

pub(super) fn bundle_links() -> io::Result<ExternalLinks> {
    let value = NSBundle::mainBundle()
        .objectForInfoDictionaryKey(&NSString::from_str(EXTERNAL_LINKS_INFO_KEY));
    metadata_links(value.as_deref())
}

fn metadata_links(value: Option<&AnyObject>) -> io::Result<ExternalLinks> {
    let Some(value) = value else {
        return Ok(ExternalLinks::default());
    };
    let values = value.downcast_ref::<NSArray<AnyObject>>().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "external links metadata must be an array",
        )
    })?;
    if values.len() > MAX_EXTERNAL_LINKS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "external links metadata exceeds its count bound",
        ));
    }
    let destinations = values
        .iter()
        .map(|value| {
            let value = value.downcast::<NSString>().map_err(|_| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "external links metadata contains a non-string",
                )
            })?;
            // NSString measures UTF-16 units. A unit requires at most three
            // UTF-8 bytes, so this check bounds the copy to 3 * the destination
            // byte budget. ExternalLinks then requires ASCII, proving one
            // byte per admitted unit and enforcing the exact byte budget.
            if value.length() > MAX_EXTERNAL_LINK_BYTES {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "external links metadata exceeds its string bound",
                ));
            }
            Ok(value.to_string())
        })
        .collect::<io::Result<Vec<_>>>()?;
    ExternalLinks::new(destinations)
}

#[cfg(test)]
mod tests;

#[cfg(target_os = "macos")]
pub(super) fn open(url: &NSURL, _mtm: MainThreadMarker) -> io::Result<()> {
    let opened = objc2_app_kit::NSWorkspace::sharedWorkspace().openURL(url);
    tracing::info!(opened, "native external link handoff completed");
    if opened {
        Ok(())
    } else {
        Err(io::Error::other(
            "macOS refused the external HTTPS destination",
        ))
    }
}

#[cfg(target_os = "ios")]
pub(super) fn open(url: &NSURL, mtm: MainThreadMarker) -> io::Result<()> {
    use objc2::runtime::Bool;
    use objc2_foundation::NSDictionary;
    use objc2_ui_kit::{UIApplication, UIApplicationOpenExternalURLOptionsKey};
    let completion = block2::RcBlock::new(|opened: Bool| {
        if opened.as_bool() {
            tracing::info!("native external link handoff completed");
        } else {
            tracing::error!("iOS refused the external HTTPS destination");
        }
    });
    let options = NSDictionary::<UIApplicationOpenExternalURLOptionsKey, AnyObject>::new();
    // SAFETY: The dictionary is empty with Apple's exact declared key/value
    // types. UIKit copies the owned completion block; it captures no borrowed
    // action, URL or delegate object and remains valid after this call.
    unsafe {
        UIApplication::sharedApplication(mtm).openURL_options_completionHandler(
            url,
            &options,
            Some(&completion),
        );
    }
    Ok(())
}

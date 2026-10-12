//! Packaged response metadata shared by AppKit and UIKit.
//!
//! A successful resource read must expose status 200, its exact MIME type and
//! byte length to Fetch. wasm-bindgen rejects `Response.ok == false` before
//! instantiating the module, so delivering bytes alone cannot start an app.

use objc2::{AnyThread, rc::Retained};
use objc2_foundation::{NSDictionary, NSHTTPURLResponse, NSString, NSURL, NSURLResponse};
use std::io;

pub(super) fn new(url: &NSURL, mime: &str, length: usize) -> io::Result<Retained<NSURLResponse>> {
    let length = isize::try_from(length)
        .map_err(|error| io::Error::new(io::ErrorKind::InvalidInput, error))?;
    let keys = ["Content-Type", "Content-Length"].map(NSString::from_str);
    let values = [
        NSString::from_str(mime),
        NSString::from_str(&length.to_string()),
    ];
    let headers = NSDictionary::from_slices(
        &keys.each_ref().map(|key| &**key),
        &values.each_ref().map(|value| &**value),
    );
    NSHTTPURLResponse::initWithURL_statusCode_HTTPVersion_headerFields(
        NSHTTPURLResponse::alloc(),
        url,
        200,
        Some(&NSString::from_str("HTTP/1.1")),
        Some(&headers),
    )
    .map(Retained::into_super)
    .ok_or_else(|| io::Error::other("Foundation rejected the packaged resource response"))
}

#[cfg(test)]
mod tests;

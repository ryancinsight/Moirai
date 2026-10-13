//! Convert borrowed WebKit action facts into the portable policy once.

use crate::{
    ExternalLinks, ResourceRoot,
    navigation::{Decision, Frame},
};
use objc2_web_kit::{WKNavigationAction, WKNavigationType};

pub(super) fn decision(links: &ExternalLinks, action: &WKNavigationAction) -> Decision {
    // SAFETY: WebKit owns the action and its frame/request objects throughout
    // each synchronous delegate callback. No borrowed object escapes this call.
    let (request, source, target, activation, download) = unsafe {
        (
            action.request(),
            action.sourceFrame(),
            action.targetFrame(),
            action.navigationType(),
            action.shouldPerformDownload(),
        )
    };
    let destination = request.URL().and_then(|url| url.absoluteString());
    let Some(destination) = destination else {
        return Decision::Deny;
    };
    // SAFETY: Source/target are retained native frames from the live action.
    let (source_uri, source_main, target) = unsafe {
        (
            source
                .as_ref()
                .and_then(|frame| frame.request().URL())
                .and_then(|url| url.absoluteString()),
            source.as_ref().is_some_and(|frame| frame.isMainFrame()),
            target.map_or(Frame::NewWindow, |frame| {
                if frame.isMainFrame() {
                    Frame::Main
                } else {
                    Frame::Subframe
                }
            }),
        )
    };
    let source_uri = source_uri.map(|uri| uri.to_string());
    let decision = links.decide(
        &destination.to_string(),
        source_uri.as_deref(),
        source_main,
        activation == WKNavigationType::LinkActivated,
        download,
        target,
    );
    tracing::info!(
        ?decision,
        ?activation,
        source_main,
        source_packaged = source_uri.as_deref().is_some_and(ResourceRoot::permits),
        download,
        ?target,
        "native navigation policy evaluated"
    );
    decision
}

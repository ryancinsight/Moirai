//! Bounded WebView2 event retention.

use std::{collections::VecDeque, io};

use super::{config::MAX_WEBVIEW_EVENTS, event::WebViewEvent};

pub(super) struct WebViewState {
    events: VecDeque<WebViewEvent>,
    overflowed: bool,
}

impl WebViewState {
    pub(super) fn new() -> io::Result<Self> {
        let mut events = VecDeque::new();
        events
            .try_reserve_exact(MAX_WEBVIEW_EVENTS)
            .map_err(|_| allocation_error())?;
        Ok(Self {
            events,
            overflowed: false,
        })
    }

    pub(super) fn push(&mut self, event: WebViewEvent) {
        if self.events.len() == MAX_WEBVIEW_EVENTS {
            self.overflowed = true;
        } else {
            self.events.push_back(event);
        }
    }

    pub(super) fn drain(&mut self) -> (Vec<WebViewEvent>, bool) {
        let events = self.events.drain(..).collect();
        let overflowed = std::mem::take(&mut self.overflowed);
        (events, overflowed)
    }

    pub(super) fn last_navigation_allowed(&self) -> Option<bool> {
        self.events.iter().rev().find_map(|event| match event {
            WebViewEvent::NavigationStarting { allowed, .. } => Some(*allowed),
            _ => None,
        })
    }
}

fn allocation_error() -> io::Error {
    io::Error::new(
        io::ErrorKind::OutOfMemory,
        "WebView2 event queue reservation failed",
    )
}

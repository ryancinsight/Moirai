//! The failures the host's leaves construct, and the bounded event push
//! that reports a contended state queue as one of them.

use std::{cell::RefCell, io, rc::Rc};

use windows::Win32::Foundation::E_ABORT;

use super::super::event::WebViewEvent;
use super::super::state::WebViewState;

pub(super) fn push_event(
    state: &Rc<RefCell<WebViewState>>,
    event: WebViewEvent,
) -> windows::core::Result<()> {
    let mut state = state
        .try_borrow_mut()
        .map_err(|_| callback_error("WebView2 event state is already borrowed"))?;
    state.push(event);
    Ok(())
}

pub(super) fn windows_error(error: windows::core::Error) -> io::Error {
    io::Error::from_raw_os_error(error.code().0)
}

pub(super) fn callback_error(message: &'static str) -> windows::core::Error {
    windows::core::Error::new(E_ABORT, message)
}

pub(super) fn closed_error() -> io::Error {
    io::Error::new(io::ErrorKind::NotConnected, "WebView2 host is closed")
}

pub(super) fn coordinate_error() -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        "WebView2 bounds exceed Win32 coordinates",
    )
}

//! Finite WebView2 callback waits and owner-thread message dispatch.

use std::{
    io,
    sync::mpsc::Receiver,
    time::{Duration, Instant},
};

use windows::Win32::{
    Foundation::{WAIT_FAILED, WAIT_TIMEOUT},
    UI::WindowsAndMessaging::{
        DispatchMessageW, MSG, MWMO_INPUTAVAILABLE, MsgWaitForMultipleObjectsEx, PM_REMOVE,
        PeekMessageW, QS_ALLINPUT, TranslateMessage,
    },
};

use super::config::MAX_WEBVIEW_WAIT_MILLISECONDS;

pub(super) fn wait_for<T>(
    receiver: Receiver<windows::core::Result<T>>,
    timeout: Duration,
) -> io::Result<T> {
    let milliseconds = u32::try_from(timeout.as_millis()).map_err(|_| timeout_error())?;
    if milliseconds > MAX_WEBVIEW_WAIT_MILLISECONDS {
        return Err(timeout_error());
    }
    let deadline = Instant::now()
        .checked_add(timeout)
        .ok_or_else(timeout_error)?;
    let mut message = MSG::default();
    loop {
        match receiver.try_recv() {
            Ok(result) => return result.map_err(windows_error),
            Err(std::sync::mpsc::TryRecvError::Empty) => {}
            Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                return Err(io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "WebView2 callback channel closed",
                ));
            }
        }
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(timeout_error());
        }
        let milliseconds = u32::try_from(remaining.as_millis())
            .map_err(|_| timeout_error())?
            .min(MAX_WEBVIEW_WAIT_MILLISECONDS);
        // SAFETY: this call observes only the creating thread's message queue,
        // accepts no handles and does not retain a pointer after returning.
        let result = unsafe {
            MsgWaitForMultipleObjectsEx(None, milliseconds, QS_ALLINPUT, MWMO_INPUTAVAILABLE)
        };
        if result == WAIT_FAILED {
            return Err(io::Error::last_os_error());
        }
        if result == WAIT_TIMEOUT {
            return Err(timeout_error());
        }
        dispatch_pending(&mut message);
    }
}

pub(super) fn dispatch_pending(message: &mut MSG) {
    for _ in 0..super::super::window::MAX_PUMP_MESSAGES {
        // SAFETY: `message` is writable storage owned by this call; `None`
        // selects the creating thread's queue and no pointer is retained.
        let present = unsafe { PeekMessageW(message, None, 0, 0, PM_REMOVE) };
        if !present.as_bool() {
            break;
        }
        // SAFETY: `message` was populated by PeekMessageW and is valid for
        // these synchronous dispatch calls.
        unsafe {
            let _ = TranslateMessage(message);
            DispatchMessageW(message);
        }
    }
}

pub(super) fn wait_for_messages(timeout: Duration) -> io::Result<bool> {
    let milliseconds = u32::try_from(timeout.as_millis()).map_err(|_| timeout_error())?;
    if milliseconds > MAX_WEBVIEW_WAIT_MILLISECONDS {
        return Err(timeout_error());
    }
    // SAFETY: this call observes only the creating thread's queue, accepts no
    // handles and retains no pointer after returning.
    let result = unsafe {
        MsgWaitForMultipleObjectsEx(None, milliseconds, QS_ALLINPUT, MWMO_INPUTAVAILABLE)
    };
    if result == WAIT_FAILED {
        return Err(io::Error::last_os_error());
    }
    if result == WAIT_TIMEOUT {
        return Ok(false);
    }
    let mut message = MSG::default();
    dispatch_pending(&mut message);
    Ok(true)
}

fn timeout_error() -> io::Error {
    io::Error::new(
        io::ErrorKind::TimedOut,
        "WebView2 callback did not complete before the finite deadline",
    )
}

fn windows_error(error: windows::core::Error) -> io::Error {
    io::Error::from_raw_os_error(error.code().0)
}

//! Mouse capture ownership for pressed buttons.
//!
//! Without capture a button released outside the client area is delivered to
//! another window, so the pointer-up that closes a drag never reaches the
//! provider. Capture follows the held buttons: it starts on a button-down and
//! ends when a button-up leaves none held.

use std::io;

use windows::Win32::Foundation::{HWND, WPARAM};
use windows::Win32::UI::Input::KeyboardAndMouse::{GetCapture, ReleaseCapture, SetCapture};

use super::super::config::windows_error;
use super::super::input::{CaptureTransition, capture_transition};

/// Applies the capture change that a button message requires.
///
/// Both native calls run window procedures before they return: `SetCapture`
/// sends `WM_CAPTURECHANGED` to the window that loses capture, which can be any
/// window of the thread and can send messages back to this one, and
/// `ReleaseCapture` sends it to the releasing window itself. Those procedures
/// can destroy this window, whose `WM_NCDESTROY` returns the window's count on
/// its state and can be followed by the owner's release. The caller must
/// therefore hold no borrow of this window's state and must keep the state
/// allocation alive through the call with a count of its own.
///
/// # Errors
/// Returns the failure to take or release capture; the caller still reports
/// the button event itself.
pub(super) fn apply(hwnd: HWND, message: u32, wparam: WPARAM) -> io::Result<()> {
    match capture_transition(message, wparam) {
        Some(CaptureTransition::Acquire) => acquire(hwnd),
        Some(CaptureTransition::Release) => release(hwnd),
        None => Ok(()),
    }
}

/// Directs mouse input to `hwnd`, a window of the calling thread.
///
/// A window that already owns the capture is left alone: `SetCapture` would
/// send `WM_CAPTURECHANGED` to that same window, an avoidable re-entry. A
/// window of another thread cannot be captured; the call then fails with no
/// effect. [`apply`]'s caller holds no borrow of any state a window
/// procedure of this thread can reach and keeps that state alive, because the
/// previous capture owner's procedure runs before this returns and can run a
/// nested destruction.
fn acquire(hwnd: HWND) -> io::Result<()> {
    // SAFETY: `GetCapture` reads the calling thread's input state only.
    if unsafe { GetCapture() } == hwnd {
        return Ok(());
    }
    // SAFETY: `SetCapture` takes a window handle by value, and a handle that
    // does not name a window of this thread makes it fail without effect.
    // It synchronously runs the procedure of the window losing capture on this
    // thread, which can re-enter this window and destroy it. `apply`'s caller
    // holds no state borrow and keeps the state allocation alive through the
    // call by a counted clone, so nested destruction cannot free it.
    unsafe { SetCapture(hwnd) };
    // `SetCapture` returns the previous owner and has no failure value, so
    // ownership is confirmed by reading it back.
    // SAFETY: `GetCapture` reads the calling thread's input state only.
    if unsafe { GetCapture() } == hwnd {
        Ok(())
    } else {
        Err(io::Error::other(
            "native window could not capture the mouse",
        ))
    }
}

/// Ends capture when `hwnd` holds it; a capture owned by another window of the
/// thread is left alone.
fn release(hwnd: HWND) -> io::Result<()> {
    // SAFETY: `GetCapture` reads the calling thread's input state only.
    if unsafe { GetCapture() } != hwnd {
        return Ok(());
    }
    // SAFETY: the check above shows the calling thread's capture belongs to
    // `hwnd`. `ReleaseCapture` synchronously sends `WM_CAPTURECHANGED` to
    // `hwnd`, whose procedure can run a nested destruction. `apply`'s caller
    // holds no borrow of this window's state and keeps its allocation alive
    // through the call by a counted clone.
    unsafe { ReleaseCapture() }.map_err(windows_error)
}

#[cfg(test)]
mod tests {
    use super::super::super::{WindowConfig, WindowVisibility};
    use super::super::NativeWindow;
    use super::*;
    use std::sync::mpsc;

    fn hidden_window() -> NativeWindow {
        let config = WindowConfig::with_visibility(
            "Moirai foreign capture",
            64,
            64,
            WindowVisibility::Hidden,
        )
        .expect("config");
        NativeWindow::new(&config).expect("native window")
    }

    #[test]
    fn capturing_a_window_of_another_thread_is_a_typed_failure() {
        let (hwnd_sender, hwnd_receiver) = mpsc::sync_channel::<usize>(1);
        let (done_sender, done_receiver) = mpsc::sync_channel::<()>(1);
        let owner = std::thread::spawn(move || {
            let mut window = hidden_window();
            hwnd_sender
                .send(window.hwnd.0.expose_provenance())
                .expect("the test thread is waiting");
            done_receiver
                .recv()
                .expect("the test thread signals completion");
            window.close().expect("destroy");
        });
        let address = hwnd_receiver
            .recv()
            .expect("the owner publishes its window");
        let foreign = HWND(std::ptr::with_exposed_provenance_mut(address));

        let error = acquire(foreign).expect_err("capture is thread-affine");
        assert_eq!(error.kind(), io::ErrorKind::Other);
        assert_eq!(
            error.to_string(),
            "native window could not capture the mouse"
        );
        // SAFETY: `GetCapture` reads the calling thread's input state only.
        assert_ne!(unsafe { GetCapture() }, foreign);
        release(foreign).expect("releasing a capture this window never held is a no-op");

        done_sender.send(()).expect("the owner is waiting");
        owner.join().expect("owner thread completes");
    }
}

//! Synchronous focus delegation to the one native content owner.
//!
//! The window owns only a Weak reference. Its thread-owned content owner must
//! revoke the binding before closing native resources. A WM_SETFOCUS handler
//! upgrades under the state borrow, releases that borrow, then calls the owner:
//! reentrant WM_KILLFOCUS therefore finds the state unborrowed. Queuing the
//! operation as a value event would allow later keyboard messages to precede it.

use std::{io, rc::Weak};

use windows::Win32::UI::Input::KeyboardAndMouse::GetFocus;
use windows::Win32::UI::WindowsAndMessaging::GetForegroundWindow;

use super::{NativeWindow, state::SharedWindowState};

/// The runtime content owner varies at this non-throughput-critical native
/// callback boundary; type erasure keeps the window independent of WebView2.
pub(crate) trait NativeFocusTarget {
    fn focus(&self) -> io::Result<()>;
}

impl NativeWindow {
    /// Binds the sole content owner and handles focus predating its creation.
    /// Only exact parent focus admits that initial transfer; a different native
    /// recipient, including a different application, remains untouched.
    pub(crate) fn bind_focus_target(
        &mut self,
        target: Weak<dyn NativeFocusTarget>,
    ) -> io::Result<()> {
        if self.is_destroyed() {
            return Err(io::Error::new(
                io::ErrorKind::NotConnected,
                "native focus target requires a live window",
            ));
        }
        self.state.with(|state| state.focus_target = Some(target));
        // SAFETY: GetFocus is an immediate owner-thread observation.
        if unsafe { GetFocus() } == self.hwnd && unsafe { GetForegroundWindow() } == self.hwnd {
            let target = self
                .state
                .with(|state| state.focus_target.as_ref().and_then(Weak::upgrade));
            if let Some(target) = target {
                target.focus()?;
            }
        }
        Ok(())
    }

    /// Revokes admission before content shutdown and preserves a callback error
    /// even when the caller closes without another native event poll.
    pub(crate) fn clear_focus_target(&mut self) -> io::Result<()> {
        self.state.with(|state| {
            state.focus_target = None;
            state.error.take().map_or(Ok(()), Err)
        })
    }
}

pub(super) fn forward(state: &SharedWindowState) {
    let target = state.with(|state| state.focus_target.as_ref().and_then(Weak::upgrade));
    if let Some(target) = target
        && let Err(error) = target.focus()
    {
        state.with(|state| state.record_error(error));
    }
}

//! Default-off, read-only controller input callbacks with deferred tracing.
//!
//! Accelerator callbacks synchronously block the windowed browser. They only
//! read supplied arguments and user32 state, enqueue Copy records, then return.
//! No Handled setter, focus operation, tracing, JSON, disk I/O or outgoing COM
//! request occurs inside a callback. Unicode packets are not all accelerators;
//! absence of such a callback never establishes lack of renderer consumption.

mod records;

use super::super::error::{callback_context, callback_error};
use super::state::Shared;
use records::{Event, Queue};
use std::{io, rc::Rc};
use webview2_com::{
    AcceleratorKeyPressedEventHandler, FocusChangedEventHandler,
    Microsoft::Web::WebView2::Win32::{
        COREWEBVIEW2_KEY_EVENT_KIND, COREWEBVIEW2_MOVE_FOCUS_REASON,
        ICoreWebView2AcceleratorKeyPressedEventHandler, ICoreWebView2Controller,
        ICoreWebView2FocusChangedEventHandler, ICoreWebView2MoveFocusRequestedEventHandler,
    },
    MoveFocusRequestedEventHandler,
};
use windows::core::BOOL;

pub(super) struct Observer {
    controller: ICoreWebView2Controller,
    queue: Rc<Queue>,
    state: Shared,
    gained: Option<(i64, ICoreWebView2FocusChangedEventHandler)>,
    lost: Option<(i64, ICoreWebView2FocusChangedEventHandler)>,
    moved: Option<(i64, ICoreWebView2MoveFocusRequestedEventHandler)>,
    key: Option<(i64, ICoreWebView2AcceleratorKeyPressedEventHandler)>,
}

impl Observer {
    #[cfg(test)]
    pub(super) fn checkpoint(&self) -> (std::time::Instant, u64) {
        self.queue.checkpoint()
    }

    pub(super) fn new(controller: &ICoreWebView2Controller, state: &Shared) -> io::Result<Self> {
        let mut observer = Self {
            controller: controller.clone(),
            queue: Rc::new(Queue::new()),
            state: Rc::clone(state),
            gained: None,
            lost: None,
            moved: None,
            key: None,
        };
        observer.install().map_err(io::Error::other)?;
        Ok(observer)
    }

    fn install(&mut self) -> windows::core::Result<()> {
        let queue = Rc::downgrade(&self.queue);
        // Type erasure is required by the SDK's runtime callback ABI.
        let handler = FocusChangedEventHandler::create(Box::new(move |_, _| {
            if let Some(queue) = queue.upgrade() {
                queue.push(Event::Focus { gained: true });
            }
            Ok(())
        }));
        let mut token = 0;
        // SAFETY: live controller retains the callback; this owner retains its
        // token until removal before controller close, including partial setup.
        unsafe {
            self.controller.add_GotFocus(&handler, &mut token)?;
        }
        self.gained = Some((token, handler));
        let queue = Rc::downgrade(&self.queue);
        // Type erasure is required by the SDK's runtime callback ABI.
        let handler = FocusChangedEventHandler::create(Box::new(move |_, _| {
            if let Some(queue) = queue.upgrade() {
                queue.push(Event::Focus { gained: false });
            }
            Ok(())
        }));
        // SAFETY: callback and token have the ownership described above.
        unsafe {
            self.controller.add_LostFocus(&handler, &mut token)?;
        }
        self.lost = Some((token, handler));
        let queue = Rc::downgrade(&self.queue);
        // Type erasure is required by the SDK's runtime callback ABI.
        let handler = MoveFocusRequestedEventHandler::create(Box::new(move |_, args| {
            if let Some(queue) = queue.upgrade() {
                let result = (|| {
                    let args =
                        args.ok_or_else(|| callback_error("MoveFocusRequested arguments missing"))?;
                    let mut reason = COREWEBVIEW2_MOVE_FOCUS_REASON(0);
                    let mut handled = BOOL(0);
                    // SAFETY: SDK arguments and writable outputs live through
                    // this synchronous callback. No property is modified.
                    unsafe {
                        args.Reason(&mut reason)?;
                        args.Handled(&mut handled)?;
                    }
                    Ok(Event::Move {
                        reason: reason.0,
                        handled: handled.as_bool(),
                    })
                })();
                queue.push(
                    result.unwrap_or_else(|error: windows::core::Error| Event::Error {
                        operation: "MoveFocusRequested",
                        hresult: error.code().0,
                    }),
                );
            }
            Ok(())
        }));
        // SAFETY: callback and token have the ownership described above.
        unsafe {
            self.controller
                .add_MoveFocusRequested(&handler, &mut token)?;
        }
        self.moved = Some((token, handler));
        let queue = Rc::downgrade(&self.queue);
        // Type erasure is required by the SDK's runtime callback ABI.
        let handler = AcceleratorKeyPressedEventHandler::create(Box::new(move |_, args| {
            if let Some(queue) = queue.upgrade() {
                let result = (|| {
                    let args = args
                        .ok_or_else(|| callback_error("AcceleratorKeyPressed arguments missing"))?;
                    let mut kind = COREWEBVIEW2_KEY_EVENT_KIND(0);
                    let mut key = 0;
                    let mut lparam = 0;
                    let mut handled = BOOL(0);
                    // SAFETY: only immediate SDK argument getters run while
                    // the windowed browser is synchronously blocked here.
                    unsafe {
                        args.KeyEventKind(&mut kind)?;
                        args.VirtualKey(&mut key)?;
                        args.KeyEventLParam(&mut lparam)?;
                        args.Handled(&mut handled)?;
                    }
                    Ok(Event::Key {
                        kind: kind.0,
                        key,
                        lparam,
                        handled: handled.as_bool(),
                    })
                })();
                queue.push(
                    result.unwrap_or_else(|error: windows::core::Error| Event::Error {
                        operation: "AcceleratorKeyPressed",
                        hresult: error.code().0,
                    }),
                );
            }
            Ok(())
        }));
        // SAFETY: callback and token have the ownership described above.
        unsafe {
            self.controller
                .add_AcceleratorKeyPressed(&handler, &mut token)?;
        }
        self.key = Some((token, handler));
        Ok(())
    }

    pub(super) fn flush(&self) {
        self.queue.flush(&self.state);
    }

    pub(super) fn registered(&self) -> usize {
        [
            self.gained.is_some(),
            self.lost.is_some(),
            self.moved.is_some(),
            self.key.is_some(),
        ]
        .into_iter()
        .filter(|present| *present)
        .count()
    }

    pub(super) fn close(&mut self) -> io::Result<()> {
        if self.queue.closed.replace(true) {
            return Ok(());
        }
        self.flush();
        let mut first = None;
        if let Some((token, _)) = self.gained.take() {
            // SAFETY: token belongs to this live retained controller.
            self.removed(
                "remove_GotFocus",
                unsafe { self.controller.remove_GotFocus(token) },
                &mut first,
            );
        }
        if let Some((token, _)) = self.lost.take() {
            // SAFETY: token belongs to this live retained controller.
            self.removed(
                "remove_LostFocus",
                unsafe { self.controller.remove_LostFocus(token) },
                &mut first,
            );
        }
        if let Some((token, _)) = self.moved.take() {
            // SAFETY: token belongs to this live retained controller.
            self.removed(
                "remove_MoveFocusRequested",
                unsafe { self.controller.remove_MoveFocusRequested(token) },
                &mut first,
            );
        }
        if let Some((token, _)) = self.key.take() {
            // SAFETY: token belongs to this live retained controller.
            self.removed(
                "remove_AcceleratorKeyPressed",
                unsafe { self.controller.remove_AcceleratorKeyPressed(token) },
                &mut first,
            );
        }
        first.map_or(Ok(()), Err)
    }

    fn removed(
        &self,
        operation: &str,
        result: windows::core::Result<()>,
        first: &mut Option<io::Error>,
    ) {
        if let Err(error) = result {
            let mut state = self.state.borrow_mut();
            state.cleanup_failed = true;
            state.error(operation, &error);
            first.get_or_insert_with(|| io::Error::other(callback_context(operation, error)));
        }
    }
}

impl Drop for Observer {
    fn drop(&mut self) {
        // close records all removal failures before returning the first one.
        let _ = self.close();
    }
}

#[cfg(test)]
mod tests;

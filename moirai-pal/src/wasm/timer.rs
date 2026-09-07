//! Cancellation-safe browser timers.

use js_sys::Promise;
use std::future::Future;
use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};
use std::time::Duration;
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{console, Window};

// JavaScript's setTimeout accepts a signed 32-bit millisecond delay.
const MAX_TIMEOUT_MILLISECONDS: u128 = 2_147_483_647;

/// A one-shot JavaScript timer that releases its callback on cancellation.
#[must_use = "poll the timer future to observe its deadline"]
pub struct WebTimer {
    future: Pin<Box<JsFuture>>,
    window: Window,
    handle: i32,
    callback: Closure<dyn FnMut()>,
}

impl WebTimer {
    /// Starts a timer for the supplied finite duration.
    ///
    /// Durations longer than the JavaScript signed timeout range are clamped
    /// to that range. Dropping the timer clears the scheduled callback.
    ///
    /// # Errors
    /// Returns an error when the current execution context has no browser
    /// window or the timeout cannot be registered.
    pub fn new(duration: Duration) -> io::Result<Self> {
        let window = web_sys::window().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                "Browser timer requires a Window execution context",
            )
        })?;
        let milliseconds = duration.as_millis().min(MAX_TIMEOUT_MILLISECONDS);
        let milliseconds = i32::try_from(milliseconds).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "Browser timer duration exceeds the supported timeout range",
            )
        })?;

        let mut resolve_function = None;
        let promise = Promise::new(&mut |resolve, _reject| {
            resolve_function = Some(resolve);
        });
        let resolve_function = resolve_function
            .ok_or_else(|| io::Error::other("Browser timer promise did not expose a resolver"))?;
        let callback = Closure::wrap(Box::new(move || {
            if let Err(error) = resolve_function.call0(&JsValue::UNDEFINED) {
                console::error_1(&error);
            }
        }) as Box<dyn FnMut()>);
        let handle = window
            .set_timeout_with_callback_and_timeout_and_arguments_0(
                callback.as_ref().unchecked_ref(),
                milliseconds,
            )
            .map_err(|_| {
                io::Error::other("Browser execution context rejected the timer callback")
            })?;

        Ok(Self {
            future: Box::pin(JsFuture::from(promise)),
            window,
            handle,
            callback,
        })
    }
}

impl Future for WebTimer {
    type Output = io::Result<()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        match this.future.as_mut().poll(cx) {
            Poll::Ready(Ok(_)) => Poll::Ready(Ok(())),
            Poll::Ready(Err(_)) => {
                Poll::Ready(Err(io::Error::other("Browser timer promise was rejected")))
            }
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Drop for WebTimer {
    fn drop(&mut self) {
        self.window.clear_timeout_with_handle(self.handle);
        let _ = &self.callback;
    }
}

//! Cancellation-safe browser animation-frame scheduling.

use js_sys::Promise;
use std::future::Future;
use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};
use wasm_bindgen::closure::Closure;
use wasm_bindgen::{JsCast, JsValue};
use wasm_bindgen_futures::JsFuture;
use web_sys::{Window, console};

/// A one-shot `requestAnimationFrame` callback that releases its browser
/// registration when the future is dropped.
#[must_use = "poll the animation-frame future to receive the frame timestamp"]
pub struct WebAnimationFrame {
    future: Pin<Box<JsFuture>>,
    window: Window,
    handle: i32,
    callback: Closure<dyn FnMut(f64)>,
}

impl WebAnimationFrame {
    /// Schedules one callback at the browser's next rendering opportunity.
    ///
    /// The returned future resolves with the browser's high-resolution frame
    /// timestamp in milliseconds. Dropping it cancels the callback before the
    /// JavaScript closure is released.
    ///
    /// # Errors
    /// Returns [`io::ErrorKind::Unsupported`] when no browser window exists,
    /// or [`io::ErrorKind::Other`] when the browser rejects the callback.
    pub fn new() -> io::Result<Self> {
        let window = web_sys::window().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Unsupported,
                "Browser animation frame requires a Window execution context",
            )
        })?;

        let mut resolve_function = None;
        let promise = Promise::new(&mut |resolve, _reject| {
            resolve_function = Some(resolve);
        });
        let resolve_function = resolve_function.ok_or_else(|| {
            io::Error::other("Browser animation frame promise did not expose a resolver")
        })?;
        let callback = Closure::wrap(Box::new(move |timestamp: f64| {
            if let Err(error) =
                resolve_function.call1(&JsValue::UNDEFINED, &JsValue::from_f64(timestamp))
            {
                console::error_1(&error);
            }
        }) as Box<dyn FnMut(f64)>);
        let handle = window
            .request_animation_frame(callback.as_ref().unchecked_ref())
            .map_err(|_| {
                io::Error::other("Browser execution context rejected the animation callback")
            })?;

        Ok(Self {
            future: Box::pin(JsFuture::from(promise)),
            window,
            handle,
            callback,
        })
    }
}

impl Future for WebAnimationFrame {
    type Output = io::Result<f64>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        match this.future.as_mut().poll(cx) {
            Poll::Ready(Ok(value)) => {
                let timestamp = value.as_f64().ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Browser animation frame returned a non-numeric timestamp",
                    )
                });
                Poll::Ready(timestamp.and_then(|timestamp| {
                    if timestamp.is_finite() {
                        Ok(timestamp)
                    } else {
                        Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            "Browser animation frame returned a non-finite timestamp",
                        ))
                    }
                }))
            }
            Poll::Ready(Err(_)) => Poll::Ready(Err(io::Error::other(
                "Browser animation frame promise was rejected",
            ))),
            Poll::Pending => Poll::Pending,
        }
    }
}

impl Drop for WebAnimationFrame {
    fn drop(&mut self) {
        let _ = self.window.cancel_animation_frame(self.handle);
        let _ = &self.callback;
    }
}

//! Fetch with bounded streaming storage and owned cancellation.
use super::{WebTimer, stream};
use crate::http_policy::{content_length, validate_limits};
use std::{
    future::{Future, poll_fn},
    io,
    pin::pin,
    task::Poll,
    time::Duration,
};
use wasm_bindgen::JsCast;
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    AbortController, Request, RequestCache, RequestCredentials, RequestInit, RequestRedirect,
    Response, Url,
};

/// Largest HTTP request or response, matching the browser read budget.
pub const MAX_HTTP_BYTES: usize = crate::http_policy::MAX_HTTP_BYTES;
/// Largest admitted whole-request deadline.
pub const MAX_HTTP_DEADLINE: Duration = crate::http_policy::MAX_HTTP_DEADLINE;
/// An admitted browser HTTP method.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WebHttpMethod {
    /// Read a resource without a request body.
    Get,
    /// Submit a binary request body.
    Post,
}
/// Response whose complete streamed body fits the requested budget.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WebHttpResponse {
    /// Actual HTTP status; non-success responses retain their body.
    pub status: u16,
    /// Complete bounded response bytes.
    pub body: Vec<u8>,
}
/// Browser HTTP mechanisms; application authority policy stays above.
pub struct WebHttp;
impl WebHttp {
    /// Sends a credential-free uncached request without following redirects.
    ///
    /// One deadline covers headers and body. Future drop aborts Fetch, cancels
    /// the pending reader and releases its lock. Declared and actual body
    /// lengths are independently bounded.
    /// # Errors
    /// Returns `InvalidInput` for URL/credential/header/budget/GET-body errors,
    /// `TimedOut` on expiry, `InvalidData` for malformed/oversized responses,
    /// or a contextual browser transport refusal.
    pub async fn request(
        method: WebHttpMethod,
        url: &str,
        content_type: Option<&str>,
        body: &[u8],
        response_bound: usize,
        deadline: Duration,
    ) -> io::Result<WebHttpResponse> {
        validate_limits(body.len(), response_bound, deadline)?;
        let parsed = Url::new(url).map_err(|_| invalid("Browser HTTP URL is invalid"))?;
        if !matches!(parsed.protocol().as_str(), "http:" | "https:")
            || !parsed.username().is_empty()
            || !parsed.password().is_empty()
        {
            return Err(invalid(
                "Browser HTTP URL must use HTTP(S) without credentials",
            ));
        }
        if method == WebHttpMethod::Get && !body.is_empty() {
            return Err(invalid("Browser HTTP GET cannot contain a body"));
        }
        let controller = AbortController::new()
            .map_err(|_| io::Error::other("Browser HTTP cancellation is unavailable"))?;
        let _abort = AbortOnDrop(controller.clone());
        let init = RequestInit::new();
        init.set_method(match method {
            WebHttpMethod::Get => "GET",
            WebHttpMethod::Post => "POST",
        });
        init.set_cache(RequestCache::NoStore);
        init.set_credentials(RequestCredentials::Omit);
        init.set_redirect(RequestRedirect::Error);
        init.set_signal(Some(&controller.signal()));
        if method == WebHttpMethod::Post {
            init.set_body(&js_sys::Uint8Array::from(body));
        }
        let request = Request::new_with_str_and_init(url, &init)
            .map_err(|_| invalid("Browser refused HTTP request construction"))?;
        if let Some(value) = content_type {
            request
                .headers()
                .set("Content-Type", value)
                .map_err(|_| invalid("Browser HTTP Content-Type is invalid"))?;
        }
        let window = web_sys::window().ok_or_else(|| {
            io::Error::new(io::ErrorKind::Unsupported, "Browser HTTP requires a Window")
        })?;
        let operation = async {
            let response = JsFuture::from(window.fetch_with_request(&request))
                .await
                .map_err(|_| {
                    io::Error::other("Browser HTTP Fetch failed or was denied by origin/CSP policy")
                })?
                .dyn_into::<Response>()
                .map_err(|_| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "Browser Fetch returned an invalid response",
                    )
                })?;
            if let Some(value) = response
                .headers()
                .get("content-length")
                .map_err(|_| io::Error::other("Browser HTTP headers are unreadable"))?
            {
                content_length(&value, response_bound)?;
            }
            let stream = response.body().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    "Browser HTTP response has no body stream",
                )
            })?;
            let mut sink = stream::VecSink {
                bytes: Vec::new(),
                bound: response_bound,
            };
            stream::read(stream, &mut sink).await?;
            Ok(WebHttpResponse {
                status: response.status(),
                body: sink.bytes,
            })
        };
        let mut operation = pin!(operation);
        let mut timer = pin!(WebTimer::new(deadline)?);
        poll_fn(|context| {
            if let Poll::Ready(result) = timer.as_mut().poll(context) {
                return Poll::Ready(result.and_then(|()| {
                    Err(io::Error::new(
                        io::ErrorKind::TimedOut,
                        "Browser HTTP deadline expired",
                    ))
                }));
            }
            operation.as_mut().poll(context)
        })
        .await
    }
}
fn invalid(message: &'static str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message)
}
struct AbortOnDrop(AbortController);
impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

use super::{
    WebPermissionKind, WebPermissionOutcome, error, geolocation, media, notification,
    registry::Registry,
};
use crate::{
    permission_state::{Delivery, Token, deadline},
    wasm::WebTimer,
};
use std::{
    cell::RefCell,
    future::Future,
    io,
    pin::Pin,
    rc::Rc,
    task::{Context, Poll, Waker},
    time::Duration,
};
use web_sys::Window;

pub(super) struct Completion {
    pub(super) token: Token,
    pub(super) registry: Rc<Registry>,
    result: RefCell<Option<io::Result<WebPermissionOutcome>>>,
    waker: RefCell<Option<Waker>>,
}

impl Completion {
    pub(super) fn finish(&self, result: io::Result<WebPermissionOutcome>) {
        let delivery = self.registry.state.borrow_mut().settle(self.token);
        if delivery == Delivery::Publish {
            *self.result.borrow_mut() = Some(result);
            let waker = self.waker.borrow_mut().take();
            if let Some(waker) = waker {
                waker.wake();
            }
        }
        // Discard drops a late MediaGranted lease and stops its tracks.
    }

    pub(super) fn reject(&self, error: super::WebPermissionError) {
        self.finish(Ok(WebPermissionOutcome::Rejected {
            kind: self.token.kind,
            error,
        }));
    }

    pub(super) fn unavailable(&self) {
        self.finish(Ok(WebPermissionOutcome::Unavailable(self.token.kind)));
    }
}

/// Owned request future; dropping it cancels publication and releases a watch.
/// Browser requests retain bounded settlement state until an actual callback or
/// Promise settlement. A cleared geolocation watch may not callback again; its
/// slot then stays busy until document destruction.
#[must_use = "retain and poll the request; drop cancels publication"]
pub struct WebPermissionRequest {
    completion: Rc<Completion>,
    timer: Option<WebTimer>,
    watch: Option<geolocation::Watch>,
    finished: bool,
}

pub(super) fn start(
    window: &Window,
    registry: Rc<Registry>,
    kind: WebPermissionKind,
    wait: Duration,
) -> io::Result<WebPermissionRequest> {
    let wait = deadline(wait)?;
    let timer = WebTimer::new(wait)?;
    let token = registry.state.borrow_mut().begin(kind)?;
    let completion = Rc::new(Completion {
        token,
        registry,
        result: RefCell::new(None),
        waker: RefCell::new(None),
    });
    let mut request = WebPermissionRequest {
        completion: Rc::clone(&completion),
        timer: Some(timer),
        watch: None,
        finished: false,
    };
    match kind {
        WebPermissionKind::Geolocation => {
            request.watch = geolocation::start(window, &completion, wait)
        }
        WebPermissionKind::Camera | WebPermissionKind::Microphone => {
            media::start(window, Rc::clone(&completion))
        }
        WebPermissionKind::Notifications => notification::start(window, Rc::clone(&completion)),
    }
    Ok(request)
}

impl WebPermissionRequest {
    fn release_consumer(&mut self, consumed: bool) {
        if let Some(watch) = self.watch.take() {
            drop(watch);
        }
        if consumed {
            self.completion
                .registry
                .state
                .borrow_mut()
                .consumed(self.completion.token);
        } else {
            self.completion
                .registry
                .state
                .borrow_mut()
                .cancel(self.completion.token);
        }
        self.completion.waker.borrow_mut().take();
        self.timer.take();
        self.finished = true;
    }
}

impl Future for WebPermissionRequest {
    type Output = io::Result<WebPermissionOutcome>;

    fn poll(self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        assert!(!this.finished, "Permission request polled after completion");
        let result = this.completion.result.borrow_mut().take();
        if let Some(result) = result {
            this.release_consumer(true);
            return Poll::Ready(result);
        }
        if let Some(timer) = this.timer.as_mut()
            && let Poll::Ready(result) = Pin::new(timer).poll(context)
        {
            let result = result.and_then(|()| {
                Err(io::Error::new(
                    io::ErrorKind::TimedOut,
                    format!(
                        "Browser {:?} permission deadline expired",
                        this.completion.token.kind
                    ),
                ))
            });
            this.release_consumer(false);
            return Poll::Ready(result);
        }
        *this.completion.waker.borrow_mut() = Some(context.waker().clone());
        Poll::Pending
    }
}

impl Drop for WebPermissionRequest {
    fn drop(&mut self) {
        if !self.finished {
            self.release_consumer(false);
        }
        // A completed result not consumed by the caller may own a stream.
        self.completion.result.borrow_mut().take();
    }
}

pub(super) fn malformed(context: &'static str) -> io::Error {
    error::invalid(context)
}

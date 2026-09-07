//! Browser WebSocket receive state shared by the WebAssembly PAL and tests.

use std::collections::VecDeque;
use std::future::Future;
use std::io;
use std::pin::Pin;
use std::sync::{Arc, Mutex, MutexGuard};
use std::task::{Context, Poll, Waker};

/// Default maximum size of one browser WebSocket message.
pub const DEFAULT_MAX_MESSAGE_BYTES: usize = 8 * 1024 * 1024;

/// Default maximum number of messages retained per browser WebSocket.
pub const DEFAULT_MAX_QUEUED_MESSAGES: usize = 64;

/// Limits applied before browser data is copied into Rust-owned memory.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WebSocketLimits {
    max_message_bytes: usize,
    max_queued_messages: usize,
}

impl WebSocketLimits {
    /// Construct limits with non-zero message and queue bounds.
    ///
    /// # Errors
    ///
    /// Returns [`io::ErrorKind::InvalidInput`] when either bound is zero.
    pub fn new(max_message_bytes: usize, max_queued_messages: usize) -> io::Result<Self> {
        if max_message_bytes == 0 || max_queued_messages == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "WebSocket limits must be non-zero",
            ));
        }

        Ok(Self {
            max_message_bytes,
            max_queued_messages,
        })
    }

    /// Return the maximum number of bytes accepted in one message.
    #[must_use]
    pub const fn max_message_bytes(self) -> usize {
        self.max_message_bytes
    }

    /// Return the maximum number of messages retained for one connection.
    #[must_use]
    pub const fn max_queued_messages(self) -> usize {
        self.max_queued_messages
    }
}

impl Default for WebSocketLimits {
    fn default() -> Self {
        Self {
            max_message_bytes: DEFAULT_MAX_MESSAGE_BYTES,
            max_queued_messages: DEFAULT_MAX_QUEUED_MESSAGES,
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum WebSocketStatus {
    Connecting,
    Open,
    Closed {
        code: u16,
    },
    Failed {
        kind: io::ErrorKind,
        message: &'static str,
    },
}

#[derive(Debug)]
pub(crate) struct WebSocketState {
    status: WebSocketStatus,
    incoming: VecDeque<Vec<u8>>,
    waiter: Option<Waker>,
    #[cfg(any(target_arch = "wasm32", test))]
    open_waiter: Option<Waker>,
    limits: WebSocketLimits,
}

impl WebSocketState {
    pub(crate) fn new(limits: WebSocketLimits) -> Self {
        Self {
            status: WebSocketStatus::Connecting,
            // Defer allocation until a message arrives. The caller controls
            // the queue bound, so eagerly reserving that entire bound would
            // turn a valid policy value into an unbounded allocation request.
            incoming: VecDeque::new(),
            waiter: None,
            #[cfg(any(target_arch = "wasm32", test))]
            open_waiter: None,
            limits,
        }
    }

    pub(crate) fn open(&mut self) -> bool {
        if matches!(self.status, WebSocketStatus::Connecting) {
            self.status = WebSocketStatus::Open;
            true
        } else {
            false
        }
    }

    #[cfg(any(target_arch = "wasm32", test))]
    pub(crate) fn take_open_waiter(&mut self) -> Option<Waker> {
        self.open_waiter.take()
    }

    #[cfg(any(target_arch = "wasm32", test))]
    fn poll_open(
        &mut self,
        cx: &Context<'_>,
        replaces_existing_waiter: bool,
    ) -> Poll<io::Result<()>> {
        match self.status {
            WebSocketStatus::Connecting => {
                if let Some(waiter) = &self.open_waiter {
                    if !waiter.will_wake(cx.waker()) {
                        if replaces_existing_waiter {
                            self.open_waiter = Some(cx.waker().clone());
                            return Poll::Pending;
                        }
                        return Poll::Ready(Err(io::Error::new(
                            io::ErrorKind::AlreadyExists,
                            "only one WebSocket OPEN waiter may be pending",
                        )));
                    }
                } else {
                    self.open_waiter = Some(cx.waker().clone());
                }
                Poll::Pending
            }
            WebSocketStatus::Open => Poll::Ready(Ok(())),
            WebSocketStatus::Closed { code } => Poll::Ready(Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("WebSocket closed with code {code} before OPEN"),
            ))),
            WebSocketStatus::Failed { kind, message } => {
                Poll::Ready(Err(io::Error::new(kind, message)))
            }
        }
    }

    pub(crate) fn enqueue_message(&mut self, message: Vec<u8>) -> MessageEnqueue {
        if message.len() > self.limits.max_message_bytes() {
            let waker = self.fail(
                io::ErrorKind::InvalidData,
                "WebSocket message exceeds configured byte bound",
            );
            return MessageEnqueue::Rejected {
                error: io::Error::new(
                    io::ErrorKind::InvalidData,
                    "WebSocket message exceeds configured byte bound",
                ),
                waker,
            };
        }

        if self.incoming.len() >= self.limits.max_queued_messages() {
            let waker = self.fail(
                io::ErrorKind::OutOfMemory,
                "WebSocket receive queue exceeds configured message bound",
            );
            return MessageEnqueue::Rejected {
                error: io::Error::new(
                    io::ErrorKind::OutOfMemory,
                    "WebSocket receive queue exceeds configured message bound",
                ),
                waker,
            };
        }

        if matches!(self.status, WebSocketStatus::Connecting) {
            self.status = WebSocketStatus::Open;
        }
        if !matches!(self.status, WebSocketStatus::Open) {
            return MessageEnqueue::Rejected {
                error: io::Error::new(
                    io::ErrorKind::BrokenPipe,
                    "WebSocket is no longer accepting messages",
                ),
                waker: None,
            };
        }

        if self.incoming.try_reserve(1).is_err() {
            let waker = self.fail(
                io::ErrorKind::OutOfMemory,
                "WebSocket receive queue could not reserve capacity",
            );
            return MessageEnqueue::Rejected {
                error: io::Error::new(
                    io::ErrorKind::OutOfMemory,
                    "WebSocket receive queue could not reserve capacity",
                ),
                waker,
            };
        }

        self.incoming.push_back(message);
        MessageEnqueue::Accepted(self.waiter.take())
    }

    pub(crate) fn close(&mut self, code: u16) -> Option<Waker> {
        if matches!(
            self.status,
            WebSocketStatus::Connecting | WebSocketStatus::Open
        ) {
            self.status = WebSocketStatus::Closed { code };
            self.waiter.take()
        } else {
            None
        }
    }

    pub(crate) fn fail(&mut self, kind: io::ErrorKind, message: &'static str) -> Option<Waker> {
        if matches!(
            self.status,
            WebSocketStatus::Connecting | WebSocketStatus::Open
        ) {
            self.incoming.clear();
            self.status = WebSocketStatus::Failed { kind, message };
            self.waiter.take()
        } else {
            None
        }
    }

    pub(crate) fn take_message(&mut self) -> io::Result<Vec<u8>> {
        if let Some(message) = self.incoming.pop_front() {
            return Ok(message);
        }

        match self.status {
            WebSocketStatus::Connecting | WebSocketStatus::Open => Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "WebSocket has no message available",
            )),
            WebSocketStatus::Closed { code } => Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("WebSocket closed with code {code}"),
            )),
            WebSocketStatus::Failed { kind, message } => Err(io::Error::new(kind, message)),
        }
    }

    fn poll_receive(
        &mut self,
        cx: &Context<'_>,
        replaces_existing_waiter: bool,
    ) -> Poll<io::Result<Vec<u8>>> {
        if let Some(message) = self.incoming.pop_front() {
            self.waiter = None;
            return Poll::Ready(Ok(message));
        }

        match self.status {
            WebSocketStatus::Connecting | WebSocketStatus::Open => {
                if let Some(waiter) = &self.waiter {
                    if !waiter.will_wake(cx.waker()) {
                        if replaces_existing_waiter {
                            self.waiter = Some(cx.waker().clone());
                            return Poll::Pending;
                        }
                        return Poll::Ready(Err(io::Error::new(
                            io::ErrorKind::AlreadyExists,
                            "only one WebSocket receive may be pending",
                        )));
                    }
                } else {
                    self.waiter = Some(cx.waker().clone());
                }
                Poll::Pending
            }
            WebSocketStatus::Closed { code } => Poll::Ready(Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("WebSocket closed with code {code}"),
            ))),
            WebSocketStatus::Failed { kind, message } => {
                Poll::Ready(Err(io::Error::new(kind, message)))
            }
        }
    }

    fn clear_waiter(&mut self, candidate: Option<&Waker>) {
        if let Some(candidate) = candidate {
            if self
                .waiter
                .as_ref()
                .is_some_and(|waiter| waiter.will_wake(candidate))
            {
                self.waiter = None;
            }
        }
    }

    #[cfg(any(target_arch = "wasm32", test))]
    fn clear_open_waiter(&mut self, candidate: Option<&Waker>) {
        if let Some(candidate) = candidate {
            if self
                .open_waiter
                .as_ref()
                .is_some_and(|waiter| waiter.will_wake(candidate))
            {
                self.open_waiter = None;
            }
        }
    }
}

pub(crate) enum MessageEnqueue {
    Accepted(Option<Waker>),
    Rejected {
        error: io::Error,
        waker: Option<Waker>,
    },
}

fn lock_state<'a>(
    state: &'a Arc<Mutex<WebSocketState>>,
) -> io::Result<MutexGuard<'a, WebSocketState>> {
    state
        .lock()
        .map_err(|_| io::Error::other("WebSocket receive state lock is poisoned"))
}

/// A cancellation-safe future for the next browser WebSocket message.
#[must_use = "poll the future to observe the next WebSocket message"]
pub struct WebSocketReceive {
    state: Arc<Mutex<WebSocketState>>,
    registered_waker: Option<Waker>,
}

impl WebSocketReceive {
    pub(crate) fn new(state: Arc<Mutex<WebSocketState>>) -> Self {
        Self {
            state,
            registered_waker: None,
        }
    }
}

impl Future for WebSocketReceive {
    type Output = io::Result<Vec<u8>>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let mut state = match lock_state(&this.state) {
            Ok(state) => state,
            Err(error) => return Poll::Ready(Err(error)),
        };
        let replaces_existing_waiter = this.registered_waker.as_ref().is_some_and(|registered| {
            state
                .waiter
                .as_ref()
                .is_some_and(|waiter| waiter.will_wake(registered))
        });
        let result = state.poll_receive(cx, replaces_existing_waiter);
        match result {
            Poll::Pending => this.registered_waker = Some(cx.waker().clone()),
            Poll::Ready(_) => this.registered_waker = None,
        }
        result
    }
}

impl Drop for WebSocketReceive {
    fn drop(&mut self) {
        if let Ok(mut state) = self.state.lock() {
            state.clear_waiter(self.registered_waker.as_ref());
        }
    }
}

/// A cancellation-safe future that resolves when a browser WebSocket is OPEN.
#[cfg(any(target_arch = "wasm32", test))]
#[must_use = "poll the future to observe WebSocket OPEN"]
pub struct WebSocketOpen {
    state: Arc<Mutex<WebSocketState>>,
    registered_waker: Option<Waker>,
}

impl WebSocketOpen {
    pub(crate) fn new(state: Arc<Mutex<WebSocketState>>) -> Self {
        Self {
            state,
            registered_waker: None,
        }
    }
}

#[cfg(any(target_arch = "wasm32", test))]
impl Future for WebSocketOpen {
    type Output = io::Result<()>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.get_mut();
        let mut state = match lock_state(&this.state) {
            Ok(state) => state,
            Err(error) => return Poll::Ready(Err(error)),
        };
        let replaces_existing_waiter = this.registered_waker.as_ref().is_some_and(|registered| {
            state
                .open_waiter
                .as_ref()
                .is_some_and(|waiter| waiter.will_wake(registered))
        });
        let result = state.poll_open(cx, replaces_existing_waiter);
        match result {
            Poll::Pending => this.registered_waker = Some(cx.waker().clone()),
            Poll::Ready(_) => this.registered_waker = None,
        }
        result
    }
}

#[cfg(any(target_arch = "wasm32", test))]
impl Drop for WebSocketOpen {
    fn drop(&mut self) {
        if let Ok(mut state) = self.state.lock() {
            state.clear_open_waiter(self.registered_waker.as_ref());
        }
    }
}

#[cfg(test)]
#[path = "websocket_state_tests.rs"]
mod tests;

//! WebAssembly async I/O reactor implementation.
//!
//! This module provides async I/O support for WebAssembly environments,
//! integrating with JavaScript Promise/async-await patterns and Web APIs.

mod dom;
mod file;
mod timer;
mod websocket;

use std::collections::{HashMap, VecDeque};
use std::io;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use web_sys::console;

use crate::{Event, Interest, RawFd, Reactor};

pub use self::dom::{
    CompositionMetadata, DropFiles, DropMetadata, DroppedFile, DroppedFileAccess, PointerMetadata,
    PointerModifiers, PointerType, TextInputMetadata, TextSelection, TextSelectionDirection,
    WebDocument, WebElement, WebEvent, WebEventListener, WheelDeltaMode, WheelMetadata,
};
pub use self::file::{MAX_READ_BYTES, WebFile};
pub use self::timer::WebTimer;
pub use crate::local_task::LocalTaskHandle;
pub use crate::websocket_state::{WebSocketLimits, WebSocketOpen, WebSocketReceive};

use self::websocket::EVENT_QUEUE_CAPACITY;
use self::websocket::WebSocketConnection;

/// Schedules a future on the browser event loop.
///
/// The future is owned by the JavaScript event-loop integration and is
/// dropped when it resolves or is cancelled by the future itself. Callers
/// must keep any application state captured by the future in an owned handle.
pub fn spawn_local<F>(future: F)
where
    F: std::future::Future<Output = ()> + 'static,
{
    wasm_bindgen_futures::spawn_local(future);
}

/// Schedules a browser task and returns its cancellation handle.
///
/// Cancelling or dropping the handle wakes the task and drops its child
/// future. This releases owned WebSocket receives, timers and other PAL
/// resources without waiting for another browser event.
#[must_use = "retain the handle to cancel the browser task"]
pub fn spawn_local_with_handle<F>(future: F) -> LocalTaskHandle
where
    F: std::future::Future<Output = ()> + 'static,
{
    let (handle, future) = crate::local_task::cancellable(future);
    wasm_bindgen_futures::spawn_local(future);
    handle
}

/// WebAssembly-based I/O reactor using Web APIs.
pub struct WebReactor {
    /// JavaScript event queue for async operations
    pending_events: Arc<Mutex<VecDeque<Event>>>,
    /// WebSocket connections tracking
    websockets: HashMap<RawFd, WebSocketConnection>,
    /// Next file descriptor ID
    next_fd: RawFd,
    /// Registered interests for file descriptors
    fd_interests: Arc<Mutex<HashMap<RawFd, Interest>>>,
}

impl WebReactor {
    /// Create a new WebAssembly reactor.
    pub fn new() -> io::Result<Self> {
        console::log_1(&"Initializing Moirai WebAssembly reactor".into());

        Ok(Self {
            pending_events: Arc::new(Mutex::new(VecDeque::with_capacity(EVENT_QUEUE_CAPACITY))),
            websockets: HashMap::new(),
            next_fd: 1,
            fd_interests: Arc::new(Mutex::new(HashMap::new())),
        })
    }

    /// Allocate a new file descriptor ID.
    fn allocate_fd(&mut self) -> RawFd {
        let fd = self.next_fd;
        self.next_fd += 1;
        fd
    }

    /// Create a WebSocket connection and return its file descriptor.
    pub fn create_websocket(&mut self, url: &str) -> io::Result<RawFd> {
        self.create_websocket_with_limits(url, WebSocketLimits::default())
    }

    /// Create a WebSocket with explicit message and queue bounds.
    pub fn create_websocket_with_limits(
        &mut self,
        url: &str,
        limits: WebSocketLimits,
    ) -> io::Result<RawFd> {
        let fd = self.allocate_fd();
        let websocket = WebSocketConnection::new(
            fd,
            url,
            limits,
            Arc::clone(&self.pending_events),
            Arc::clone(&self.fd_interests),
        )?;
        self.websockets.insert(fd, websocket);
        Ok(fd)
    }

    /// Send data through a WebSocket.
    pub fn websocket_send(&self, fd: RawFd, data: &[u8]) -> io::Result<()> {
        let websocket = self
            .websockets
            .get(&fd)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "WebSocket not found"))?;
        websocket.send(data)
    }

    /// Receive one queued WebSocket message without blocking.
    pub fn websocket_recv(&self, fd: RawFd) -> io::Result<Vec<u8>> {
        let websocket = self
            .websockets
            .get(&fd)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "WebSocket not found"))?;
        websocket.receive()
    }

    /// Return a cancellation-safe future for the next WebSocket message.
    pub fn websocket_recv_async(&self, fd: RawFd) -> io::Result<WebSocketReceive> {
        let websocket = self
            .websockets
            .get(&fd)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "WebSocket not found"))?;
        Ok(websocket.receive_async())
    }

    /// Return a cancellation-safe future that resolves when the WebSocket is OPEN.
    pub fn websocket_open_async(&self, fd: RawFd) -> io::Result<WebSocketOpen> {
        let websocket = self
            .websockets
            .get(&fd)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "WebSocket not found"))?;
        Ok(websocket.open_async())
    }

    /// Close a WebSocket connection.
    pub fn websocket_close(&mut self, fd: RawFd) -> io::Result<()> {
        let websocket = self
            .websockets
            .remove(&fd)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "WebSocket not found"))?;
        self.fd_interests
            .lock()
            .map_err(|_| io::Error::other("WebSocket interest lock is poisoned"))?
            .remove(&fd);
        self.pending_events
            .lock()
            .map_err(|_| io::Error::other("WebSocket event lock is poisoned"))?
            .retain(|event| event.fd != fd);
        drop(websocket);
        Ok(())
    }
}

impl Reactor for WebReactor {
    fn register_fd(&self, fd: RawFd, interest: Interest) -> io::Result<()> {
        if !self.websockets.contains_key(&fd) {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "WebSocket not found",
            ));
        }
        self.fd_interests
            .lock()
            .map_err(|_| io::Error::other("WebSocket interest lock is poisoned"))?
            .insert(fd, interest);
        Ok(())
    }

    fn unregister_fd(&self, fd: RawFd) -> io::Result<()> {
        self.fd_interests
            .lock()
            .map_err(|_| io::Error::other("WebSocket interest lock is poisoned"))?
            .remove(&fd);
        Ok(())
    }

    fn poll_events(&self, _timeout: Option<Duration>) -> io::Result<Vec<Event>> {
        let mut pending_events = self
            .pending_events
            .lock()
            .map_err(|_| io::Error::other("WebSocket event lock is poisoned"))?;
        Ok(pending_events.drain(..).collect())
    }

    fn wake(&self) -> io::Result<()> {
        // Browser callbacks wake the exact receive future that owns the wait;
        // there is no blocking reactor thread to interrupt here.
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_web_reactor_creation() {
        let reactor = WebReactor::new();
        assert!(reactor.is_ok());
    }
}

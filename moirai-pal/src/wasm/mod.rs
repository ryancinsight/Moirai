//! WebAssembly async I/O reactor implementation.
//!
//! This module provides async I/O support for WebAssembly environments,
//! integrating with JavaScript Promise/async-await patterns and Web APIs.

mod websocket;

use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::io;
use std::rc::Rc;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use js_sys::Promise;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::console;

use crate::{Event, Interest, RawFd, Reactor};

pub use crate::websocket_state::{WebSocketLimits, WebSocketReceive};

use self::websocket::EVENT_QUEUE_CAPACITY;
use self::websocket::WebSocketConnection;

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

/// WebAssembly-specific async file operations using File API.
pub struct WebFile {
    /// JavaScript File object
    file_handle: web_sys::File,
    /// Current read position
    position: u64,
}

/// Owns FileReader callbacks for one read and detaches them when the future
/// completes or is cancelled. Keeping the closures in this guard avoids the
/// permanent JavaScript roots created by `Closure::forget`.
struct FileReaderCallbacks {
    reader: web_sys::FileReader,
    onload: Closure<dyn FnMut(JsValue)>,
    onerror: Closure<dyn FnMut(JsValue)>,
}

impl Drop for FileReaderCallbacks {
    fn drop(&mut self) {
        // Reading the handles makes their ownership explicit: dropping them
        // releases the JavaScript callbacks after the event target is cleared.
        let _ = (&self.onload, &self.onerror);
        self.reader.set_onload(None);
        self.reader.set_onerror(None);
        self.reader.abort();
    }
}

impl WebFile {
    /// Create a WebFile from a JavaScript File object.
    pub fn from_js_file(file: web_sys::File) -> Self {
        Self {
            file_handle: file,
            position: 0,
        }
    }

    /// Read data from the file using FileReader API.
    pub async fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let file_reader = web_sys::FileReader::new()
            .map_err(|_| io::Error::other("Failed to create FileReader"))?;

        // Create a blob slice for the read operation
        let end_position = std::cmp::min(
            self.position + buf.len() as u64,
            self.file_handle.size() as u64,
        );

        let blob = self
            .file_handle
            .slice_with_f64_and_f64(self.position as f64, end_position as f64)
            .map_err(|_| io::Error::other("Failed to create blob slice"))?;

        // Convert the FileReader operation to a Future. The callback guard is
        // stored outside the Promise constructor so its closures remain owned
        // until the read resolves or the async operation is cancelled.
        let callback_slot = Rc::new(RefCell::new(None));
        let callback_slot_for_promise = Rc::clone(&callback_slot);
        let reader_for_promise = file_reader.clone();
        let promise = Promise::new(&mut |resolve, reject| {
            let onload = Closure::wrap(Box::new(move |_event: JsValue| {
                if let Err(error) = resolve.call0(&JsValue::NULL) {
                    console::error_1(&error);
                }
            }) as Box<dyn FnMut(JsValue)>);

            let onerror = Closure::wrap(Box::new(move |_event: JsValue| {
                if let Err(error) = reject.call0(&JsValue::NULL) {
                    console::error_1(&error);
                }
            }) as Box<dyn FnMut(JsValue)>);

            reader_for_promise.set_onload(Some(onload.as_ref().unchecked_ref()));
            reader_for_promise.set_onerror(Some(onerror.as_ref().unchecked_ref()));
            *callback_slot_for_promise.borrow_mut() = Some(FileReaderCallbacks {
                reader: reader_for_promise.clone(),
                onload,
                onerror,
            });
        });

        let _callbacks = callback_slot
            .borrow_mut()
            .take()
            .ok_or_else(|| io::Error::other("FileReader callbacks were not installed"))?;

        // Start the read only after both callbacks are attached.
        file_reader
            .read_as_array_buffer(&blob)
            .map_err(|_| io::Error::other("Failed to start read operation"))?;

        // Wait for the read to complete
        JsFuture::from(promise)
            .await
            .map_err(|_| io::Error::other("File read failed"))?;

        // Get the result and copy to buffer
        let result = file_reader
            .result()
            .map_err(|_| io::Error::other("Failed to get read result"))?;

        let array_buffer = js_sys::ArrayBuffer::from(result);
        let uint8_array = js_sys::Uint8Array::new(&array_buffer);
        let len = std::cmp::min(uint8_array.length() as usize, buf.len());

        uint8_array.copy_to(&mut buf[..len]);
        self.position += len as u64;

        Ok(len)
    }

    /// Get file size.
    pub fn size(&self) -> u64 {
        self.file_handle.size() as u64
    }

    /// Get current position.
    pub fn position(&self) -> u64 {
        self.position
    }

    /// Seek to a position.
    pub fn seek(&mut self, pos: u64) -> io::Result<()> {
        if pos > self.size() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "Seek position beyond file size",
            ));
        }
        self.position = pos;
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

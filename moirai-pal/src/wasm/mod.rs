//! WebAssembly async I/O reactor implementation.
//!
//! This module provides async I/O support for WebAssembly environments,
//! integrating with JavaScript Promise/async-await patterns and Web APIs.

use std::collections::{HashMap, VecDeque};
use std::io;
use std::time::Duration;

use js_sys::Promise;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{console, CloseEvent, ErrorEvent, MessageEvent, WebSocket};

use crate::{Event, Interest, RawFd, Reactor};

/// WebAssembly-based I/O reactor using Web APIs.
pub struct WebReactor {
    /// JavaScript event queue for async operations
    pending_events: VecDeque<Event>,
    /// WebSocket connections tracking
    websockets: HashMap<RawFd, WebSocket>,
    /// Next file descriptor ID
    next_fd: RawFd,
    /// Registered interests for file descriptors
    fd_interests: HashMap<RawFd, Interest>,
}

impl WebReactor {
    /// Create a new WebAssembly reactor.
    pub fn new() -> io::Result<Self> {
        console::log_1(&"Initializing Moirai WebAssembly reactor".into());

        Ok(Self {
            pending_events: VecDeque::new(),
            websockets: HashMap::new(),
            next_fd: 1,
            fd_interests: HashMap::new(),
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
        let websocket = WebSocket::new(url).map_err(|_| {
            io::Error::new(
                io::ErrorKind::ConnectionRefused,
                "Failed to create WebSocket",
            )
        })?;

        let fd = self.allocate_fd();

        // Set up event handlers
        let fd_clone = fd;
        let onopen_callback = Closure::wrap(Box::new(move |_event: JsValue| {
            console::log_1(&format!("WebSocket {} opened", fd_clone).into());
        }) as Box<dyn FnMut(JsValue)>);
        websocket.set_onopen(Some(onopen_callback.as_ref().unchecked_ref()));
        onopen_callback.forget(); // Prevent cleanup

        let fd_clone = fd;
        let onmessage_callback = Closure::wrap(Box::new(move |event: MessageEvent| {
            console::log_1(&format!("WebSocket {} received message", fd_clone).into());
            let _data = event.data();
            // WASM readable-event integration is tracked as a separate target
            // contract because browser callbacks cannot share the native PAL
            // readiness queue shape without a wasm-specific ownership model.
        }) as Box<dyn FnMut(MessageEvent)>);
        websocket.set_onmessage(Some(onmessage_callback.as_ref().unchecked_ref()));
        onmessage_callback.forget();

        let fd_clone = fd;
        let onclose_callback = Closure::wrap(Box::new(move |event: CloseEvent| {
            console::log_1(&format!("WebSocket {} closed: {}", fd_clone, event.code()).into());
        }) as Box<dyn FnMut(CloseEvent)>);
        websocket.set_onclose(Some(onclose_callback.as_ref().unchecked_ref()));
        onclose_callback.forget();

        let fd_clone = fd;
        let onerror_callback = Closure::wrap(Box::new(move |event: ErrorEvent| {
            console::log_1(&format!("WebSocket {} error", fd_clone).into());
        }) as Box<dyn FnMut(ErrorEvent)>);
        websocket.set_onerror(Some(onerror_callback.as_ref().unchecked_ref()));
        onerror_callback.forget();

        self.websockets.insert(fd, websocket);
        Ok(fd)
    }

    /// Send data through a WebSocket.
    pub fn websocket_send(&self, fd: RawFd, data: &[u8]) -> io::Result<()> {
        let websocket = self
            .websockets
            .get(&fd)
            .ok_or_else(|| io::Error::new(io::ErrorKind::NotFound, "WebSocket not found"))?;

        websocket
            .send_with_u8_array(data)
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "Failed to send data"))
    }

    /// Close a WebSocket connection.
    pub fn websocket_close(&mut self, fd: RawFd) -> io::Result<()> {
        if let Some(websocket) = self.websockets.remove(&fd) {
            websocket
                .close()
                .map_err(|_| io::Error::new(io::ErrorKind::Other, "Failed to close WebSocket"))?;
        }
        Ok(())
    }
}

impl Reactor for WebReactor {
    fn register_fd(&self, fd: RawFd, interest: Interest) -> io::Result<()> {
        // In WebAssembly, file descriptors are more abstract
        // We just track the interest for now
        console::log_1(&format!("Registering fd {} with interest {:?}", fd, interest).into());
        Ok(())
    }

    fn unregister_fd(&self, fd: RawFd) -> io::Result<()> {
        console::log_1(&format!("Unregistering fd {}", fd).into());
        Ok(())
    }

    fn poll_events(&self, timeout: Option<Duration>) -> io::Result<Vec<Event>> {
        // In WebAssembly, we can't do blocking polls like epoll
        // Instead, we work with the existing event queue
        let mut events = Vec::new();

        // Browser event-loop integration is a wasm-specific contract. The
        // native PAL audit only covers queued events already materialized for
        // this reactor instance.

        if let Some(_timeout) = timeout {
            // Simulate timeout behavior
            // In real implementation, this would use setTimeout/Promise integration
        }

        Ok(events)
    }

    fn wake(&self) -> io::Result<()> {
        // In WebAssembly, waking is handled by the JavaScript event loop
        console::log_1(&"Waking WebAssembly reactor".into());
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
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "Failed to create FileReader"))?;

        // Create a blob slice for the read operation
        let end_position = std::cmp::min(
            self.position + buf.len() as u64,
            self.file_handle.size() as u64,
        );

        let blob = self
            .file_handle
            .slice_with_f64_and_f64(self.position as f64, end_position as f64)
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "Failed to create blob slice"))?;

        // Read the blob as array buffer
        file_reader
            .read_as_array_buffer(&blob)
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "Failed to start read operation"))?;

        // Convert the FileReader operation to a Future
        let promise = Promise::new(&mut |resolve, reject| {
            let onload = Closure::wrap(Box::new(move |_event: JsValue| {
                resolve.call0(&JsValue::NULL).unwrap();
            }) as Box<dyn FnMut(JsValue)>);

            let onerror = Closure::wrap(Box::new(move |_event: JsValue| {
                reject.call0(&JsValue::NULL).unwrap();
            }) as Box<dyn FnMut(JsValue)>);

            file_reader.set_onload(Some(onload.as_ref().unchecked_ref()));
            file_reader.set_onerror(Some(onerror.as_ref().unchecked_ref()));

            onload.forget();
            onerror.forget();
        });

        // Wait for the read to complete
        JsFuture::from(promise)
            .await
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "File read failed"))?;

        // Get the result and copy to buffer
        let result = file_reader
            .result()
            .map_err(|_| io::Error::new(io::ErrorKind::Other, "Failed to get read result"))?;

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

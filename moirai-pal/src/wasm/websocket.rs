//! Owned browser WebSocket callbacks and bounded receive transport.

use std::collections::{HashMap, VecDeque};
use std::io;
use std::sync::{Arc, Mutex, MutexGuard};

use js_sys::{ArrayBuffer, Uint8Array};
use wasm_bindgen::prelude::*;
use web_sys::{console, BinaryType, CloseEvent, ErrorEvent, MessageEvent, WebSocket};

use crate::websocket_state::{MessageEnqueue, WebSocketOpen, WebSocketReceive, WebSocketState};
use crate::{Event, Interest, RawFd};

pub(crate) const EVENT_QUEUE_CAPACITY: usize = 256;

/// A browser WebSocket with Rust-owned callback lifetime.
pub(crate) struct WebSocketConnection {
    socket: WebSocket,
    state: Arc<Mutex<WebSocketState>>,
    onopen: Closure<dyn FnMut(JsValue)>,
    onmessage: Closure<dyn FnMut(MessageEvent)>,
    onclose: Closure<dyn FnMut(CloseEvent)>,
    onerror: Closure<dyn FnMut(ErrorEvent)>,
}

impl WebSocketConnection {
    pub(crate) fn new(
        fd: RawFd,
        url: &str,
        limits: crate::websocket_state::WebSocketLimits,
        pending_events: Arc<Mutex<VecDeque<Event>>>,
        fd_interests: Arc<Mutex<HashMap<RawFd, Interest>>>,
    ) -> io::Result<Self> {
        let socket = WebSocket::new(url).map_err(|_| {
            io::Error::new(
                io::ErrorKind::ConnectionRefused,
                "Failed to create WebSocket",
            )
        })?;
        socket.set_binary_type(BinaryType::Arraybuffer);

        let state = Arc::new(Mutex::new(WebSocketState::new(limits)));

        let open_state = Arc::clone(&state);
        let open_events = Arc::clone(&pending_events);
        let open_interests = Arc::clone(&fd_interests);
        let onopen = Closure::wrap(Box::new(move |_event: JsValue| {
            let (opened, waiter) = match open_state.lock() {
                Ok(mut state) => {
                    let opened = state.open();
                    let waiter = opened.then(|| state.take_open_waiter()).flatten();
                    (opened, waiter)
                }
                Err(_) => {
                    fail_connection(
                        &open_state,
                        &open_events,
                        &open_interests,
                        fd,
                        io::ErrorKind::Other,
                        "WebSocket receive state lock is poisoned",
                    );
                    return;
                }
            };
            if let Some(waiter) = waiter {
                waiter.wake();
            }
            if opened {
                let queued = enqueue_event(
                    &open_events,
                    &open_interests,
                    Event {
                        fd,
                        readable: false,
                        writable: true,
                        error: false,
                        hangup: false,
                    },
                );
                if !queued {
                    fail_connection(
                        &open_state,
                        &open_events,
                        &open_interests,
                        fd,
                        io::ErrorKind::OutOfMemory,
                        "WebSocket event queue exceeds its configured bound",
                    );
                }
            }
        }) as Box<dyn FnMut(JsValue)>);
        socket.set_onopen(Some(onopen.as_ref().unchecked_ref()));

        let message_state = Arc::clone(&state);
        let message_events = Arc::clone(&pending_events);
        let message_interests = Arc::clone(&fd_interests);
        let onmessage = Closure::wrap(Box::new(move |event: MessageEvent| {
            let payload = match decode_message(event.data(), limits.max_message_bytes()) {
                Ok(payload) => payload,
                Err((kind, message)) => {
                    fail_connection(
                        &message_state,
                        &message_events,
                        &message_interests,
                        fd,
                        kind,
                        message,
                    );
                    return;
                }
            };

            let enqueue = message_state.lock().map(|mut state| {
                let opened = state.open();
                let open_waiter = opened.then(|| state.take_open_waiter()).flatten();
                (open_waiter, state.enqueue_message(payload))
            });
            let (open_waiter, enqueue) = match enqueue {
                Ok(enqueue) => enqueue,
                Err(_) => {
                    fail_connection(
                        &message_state,
                        &message_events,
                        &message_interests,
                        fd,
                        io::ErrorKind::Other,
                        "WebSocket receive state lock is poisoned",
                    );
                    return;
                }
            };
            if let Some(open_waiter) = open_waiter {
                open_waiter.wake();
            }

            match enqueue {
                MessageEnqueue::Accepted(waiter) => {
                    if let Some(waiter) = waiter {
                        waiter.wake();
                    }
                    if !enqueue_event(
                        &message_events,
                        &message_interests,
                        Event {
                            fd,
                            readable: true,
                            writable: false,
                            error: false,
                            hangup: false,
                        },
                    ) {
                        fail_connection(
                            &message_state,
                            &message_events,
                            &message_interests,
                            fd,
                            io::ErrorKind::OutOfMemory,
                            "WebSocket event queue exceeds its configured bound",
                        );
                    }
                }
                MessageEnqueue::Rejected { error, waker } => {
                    if let Some(waker) = waker {
                        waker.wake();
                    }
                    let queued = enqueue_event(
                        &message_events,
                        &message_interests,
                        Event {
                            fd,
                            readable: false,
                            writable: false,
                            error: true,
                            hangup: false,
                        },
                    );
                    if !queued {
                        console::error_1(&"WebSocket error event could not be queued".into());
                    }
                    console::error_1(&error.to_string().into());
                }
            }
        }) as Box<dyn FnMut(MessageEvent)>);
        socket.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));

        let close_state = Arc::clone(&state);
        let close_events = Arc::clone(&pending_events);
        let close_interests = Arc::clone(&fd_interests);
        let onclose = Closure::wrap(Box::new(move |event: CloseEvent| {
            let waiters = close_state.lock().ok().map(|mut state| {
                let waiter = state.close(event.code());
                let open_waiter = state.take_open_waiter();
                (waiter, open_waiter)
            });
            let (waiter, open_waiter) = waiters.unwrap_or((None, None));
            if let Some(waiter) = waiter {
                waiter.wake();
            }
            if let Some(open_waiter) = open_waiter {
                open_waiter.wake();
            }
            let queued = enqueue_event(
                &close_events,
                &close_interests,
                Event {
                    fd,
                    readable: false,
                    writable: false,
                    error: !event.was_clean(),
                    hangup: true,
                },
            );
            if !queued {
                console::error_1(&"WebSocket close event could not be queued".into());
            }
        }) as Box<dyn FnMut(CloseEvent)>);
        socket.set_onclose(Some(onclose.as_ref().unchecked_ref()));

        let error_state = Arc::clone(&state);
        let error_events = Arc::clone(&pending_events);
        let error_interests = Arc::clone(&fd_interests);
        let onerror = Closure::wrap(Box::new(move |_event: ErrorEvent| {
            fail_connection(
                &error_state,
                &error_events,
                &error_interests,
                fd,
                io::ErrorKind::ConnectionAborted,
                "WebSocket reported a browser error",
            );
        }) as Box<dyn FnMut(ErrorEvent)>);
        socket.set_onerror(Some(onerror.as_ref().unchecked_ref()));

        Ok(Self {
            socket,
            state,
            onopen,
            onmessage,
            onclose,
            onerror,
        })
    }

    pub(crate) fn send(&self, data: &[u8]) -> io::Result<()> {
        self.socket
            .send_with_u8_array(data)
            .map_err(|_| io::Error::new(io::ErrorKind::BrokenPipe, "Failed to send data"))
    }

    pub(crate) fn receive(&self) -> io::Result<Vec<u8>> {
        lock_state(&self.state)?.take_message()
    }

    pub(crate) fn receive_async(&self) -> WebSocketReceive {
        WebSocketReceive::new(Arc::clone(&self.state))
    }

    pub(crate) fn open_async(&self) -> WebSocketOpen {
        WebSocketOpen::new(Arc::clone(&self.state))
    }
}

impl Drop for WebSocketConnection {
    fn drop(&mut self) {
        self.socket.set_onopen(None);
        self.socket.set_onmessage(None);
        self.socket.set_onclose(None);
        self.socket.set_onerror(None);

        if let Ok(mut state) = self.state.lock() {
            let waiter = state.fail(
                io::ErrorKind::Interrupted,
                "WebSocket connection was cancelled",
            );
            let open_waiter = state.take_open_waiter();
            if let Some(waiter) = waiter {
                waiter.wake();
            }
            if let Some(open_waiter) = open_waiter {
                open_waiter.wake();
            }
        }

        if let Err(_error) = self.socket.close() {
            console::error_1(&"Failed to close cancelled WebSocket".into());
        }

        let _ = (&self.onopen, &self.onmessage, &self.onclose, &self.onerror);
    }
}

fn decode_message(
    data: JsValue,
    max_message_bytes: usize,
) -> Result<Vec<u8>, (io::ErrorKind, &'static str)> {
    if let Some(text) = data.as_string() {
        if text.len() > max_message_bytes {
            return Err((
                io::ErrorKind::InvalidData,
                "WebSocket message exceeds configured byte bound",
            ));
        }
        return Ok(text.into_bytes());
    }

    if data.is_instance_of::<ArrayBuffer>() || ArrayBuffer::is_view(&data) {
        let bytes = Uint8Array::new(&data);
        let length = usize::try_from(bytes.length()).map_err(|_| {
            (
                io::ErrorKind::InvalidData,
                "WebSocket message length cannot be represented",
            )
        })?;
        if length > max_message_bytes {
            return Err((
                io::ErrorKind::InvalidData,
                "WebSocket message exceeds configured byte bound",
            ));
        }
        return Ok(bytes.to_vec());
    }

    Err((
        io::ErrorKind::InvalidData,
        "WebSocket message is neither text nor binary data",
    ))
}

fn fail_connection(
    state: &Arc<Mutex<WebSocketState>>,
    pending_events: &Arc<Mutex<VecDeque<Event>>>,
    fd_interests: &Arc<Mutex<HashMap<RawFd, Interest>>>,
    fd: RawFd,
    kind: io::ErrorKind,
    message: &'static str,
) {
    let waiters = state.lock().ok().map(|mut state| {
        let waiter = state.fail(kind, message);
        let open_waiter = state.take_open_waiter();
        (waiter, open_waiter)
    });
    let (waiter, open_waiter) = waiters.unwrap_or((None, None));
    if let Some(waiter) = waiter {
        waiter.wake();
    }
    if let Some(open_waiter) = open_waiter {
        open_waiter.wake();
    }
    let queued = enqueue_event(
        pending_events,
        fd_interests,
        Event {
            fd,
            readable: false,
            writable: false,
            error: true,
            hangup: false,
        },
    );
    if !queued {
        console::error_1(&"WebSocket error event could not be queued".into());
    }
}

fn enqueue_event(
    pending_events: &Arc<Mutex<VecDeque<Event>>>,
    fd_interests: &Arc<Mutex<HashMap<RawFd, Interest>>>,
    event: Event,
) -> bool {
    let interest = match fd_interests.lock() {
        Ok(interests) => interests.get(&event.fd).copied(),
        Err(_) => return false,
    };
    let Some(interest) = interest else {
        return true;
    };
    if !((event.readable && interest.readable)
        || (event.writable && interest.writable)
        || (event.error && interest.error)
        || (event.hangup && interest.error))
    {
        return true;
    }

    let Ok(mut events) = pending_events.lock() else {
        return false;
    };
    if events.len() >= EVENT_QUEUE_CAPACITY {
        return false;
    }
    events.push_back(event);
    true
}

fn lock_state<'a>(
    state: &'a Arc<Mutex<WebSocketState>>,
) -> io::Result<MutexGuard<'a, WebSocketState>> {
    state
        .lock()
        .map_err(|_| io::Error::other("WebSocket receive state lock is poisoned"))
}

use super::*;

impl ConnectionDriver {
    pub(super) fn new(conn: ConnectionRef) -> Self {
        let span = {
            let conn = &mut conn.state.lock("poll");
            debug_span!("drive", id = conn.handle.0)
        };

        Self { conn, span }
    }
}

impl Future for ConnectionDriver {
    type Output = Result<(), io::Error>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Self::Output> {
        let (mut conn, poisoned) = self.conn.state.lock_for_teardown("poll");
        if poisoned {
            conn.terminate(ConnectionError::LocallyClosed, &self.conn.shared);
            return Poll::Ready(Err(io::Error::other("QUIC connection state poisoned")));
        }
        let _guard = self.span.enter();

        let mut keep_going = match conn.process_conn_events(&self.conn.shared, cx) {
            Ok(more) => more,
            Err(e) => {
                conn.terminate(e, &self.conn.shared);
                return Poll::Ready(Ok(()));
            }
        };
        keep_going |= conn.drive_transmit(cx)?;
        // If a timer expires, there might be more to transmit. When we transmit something, we
        // might need to reset a timer. Hence, we must loop until neither happens.
        keep_going |= conn.drive_timer(cx);
        conn.forward_endpoint_events();
        conn.forward_app_events(&self.conn.shared);

        if !conn.inner.is_drained() {
            if keep_going {
                // If the connection hasn't processed all tasks, schedule it again
                // after releasing the state lock: wakers may poll immediately.
                drop(conn);
                cx.waker().wake_by_ref();
            } else {
                conn.driver = Some(cx.waker().clone());
            }
            return Poll::Pending;
        }
        if conn.error.is_none() {
            unreachable!("drained connections always have an error");
        }
        Poll::Ready(Ok(()))
    }
}

impl ConnectionRef {
    pub(super) fn new(init: ConnectionInit) -> Self {
        let ConnectionInit {
            handle,
            conn,
            endpoint_events,
            conn_events,
            on_handshake_data,
            on_connected,
            socket,
            runtime,
        } = init;
        Self(Arc::new(ConnectionInner {
            state: Mutex::new(State {
                inner: conn,
                driver: None,
                handle,
                on_handshake_data: Some(on_handshake_data),
                on_connected: Some(on_connected),
                connected: false,
                timer: None,
                timer_deadline: None,
                conn_events,
                endpoint_events,
                blocked_writers: FxHashMap::default(),
                blocked_readers: FxHashMap::default(),
                stopped: FxHashMap::default(),
                error: None,
                io_poller: socket.clone().create_io_poller(),
                socket,
                runtime,
                send_buffer: Vec::new(),
                buffered_transmit: None,
            }),
            shared: Shared::default(),
        }))
    }

    pub(super) fn stable_id(&self) -> usize {
        &*self.0 as *const _ as usize
    }
}

impl Clone for ConnectionRef {
    fn clone(&self) -> Self {
        self.shared.ref_count.fetch_add(1, Ordering::Relaxed);
        Self(self.0.clone())
    }
}

impl Drop for ConnectionRef {
    fn drop(&mut self) {
        if self.shared.ref_count.fetch_sub(1, Ordering::Relaxed) > 1 {
            return;
        }

        let (mut conn, poisoned) = self.state.lock_for_teardown("drop");

        if poisoned {
            // A panic may have interrupted protocol mutation. Release application waiters and
            // State::drop then tells the endpoint to discard the connection; do not call proto.
            conn.terminate(ConnectionError::LocallyClosed, &self.shared);
            conn.wake();
            return;
        }

        if !conn.inner.is_closed() {
            // If the driver is alive, it's just it and us, so we'd better shut it down. If it's
            // not, we can't do any harm. If there were any streams being opened, then either
            // the connection will be closed for an unrelated reason or a fresh reference will
            // be constructed for the newly opened stream.
            conn.implicit_close(&self.shared);
        }
    }
}

impl std::ops::Deref for ConnectionRef {
    type Target = ConnectionInner;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

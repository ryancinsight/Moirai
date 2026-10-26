use super::*;

impl State {
    pub(super) fn drive_transmit(&mut self, cx: &mut Context) -> io::Result<bool> {
        let now = self.runtime.now();
        let mut transmits = 0;

        let max_datagrams = self
            .socket
            .max_transmit_segments()
            .min(MAX_TRANSMIT_SEGMENTS);

        loop {
            // Retry the last transmit, or get a new one.
            let t = match self.buffered_transmit.take() {
                Some(t) => {
                    transmits += transmit_datagrams(&t);
                    t
                }
                None => {
                    self.send_buffer.clear();
                    self.send_buffer.reserve(self.inner.current_mtu() as usize);
                    match self
                        .inner
                        .poll_transmit(now, max_datagrams, &mut self.send_buffer)
                    {
                        Some(t) => {
                            transmits += transmit_datagrams(&t);
                            t
                        }
                        None => break,
                    }
                }
            };

            if self.io_poller.as_mut().poll_writable(cx)?.is_pending() {
                // Retry after a future wakeup
                self.buffered_transmit = Some(t);
                return Ok(false);
            }

            let len = t.size;
            let retry = match self
                .socket
                .try_send(&udp_transmit(&t, &self.send_buffer[..len]))
            {
                Ok(()) => false,
                Err(ref e) if e.kind() == io::ErrorKind::WouldBlock => true,
                Err(e) => return Err(e),
            };
            if retry {
                // We thought the socket was writable, but it wasn't. Retry so that either another
                // `poll_writable` call determines that the socket is indeed not writable and
                // registers us for a wakeup, or the send succeeds if this really was just a
                // transient failure.
                self.buffered_transmit = Some(t);
                continue;
            }

            if transmits >= MAX_TRANSMIT_DATAGRAMS {
                // A final protocol poll distinguishes a genuinely exhausted send queue from a
                // busy one. The former updates proto's app-limited congestion state now; the
                // latter stages exactly one transmit for the next cooperative driver poll.
                self.send_buffer.clear();
                self.send_buffer.reserve(self.inner.current_mtu() as usize);
                self.buffered_transmit =
                    self.inner
                        .poll_transmit(now, max_datagrams, &mut self.send_buffer);
                return Ok(self.buffered_transmit.is_some());
            }
        }

        Ok(false)
    }

    pub(super) fn forward_endpoint_events(&mut self) {
        while let Some(event) = self.inner.poll_endpoint_events() {
            // If the endpoint driver is gone, noop.
            let _ = self.endpoint_events.enqueue((self.handle, event));
        }
    }

    /// If this returns `Err`, the endpoint is dead, so the driver should exit immediately.
    pub(super) fn process_conn_events(
        &mut self,
        shared: &Shared,
        cx: &mut Context,
    ) -> Result<bool, ConnectionError> {
        for _ in 0..MAX_CONNECTION_EVENTS_PER_POLL {
            match self.conn_events.poll_recv(cx) {
                Poll::Ready(Some(ConnectionEvent::Rebind(socket))) => {
                    self.socket = socket;
                    self.io_poller = self.socket.clone().create_io_poller();
                    self.inner.local_address_changed();
                }
                Poll::Ready(Some(ConnectionEvent::Proto(event))) => {
                    self.inner.handle_event(event);
                }
                Poll::Ready(Some(ConnectionEvent::Close { reason, error_code })) => {
                    self.close(error_code, reason, shared);
                }
                Poll::Ready(None) => {
                    return Err(ConnectionError::TransportError(proto::TransportError {
                        code: proto::TransportErrorCode::INTERNAL_ERROR,
                        frame: None,
                        reason: "endpoint driver future was dropped".to_string(),
                    }));
                }
                Poll::Pending => {
                    return Ok(false);
                }
            }
        }

        // The queue may still contain events. Continue the driver after this
        // poll's transmit, timer and application-event work has run.
        Ok(true)
    }

    pub(super) fn forward_app_events(&mut self, shared: &Shared) {
        while let Some(event) = self.inner.poll() {
            use proto::Event::*;
            match event {
                HandshakeDataReady => {
                    if let Some(x) = self.on_handshake_data.take() {
                        let _ = x.send(());
                    }
                }
                Connected => {
                    self.connected = true;
                    if let Some(x) = self.on_connected.take() {
                        // We don't care if the on-connected future was dropped
                        let _ = x.send(self.inner.accepted_0rtt());
                    }
                    if self.inner.side().is_client() && !self.inner.accepted_0rtt() {
                        // Wake up rejected 0-RTT streams so they can fail immediately with
                        // `ZeroRttRejected` errors.
                        wake_all(&mut self.blocked_writers);
                        wake_all(&mut self.blocked_readers);
                        wake_all_notify(&mut self.stopped);
                    }
                }
                ConnectionLost { reason } => {
                    self.terminate(reason, shared);
                }
                Stream(StreamEvent::Writable { id }) => wake_stream(id, &mut self.blocked_writers),
                Stream(StreamEvent::Opened { dir: Dir::Uni }) => {
                    shared.stream_incoming[Dir::Uni as usize].notify_waiters();
                }
                Stream(StreamEvent::Opened { dir: Dir::Bi }) => {
                    shared.stream_incoming[Dir::Bi as usize].notify_waiters();
                }
                DatagramReceived => {
                    shared.datagram_received.notify_waiters();
                }
                DatagramsUnblocked => {
                    shared.datagrams_unblocked.notify_waiters();
                }
                Stream(StreamEvent::Readable { id }) => wake_stream(id, &mut self.blocked_readers),
                Stream(StreamEvent::Available { dir }) => {
                    // Might mean any number of streams are ready, so we wake up everyone
                    shared.stream_budget_available[dir as usize].notify_waiters();
                }
                Stream(StreamEvent::Finished { id }) => wake_stream_notify(id, &mut self.stopped),
                Stream(StreamEvent::Stopped { id, .. }) => {
                    wake_stream_notify(id, &mut self.stopped);
                    wake_stream(id, &mut self.blocked_writers);
                }
            }
        }
    }

    pub(super) fn drive_timer(&mut self, cx: &mut Context<'_>) -> bool {
        let Some(deadline) = self.inner.poll_timeout() else {
            self.timer_deadline = None;
            return false;
        };

        // Use the clock rather than the async timer to detect expiry: Sleep::poll
        // respects Tokio's cooperative budget and can return Pending for elapsed
        // deadlines.
        let now = self.runtime.now();
        if now >= deadline {
            self.inner.handle_timeout(now);
            self.timer_deadline = None;
            return true;
        }

        match &mut self.timer {
            // Avoid resetting the timer when the deadline is unchanged.
            Some(delay) if self.timer_deadline != Some(deadline) => {
                delay.as_mut().reset(deadline);
            }
            None => {
                self.timer = Some(self.runtime.new_timer(deadline));
            }
            _ => {}
        }
        self.timer_deadline = Some(deadline);

        let delay = self
            .timer
            .as_mut()
            .expect("timer must exist in this state")
            .as_mut();
        if delay.poll(cx).is_pending() {
            return false;
        }

        // The deadline elapsed in the window between the clock check and poll.
        self.inner.handle_timeout(self.runtime.now());
        self.timer_deadline = None;
        true
    }

    /// Wake up a blocked `Driver` task to process I/O
    pub(crate) fn wake(&mut self) {
        if let Some(x) = self.driver.take() {
            x.wake();
        }
    }

    /// Used to wake up all blocked futures when the connection becomes closed for any reason
    pub(super) fn terminate(&mut self, reason: ConnectionError, shared: &Shared) {
        self.error = Some(reason.clone());
        if let Some(x) = self.on_handshake_data.take() {
            let _ = x.send(());
        }
        wake_all(&mut self.blocked_writers);
        wake_all(&mut self.blocked_readers);
        shared.stream_budget_available[Dir::Uni as usize].notify_waiters();
        shared.stream_budget_available[Dir::Bi as usize].notify_waiters();
        shared.stream_incoming[Dir::Uni as usize].notify_waiters();
        shared.stream_incoming[Dir::Bi as usize].notify_waiters();
        shared.datagram_received.notify_waiters();
        shared.datagrams_unblocked.notify_waiters();
        if let Some(x) = self.on_connected.take() {
            let _ = x.send(false);
        }
        wake_all_notify(&mut self.stopped);
        shared.closed.notify_waiters();
    }

    pub(super) fn close(&mut self, error_code: VarInt, reason: Bytes, shared: &Shared) {
        self.inner.close(self.runtime.now(), error_code, reason);
        self.terminate(ConnectionError::LocallyClosed, shared);
        self.wake();
    }

    /// Close for a reason other than the application's explicit request
    pub(crate) fn implicit_close(&mut self, shared: &Shared) {
        self.close(0u32.into(), Bytes::new(), shared);
    }

    pub(crate) fn check_0rtt(&self) -> Result<(), ()> {
        if self.inner.is_handshaking()
            || self.inner.accepted_0rtt()
            || self.inner.side().is_server()
        {
            Ok(())
        } else {
            Err(())
        }
    }
}

fn transmit_datagrams(transmit: &proto::Transmit) -> usize {
    match transmit.segment_size {
        None => 1,
        Some(segment_size) => transmit.size.div_ceil(segment_size),
    }
}

impl Drop for State {
    fn drop(&mut self) {
        if !self.inner.is_drained() {
            // Ensure the endpoint can tidy up
            let _ = self
                .endpoint_events
                .enqueue((self.handle, proto::EndpointEvent::drained()));
        }
    }
}

#[cfg(all(test, any(feature = "rustls-aws-lc-rs", feature = "rustls-ring")))]
#[path = "event_tests.rs"]
mod event_tests;

impl fmt::Debug for State {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("State").field("inner", &self.inner).finish()
    }
}

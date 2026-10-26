use super::*;

impl Future for EndpointDriver {
    type Output = Result<(), io::Error>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Self::Output> {
        let mut endpoint = self.0.state.lock().unwrap();
        if endpoint.driver.is_none() {
            endpoint.driver = Some(cx.waker().clone());
        }

        let now = endpoint.runtime.now();
        let mut keep_going = false;
        keep_going |= endpoint.drive_recv(cx, now)?;
        keep_going |= endpoint.handle_events(cx, &self.0.shared);

        if !endpoint.recv_state.incoming.is_empty() {
            self.0.shared.incoming.notify_waiters();
        }

        if self.0.shared.ref_count.load(Ordering::Relaxed) == 0
            && endpoint.recv_state.connections.is_empty()
        {
            Poll::Ready(Ok(()))
        } else {
            drop(endpoint);
            // If there is more work to do schedule the endpoint task again.
            // `wake_by_ref()` is called outside the lock to minimize
            // lock contention on a multithreaded runtime.
            if keep_going {
                cx.waker().wake_by_ref();
            }
            Poll::Pending
        }
    }
}

impl Drop for EndpointDriver {
    fn drop(&mut self) {
        let mut endpoint = self.0.state.lock().unwrap();
        endpoint.driver_lost = true;
        self.0.shared.incoming.notify_waiters();
        // Drop all outgoing channels, signaling the termination of the endpoint to the associated
        // connections.
        endpoint.recv_state.connections.senders.clear();
    }
}

impl State {
    pub(super) fn drive_recv(&mut self, cx: &mut Context, now: Instant) -> Result<bool, io::Error> {
        let get_time = || self.runtime.now();
        self.recv_state.recv_limiter.start_cycle(get_time);
        if let Some(socket) = &self.prev_socket {
            // We don't care about the `PollProgress` from old sockets.
            let poll_res =
                self.recv_state
                    .poll_socket(cx, &mut self.inner, &**socket, &*self.runtime, now);
            if poll_res.is_err() {
                self.prev_socket = None;
            }
        };
        let poll_res =
            self.recv_state
                .poll_socket(cx, &mut self.inner, &*self.socket, &*self.runtime, now);
        self.recv_state.recv_limiter.finish_cycle(get_time);
        let poll_res = poll_res?;
        if poll_res.received_connection_packet {
            // Traffic has arrived on self.socket, therefore there is no need for the abandoned
            // one anymore. TODO: Account for multiple outgoing connections.
            self.prev_socket = None;
        }
        Ok(poll_res.keep_going)
    }

    pub(super) fn handle_events(&mut self, cx: &mut Context, shared: &Shared) -> bool {
        for _ in 0..IO_LOOP_BOUND {
            let (ch, event) = match self.events.poll_recv(cx) {
                Poll::Ready(Some(x)) => x,
                Poll::Ready(None) => unreachable!("EndpointInner owns one sender"),
                Poll::Pending => {
                    return false;
                }
            };

            if event.is_drained() {
                self.recv_state.connections.senders.remove(&ch);
                if self.recv_state.connections.is_empty() {
                    shared.idle.notify_waiters();
                }
            }
            let Some(event) = self.inner.handle_event(ch, event) else {
                continue;
            };
            // Ignoring errors from dropped connections that haven't yet been cleaned up
            let _ = self
                .recv_state
                .connections
                .senders
                .get_mut(&ch)
                .unwrap()
                .enqueue(ConnectionEvent::Proto(event));
        }

        true
    }
}

impl Drop for State {
    fn drop(&mut self) {
        for incoming in self.recv_state.incoming.drain(..) {
            self.inner.ignore(incoming);
        }
    }
}

use super::*;

impl Future for EndpointDriver {
    type Output = Result<(), io::Error>;

    fn poll(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Self::Output> {
        let mut endpoint = self
            .0
            .state
            .lock()
            .map_err(|_| io::Error::other("QUIC endpoint state poisoned"))?;
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
        let mut endpoint = match self.0.state.lock() {
            Ok(endpoint) => endpoint,
            // Driver teardown only marks closure, removes senders and wakes waiters; it does not
            // resume the possibly inconsistent protocol state after a panic.
            Err(poisoned) => poisoned.into_inner(),
        };
        endpoint.driver_lost = true;
        // Drop all outgoing channels, signaling the termination of the endpoint to the associated
        // connections.
        let senders = mem::take(&mut endpoint.recv_state.connections.senders);
        let retiring = mem::take(&mut endpoint.retiring_sockets);
        drop(endpoint);
        self.0.shared.incoming.notify_waiters();
        self.0.shared.idle.notify_waiters();
        drop(senders);
        drop(retiring);
    }
}

impl State {
    pub(super) fn drive_recv(&mut self, cx: &mut Context, now: Instant) -> Result<bool, io::Error> {
        let get_time = || self.runtime.now();
        self.recv_state.recv_limiter.start_cycle(get_time);
        if self.retired_poll_remaining == 0 {
            self.retired_poll_remaining = self.retiring_sockets.len();
        }
        let polls = self
            .retired_poll_remaining
            .min(MAX_RETIRED_SOCKET_POLLS_PER_ITERATION);
        let mut old_keep_going = false;
        for _ in 0..polls {
            let index = self.retired_poll_cursor % self.retiring_sockets.len();
            self.retired_poll_cursor = index + 1;
            self.retired_poll_remaining -= 1;
            let socket = Arc::clone(&self.retiring_sockets[index].socket);
            let retiring = &mut self.retiring_sockets;
            let result = self.recv_state.poll_socket(
                cx,
                &mut self.inner,
                &*socket,
                &*self.runtime,
                now,
                |handle| observe_connection_on(retiring, index, handle),
            );
            match result {
                Ok(progress) => old_keep_going |= progress.keep_going,
                Err(_) => retiring[index].pending.clear(),
            }
        }
        let retiring = &mut self.retiring_sockets;
        let current_generation = retiring.len();
        let poll_res = self.recv_state.poll_socket(
            cx,
            &mut self.inner,
            &*self.socket,
            &*self.runtime,
            now,
            |handle| observe_connection_on(retiring, current_generation, handle),
        );
        self.recv_state.recv_limiter.finish_cycle(get_time);
        let poll_res = poll_res?;
        self.retiring_sockets.retain(|old| !old.pending.is_empty());
        self.retired_poll_remaining = self.retired_poll_remaining.min(self.retiring_sockets.len());
        if self.retiring_sockets.is_empty() {
            self.retired_poll_cursor = 0;
        } else {
            self.retired_poll_cursor %= self.retiring_sockets.len();
        }
        Ok(poll_res.keep_going || old_keep_going || self.retired_poll_remaining > 0)
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
                let generation = self.retiring_sockets.len();
                observe_connection_on(&mut self.retiring_sockets, generation, ch);
                self.retiring_sockets.retain(|old| !old.pending.is_empty());
                self.retired_poll_remaining =
                    self.retired_poll_remaining.min(self.retiring_sockets.len());
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
                .dispatch_if_live(ch, ConnectionEvent::Proto(event));
        }

        true
    }
}

/// Poll at most two historical sockets alongside the current socket in one driver iteration.
const MAX_RETIRED_SOCKET_POLLS_PER_ITERATION: usize = 2;

impl Drop for State {
    fn drop(&mut self) {
        for incoming in self.recv_state.incoming.drain(..) {
            self.inner.ignore(incoming);
        }
    }
}

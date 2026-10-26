use super::*;

impl RecvState {
    pub(super) fn new(
        sender: mpsc::Sender<(ConnectionHandle, EndpointEvent), mpsc::Unbounded>,
        max_receive_segments: usize,
        endpoint: &proto::Endpoint,
    ) -> Self {
        let recv_buf = vec![
            0;
            endpoint.config().get_max_udp_payload_size().min(64 * 1024) as usize
                * max_receive_segments
                * BATCH_SIZE
        ];
        Self {
            connections: ConnectionSet {
                senders: FxHashMap::default(),
                sender,
                close: None,
            },
            incoming: VecDeque::new(),
            recv_buf: recv_buf.into(),
            recv_limiter: WorkLimiter::new(RECV_TIME_BOUND),
        }
    }

    pub(super) fn poll_socket<F: FnMut(ConnectionHandle)>(
        &mut self,
        cx: &mut Context,
        endpoint: &mut proto::Endpoint,
        socket: &dyn AsyncUdpSocket,
        runtime: &dyn Runtime,
        now: Instant,
        mut on_connection_packet: F,
    ) -> Result<PollProgress, io::Error> {
        let mut metas = [RecvMeta::default(); BATCH_SIZE];
        let mut iovs: [IoSliceMut; BATCH_SIZE] = {
            let mut bufs = self
                .recv_buf
                .chunks_mut(self.recv_buf.len() / BATCH_SIZE)
                .map(IoSliceMut::new);

            // expect() safe as self.recv_buf is chunked into BATCH_SIZE items
            // and iovs will be of size BATCH_SIZE, thus from_fn is called
            // exactly BATCH_SIZE times.
            std::array::from_fn(|_| bufs.next().expect("BATCH_SIZE elements"))
        };
        loop {
            match socket.poll_recv(cx, &mut iovs, &mut metas) {
                Poll::Ready(Ok(msgs)) => {
                    self.recv_limiter.record_work(msgs);
                    for (meta, buf) in metas.iter().zip(iovs.iter()).take(msgs) {
                        let mut data: BytesMut = buf[0..meta.len].into();
                        while !data.is_empty() {
                            let buf = data.split_to(meta.stride.min(data.len()));
                            let mut response_buffer = Vec::new();
                            match endpoint.handle(
                                now,
                                meta.addr,
                                meta.dst_ip,
                                meta.ecn.map(proto_ecn),
                                buf,
                                &mut response_buffer,
                            ) {
                                Some(DatagramEvent::NewConnection(incoming)) => {
                                    if self.connections.close.is_none() {
                                        self.incoming.push_back(incoming);
                                    } else {
                                        let transmit =
                                            endpoint.refuse(incoming, &mut response_buffer);
                                        respond(transmit, &response_buffer, socket);
                                    }
                                }
                                Some(DatagramEvent::ConnectionEvent(handle, event)) => {
                                    // Ignoring errors from dropped connections that haven't yet been cleaned up
                                    on_connection_packet(handle);
                                    let _ = self
                                        .connections
                                        .dispatch_if_live(handle, ConnectionEvent::Proto(event));
                                }
                                Some(DatagramEvent::Response(transmit)) => {
                                    respond(transmit, &response_buffer, socket);
                                }
                                None => {}
                            }
                        }
                    }
                }
                Poll::Pending => {
                    return Ok(PollProgress { keep_going: false });
                }
                // Ignore ECONNRESET as it's undefined in QUIC and may be injected by an
                // attacker
                Poll::Ready(Err(ref e)) if e.kind() == io::ErrorKind::ConnectionReset => {
                    continue;
                }
                Poll::Ready(Err(e)) => {
                    return Err(e);
                }
            }
            if !self.recv_limiter.allow_work(|| runtime.now()) {
                return Ok(PollProgress { keep_going: true });
            }
        }
    }
}

impl fmt::Debug for RecvState {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("RecvState")
            .field("incoming", &self.incoming)
            .field("connections", &self.connections)
            // recv_buf too large
            .field("recv_limiter", &self.recv_limiter)
            .finish_non_exhaustive()
    }
}

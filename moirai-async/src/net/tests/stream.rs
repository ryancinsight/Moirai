use super::*;

#[test]
fn test_tcp_loopback_read_write_and_stats_values() {
    futures::executor::block_on(async {
        let listener = TcpListener::bind_with_config("127.0.0.1:0", test_config())
            .await
            .expect("listener bind must succeed");
        let addr = listener.local_addr().expect("listener address must exist");

        let client = std::thread::spawn(move || {
            let mut stream = StdTcpStream::connect(addr).expect("client connection must succeed");
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .expect("client read timeout must be set");
            stream
                .set_write_timeout(Some(Duration::from_secs(2)))
                .expect("client write timeout must be set");
            stream
                .write_all(b"ping")
                .expect("client write must succeed");

            let mut echo = [0_u8; 4];
            stream
                .read_exact(&mut echo)
                .expect("client echo read must succeed");
            assert_eq!(&echo, b"pong");
        });

        let deadline = Instant::now() + Duration::from_secs(2);
        let (mut stream, _peer) = accept_before(&listener, deadline).await;

        let mut inbound = [0_u8; 4];
        let bytes_read = read_before(&mut stream, &mut inbound, deadline).await;
        assert_eq!(bytes_read, 4);
        assert_eq!(&inbound, b"ping");

        let bytes_written = write_all_before(&mut stream, b"pong", deadline).await;
        assert_eq!(bytes_written, 4);
        stream.flush().await.expect("server flush must succeed");

        client.join().expect("client thread must complete");

        let stats = listener.stats();
        assert_eq!(stats.total_connections, 1);
        assert_eq!(stats.active_connections, 1);
        assert_eq!(stats.bytes_received, 4);
        assert_eq!(stats.bytes_sent, 4);
    });
}

#[test]
fn test_tcp_shutdown_write_sends_eof_and_stats_values() {
    futures::executor::block_on(async {
        let listener = TcpListener::bind_with_config("127.0.0.1:0", test_config())
            .await
            .expect("listener bind must succeed");
        let addr = listener.local_addr().expect("listener address must exist");

        let client = std::thread::spawn(move || {
            let mut stream = StdTcpStream::connect(addr).expect("client connection must succeed");
            stream
                .set_read_timeout(Some(Duration::from_secs(2)))
                .expect("client read timeout must be set");
            let mut received = Vec::new();
            stream
                .read_to_end(&mut received)
                .expect("client must observe EOF after shutdown");
            assert_eq!(&received, b"closed");
        });

        let deadline = Instant::now() + Duration::from_secs(2);
        let (mut stream, _peer) = accept_before(&listener, deadline).await;
        AsyncWriteExt::write_all(&mut stream, b"closed")
            .await
            .expect("server write_all must succeed");
        AsyncWriteExt::shutdown(&mut stream)
            .await
            .expect("server shutdown must succeed");

        client.join().expect("client thread must complete");

        let stats = listener.stats();
        assert_eq!(stats.total_connections, 1);
        assert_eq!(stats.bytes_sent, 6);
    });
}

#[test]
fn test_tcp_poll_write_reports_pending_under_backpressure() {
    let (listener, addr) = backpressure_listener();
    let (release_sender, release_receiver) = std::sync::mpsc::channel();

    let server = std::thread::spawn(move || {
        let (mut stream, _peer) = listener.accept().expect("server accept must succeed");
        release_receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("client must release backpressure server");
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("server read timeout must be set");

        let mut drained = 0;
        let mut buf = [0_u8; 8 * 1024];
        loop {
            match stream.read(&mut buf) {
                Ok(0) => break,
                Ok(count) => drained += count,
                Err(error)
                    if matches!(
                        error.kind(),
                        io::ErrorKind::WouldBlock
                            | io::ErrorKind::TimedOut
                            | io::ErrorKind::ConnectionReset
                    ) =>
                {
                    break;
                }
                Err(error) => panic!("server drain failed: {error}"),
            }
        }
        drained
    });

    let client = backpressure_client(addr);
    let mut stream = TcpStream::from_std(client).expect("moirai stream must wrap std stream");
    let written = poll_write_until_pending(&mut stream);
    assert!(written > 0);
    assert!(written <= BACKPRESSURE_MAX_BYTES);

    release_sender
        .send(())
        .expect("release signal must reach server");
    drop(stream);

    let drained = server.join().expect("server thread must join");
    assert!(drained > 0);
}

#[test]
fn test_tcp_poll_read_reports_pending_before_peer_data() {
    let (listener, addr) = readiness_listener();
    let (release_sender, release_receiver) = std::sync::mpsc::channel();

    let server = std::thread::spawn(move || {
        let (mut stream, _peer) = listener.accept().expect("server accept must succeed");
        stream
            .set_nodelay(true)
            .expect("server nodelay must be set");
        release_receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("client must release readiness server");
        stream
            .write_all(&READINESS_PAYLOAD)
            .expect("server payload write must succeed");
        stream.flush().expect("server flush must succeed");
    });

    let client = readiness_client(addr);
    let mut stream = TcpStream::from_std(client).expect("moirai stream must wrap std stream");
    let mut buf = [0_u8; READINESS_PAYLOAD.len()];

    let waker = futures::task::noop_waker();
    let mut context = Context::from_waker(&waker);
    let pending = Pin::new(&mut stream).poll_read(&mut context, &mut buf);
    assert!(matches!(pending, Poll::Pending));

    release_sender
        .send(())
        .expect("release signal must reach server");
    let received = poll_read_until_ready(&mut stream, &mut buf);
    assert_eq!(received, READINESS_PAYLOAD.len());
    assert_eq!(buf, READINESS_PAYLOAD);

    server.join().expect("server thread must join");
}

#[test]
fn test_tcp_pending_read_future_drop_preserves_stream_payload() {
    let (listener, addr) = readiness_listener();
    let (release_sender, release_receiver) = std::sync::mpsc::channel();

    let server = std::thread::spawn(move || {
        let (mut stream, _peer) = listener.accept().expect("server accept must succeed");
        stream
            .set_nodelay(true)
            .expect("server nodelay must be set");
        release_receiver
            .recv_timeout(Duration::from_secs(2))
            .expect("client must release readiness server");
        stream
            .write_all(&READINESS_PAYLOAD)
            .expect("server payload write must succeed");
        stream.flush().expect("server flush must succeed");
    });

    let client = readiness_client(addr);
    let mut stream = TcpStream::from_std(client).expect("moirai stream must wrap std stream");
    let mut cancelled_buf = [0_u8; READINESS_PAYLOAD.len()];

    {
        let mut future = std::pin::pin!(stream.read_exact(&mut cancelled_buf));
        let waker = futures::task::noop_waker();
        let mut context = Context::from_waker(&waker);
        assert!(matches!(
            Future::poll(future.as_mut(), &mut context),
            Poll::Pending
        ));
    }

    assert_eq!(cancelled_buf, [0_u8; READINESS_PAYLOAD.len()]);

    release_sender
        .send(())
        .expect("release signal must reach server");
    let mut buf = [0_u8; READINESS_PAYLOAD.len()];
    let received = poll_read_until_ready(&mut stream, &mut buf);
    assert_eq!(received, READINESS_PAYLOAD.len());
    assert_eq!(buf, READINESS_PAYLOAD);

    server.join().expect("server thread must join");
}

#[test]
fn timeout_read_cancellation_preserves_stream_payload() {
    let (listener, addr) = readiness_listener();
    let peer = StdTcpStream::connect(addr).expect("peer socket must connect");
    peer.set_nodelay(true).expect("peer nodelay must be set");
    let (accepted, _peer_addr) = listener.accept().expect("server accept must succeed");
    accepted
        .set_nodelay(true)
        .expect("accepted nodelay must be set");

    let executor = Arc::new(AsyncExecutor::new().expect("async executor must start"));
    let runner_executor = Arc::clone(&executor);
    let runner = std::thread::spawn(move || runner_executor.run());

    let handle = executor.spawn(async move {
        let mut stream = TcpStream::from_std(accepted).expect("accepted socket must wrap");
        let mut timed_out_buf = [0_u8; READINESS_PAYLOAD.len()];

        let result = timeout(Duration::from_millis(25), stream.read(&mut timed_out_buf)).await;

        assert_eq!(timed_out_buf, [0_u8; READINESS_PAYLOAD.len()]);
        assert!(
            matches!(result, Err(TimeoutError)),
            "socket read must time out before peer payload"
        );

        stream
    });

    let mut stream = futures::executor::block_on(handle);
    (&peer)
        .write_all(&READINESS_PAYLOAD)
        .expect("peer write must succeed");
    (&peer).flush().expect("peer flush must succeed");

    let mut received = [0_u8; READINESS_PAYLOAD.len()];
    let bytes = futures::executor::block_on(stream.read(&mut received))
        .expect("returned stream must still read peer payload");
    assert_eq!(bytes, READINESS_PAYLOAD.len());
    assert_eq!(received, READINESS_PAYLOAD);

    executor.stop().expect("executor stop must wake reactor");
    runner
        .join()
        .expect("executor thread must not panic after read cancellation")
        .expect("executor run must stop cleanly");
}

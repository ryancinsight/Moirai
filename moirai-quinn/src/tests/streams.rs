use super::*;

#[test]
fn read_after_close() {
    let _guard = subscribe();
    let runtime = rt_basic();
    let endpoint = {
        let _guard = runtime.enter();
        endpoint()
    };

    const MSG: &[u8] = b"goodbye!";
    let endpoint2 = endpoint.clone();
    runtime.spawn(async move {
        let new_conn = endpoint2
            .accept()
            .await
            .expect("endpoint")
            .await
            .expect("connection");
        let mut s = new_conn.open_uni().await.unwrap();
        s.write_all(MSG).await.unwrap();
        s.finish().unwrap();
        // Wait for the stream to be closed, one way or another.
        _ = s.stopped().await;
    });
    runtime.block_on(async move {
        let new_conn = endpoint
            .connect(endpoint.local_addr().unwrap(), "localhost")
            .unwrap()
            .await
            .expect("connect");
        sleep(Duration::from_millis(100)).await;
        let mut stream = new_conn.accept_uni().await.expect("incoming streams");
        let msg = stream.read_to_end(usize::MAX).await.expect("read_to_end");
        assert_eq!(msg, MSG);
    });
}

#[tokio::test]
async fn stream_id_flow_control() {
    let _guard = subscribe();
    let mut cfg = TransportConfig::default();
    cfg.max_concurrent_uni_streams(1u32.into());
    let endpoint = endpoint_with_config(cfg);

    let (client, server) = tokio::join!(
        endpoint
            .connect(endpoint.local_addr().unwrap(), "localhost")
            .unwrap(),
        async { endpoint.accept().await.unwrap().await }
    );
    let client = client.unwrap();
    let server = server.unwrap();

    // If `open_uni` doesn't get unblocked when the previous stream is dropped, this will time out.
    tokio::join!(
        async {
            client.open_uni().await.unwrap();
        },
        async {
            client.open_uni().await.unwrap();
        },
        async {
            client.open_uni().await.unwrap();
        },
        async {
            server.accept_uni().await.unwrap();
            server.accept_uni().await.unwrap();
        }
    );
}

#[tokio::test]
async fn stream_stopped() {
    let _guard = subscribe();
    let factory = EndpointFactory::new();
    let server = {
        let _guard = error_span!("server").entered();
        factory.endpoint()
    };
    let server_addr = server.local_addr().unwrap();

    let client = {
        let _guard = error_span!("client1").entered();
        factory.endpoint()
    };

    let client = async move {
        let conn = client
            .connect(server_addr, "localhost")
            .unwrap()
            .await
            .unwrap();
        let mut stream = conn.open_uni().await.unwrap();
        let stopped1 = stream.stopped();
        let stopped2 = stream.stopped();
        let stopped3 = stream.stopped();

        stream.write_all(b"hi").await.unwrap();
        // spawn one of the futures into a task
        let stopped1 = tokio::task::spawn(stopped1);
        // verify that both futures resolved
        let (stopped1, stopped2) = tokio::join!(stopped1, stopped2);
        assert!(matches!(stopped1, Ok(Ok(Some(val))) if val == 42u32.into()));
        assert!(matches!(stopped2, Ok(Some(val)) if val == 42u32.into()));
        // drop the stream
        drop(stream);
        // verify that a future also resolves after dropping the stream
        let stopped3 = stopped3.await;
        assert_eq!(stopped3, Ok(Some(42u32.into())));
    };
    let client = timeout(Duration::from_millis(100), client).instrument(error_span!("client"));
    let server = async move {
        let conn = server.accept().await.unwrap().await.unwrap();
        let mut stream = conn.accept_uni().await.unwrap();
        let mut buf = [0u8; 2];
        stream.read_exact(&mut buf).await.unwrap();
        stream.stop(42u32.into()).unwrap();
        conn
    }
    .instrument(error_span!("server"));
    let (client, conn) = tokio::join!(client, server);
    client.expect("timeout");
    drop(conn);
}

#[tokio::test]
async fn stream_stopped_2() {
    let _guard = subscribe();
    let endpoint = endpoint();

    let (conn, _server_conn) = tokio::try_join!(
        endpoint
            .connect(endpoint.local_addr().unwrap(), "localhost")
            .unwrap(),
        async { endpoint.accept().await.unwrap().await }
    )
    .unwrap();
    let send_stream = conn.open_uni().await.unwrap();
    let stopped = timeout(Duration::from_millis(100), send_stream.stopped())
        .instrument(error_span!("stopped"));
    tokio::pin!(stopped);
    // poll the future once so that the waker is registered.
    tokio::select! {
        biased;
        _x = &mut stopped => {},
        _x = std::future::ready(()) => {}
    }
    // drop the send stream
    drop(send_stream);
    // make sure the stopped future still resolves
    let res = stopped.await;
    assert_eq!(res, Ok(Ok(None)));
}

#[tokio::test]
async fn stream_drop_removes_blocked_reader() {
    let _guard = subscribe();

    for drop_stream in [false, true] {
        let endpoint_factory = EndpointFactory::new();
        let server = endpoint_factory.endpoint();
        let server_address = server.local_addr().unwrap();
        let client = endpoint_factory.endpoint();

        let server_task = tokio::spawn(async move {
            let conn = server.accept().await.unwrap().await.unwrap();
            let mut stream = conn.accept_uni().await.unwrap();

            // read "hello"
            let mut buf = [0u8; 5];
            stream.read_exact(&mut buf).await.unwrap();

            let (waker, wake_counter) = new_count_waker();
            let mut cx = Context::from_waker(&waker);
            // do a blocking read which will add the stream in conn.blocked_readers
            {
                let mut buf = [0u8; 64];
                let read_fut = stream.read(&mut buf);
                tokio::pin!(read_fut);
                assert!(matches!(read_fut.as_mut().poll(&mut cx), Poll::Pending));
            }

            if !drop_stream {
                assert_eq!(wake_counter.wakes(), 0);
                // We have a blocked reader, closing the connection should wake it. We use this as
                // a proxy to assert that the stream is in conn.blocked_readers.
                conn.close(0u32.into(), b"done");
                assert_eq!(wake_counter.wakes(), 1);
            } else {
                // dropping the stream should remove it from conn.blocked_readers, so we don't
                // expect any wakeups
                drop(stream);
                assert_eq!(wake_counter.wakes(), 0, "no wakeups should have occurred");
                conn.close(0u32.into(), b"done");
                assert_eq!(wake_counter.wakes(), 0, "no wakeups should have occurred");
            }
        });

        let conn = client
            .connect(server_address, "localhost")
            .unwrap()
            .await
            .unwrap();
        let mut stream = conn.open_uni().await.unwrap();
        // need to send some data to actually start the stream
        stream.write_all(b"hello").await.unwrap();

        server_task.await.unwrap();
    }
}

/// Test that dropping a `RecvStream` after cancelling a read and then
/// explicitly `stop`ing it doesn't panic.
#[tokio::test]
async fn recv_stream_cancel_stop_drop() {
    let _guard = subscribe();
    let factory = EndpointFactory::new();
    let server = {
        let _guard = error_span!("server").entered();
        factory.endpoint()
    };
    let server_addr = server.local_addr().unwrap();

    let client = {
        let _guard = error_span!("client").entered();
        factory.endpoint()
    };
    let recv_dropped = tokio::sync::SetOnce::new();
    join!(
        async {
            let conn = server.accept().await.unwrap().await.unwrap();
            let mut recv = conn.accept_uni().await.unwrap();
            // Create a future to read from the stream, poll it once, then immediately drop it
            {
                let fut = pin!(recv.read_to_end(usize::MAX));
                let mut cx = Context::from_waker(Waker::noop());
                assert!(fut.poll(&mut cx).is_pending());
            }
            recv_dropped.set(()).unwrap();
            recv.stop(0u32.into()).unwrap();
        },
        async {
            let conn = client
                .connect(server_addr, "localhost")
                .unwrap()
                .await
                .unwrap();
            let mut send = conn.open_uni().await.unwrap();
            _ = send.write_all(b"hello").await;
            // Don't drop (finish) the send stream until the read has been
            // cancelled by the server, ensuring that read_to_end can't complete
            // immediately.
            recv_dropped.wait().await;
        },
    );
}

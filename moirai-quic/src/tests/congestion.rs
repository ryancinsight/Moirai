use super::*;

#[test]
fn selected_controllers_preserve_authenticated_bytes_and_acknowledged_fin() {
    let _trace = trace();
    let executor = executor();
    let original = config(65_560, 16, DEADLINE);
    assert_eq!(original.congestion_policy(), CongestionPolicy::Cubic);
    for policy in [
        CongestionPolicy::Cubic,
        CongestionPolicy::NewReno,
        CongestionPolicy::ExperimentalBbr,
    ] {
        let _arrange = tracing::info_span!("arrange", ?policy).entered();
        let selected = original.clone().with_congestion_policy(policy);
        assert_eq!(selected.congestion_policy(), policy);
        assert_eq!(selected.max_frame_bytes(), original.max_frame_bytes());
        assert_eq!(selected.max_streams(), original.max_streams());
        assert_eq!(selected.operation_timeout(), original.operation_timeout());
        assert_eq!(selected.alpn(), original.alpn());
        assert_eq!(selected.udp_buffers(), original.udp_buffers());
        let (identity, roots) = material();
        let server = QuicEndpoint::server(
            Arc::clone(&executor),
            "127.0.0.1:0".parse().expect("loopback"),
            identity,
            Some(roots.clone()),
            selected.clone(),
        )
        .expect("mutual TLS server");
        let (identity, _) = material();
        let client = QuicEndpoint::client(
            Arc::clone(&executor),
            "127.0.0.1:0".parse().expect("loopback"),
            roots,
            Some(identity),
            selected,
        )
        .expect("mutual TLS client");
        let (server_connection, client_connection) = pair(&server, &client);
        drop(_arrange);
        let _act = tracing::info_span!("act").entered();
        let (received, echoed) = moirai_executor::block_on(futures::future::join(
            async {
                let mut stream = server_connection.accept_stream().await.expect("accept");
                let message = stream
                    .recv_message()
                    .await
                    .expect("receive")
                    .expect("message");
                assert_eq!(stream.recv_message().await.expect("client FIN"), None);
                stream.send_message(&message).await.expect("echo");
                stream.finish_and_wait().await.expect("server FIN ACK");
                message
            },
            async {
                let mut stream = client_connection.open_stream().await.expect("open");
                stream.send_message(&payload(65_560)).await.expect("send");
                stream.finish().expect("client FIN");
                let echo = stream.recv_message().await.expect("echo").expect("message");
                assert_eq!(stream.recv_message().await.expect("server FIN"), None);
                stream.finish_and_wait().await.expect("client FIN ACK");
                echo
            },
        ));
        drop(_act);
        let _assert = tracing::info_span!("assert").entered();
        assert_eq!(received, payload(65_560));
        assert_eq!(echoed, received);
        for connection in [&server_connection, &client_connection] {
            assert_eq!(connection.alpn(), ALPN);
            assert_eq!(
                connection
                    .peer_certificates()
                    .expect("authenticated chain")
                    .len(),
                3
            );
        }
        server.close();
        client.close();
        moirai_executor::block_on(server.wait_idle()).expect("server drain");
        moirai_executor::block_on(client.wait_idle()).expect("client drain");
        drop(server_connection);
        drop(client_connection);
        drop(server);
        drop(client);
    }
    moirai_core::executor::ExecutorControl::shutdown(executor.as_ref());
}

use super::*;
use moirai_pal::net::{UdpBufferConfig, UdpBufferSizes, UdpConfig};
use std::num::NonZeroUsize;

#[test]
fn configured_queues_preserve_authenticated_messages_and_acknowledged_fin() {
    let _trace = trace();
    let _test = tracing::info_span!("quic_configured_udp").entered();
    let executor = executor();
    let buffers = UdpBufferConfig {
        receive: NonZeroUsize::new(65_536),
        send: NonZeroUsize::new(32_768),
    };
    let (server, client, expected) = tracing::info_span!("arrange").in_scope(|| {
        let config = config(4096, 2, DEADLINE)
            .with_udp_buffers(buffers)
            .expect("validated queues");
        assert_eq!(config.udp_buffers(), buffers);
        assert_eq!(config.max_frame_bytes(), 4096);
        assert_eq!(config.max_streams(), 2);
        assert_eq!(config.operation_timeout(), DEADLINE);
        let reference = UdpConfig {
            buffers,
            ..UdpConfig::default()
        }
        .bind("127.0.0.1:0".parse().expect("address"))
        .expect("native configured reference");
        let expected = UdpBufferSizes::read(&reference).expect("native effective queues");
        let (server, client) = endpoints(&executor, &config);
        (server, client, expected)
    });
    assert_eq!(server.udp_buffer_sizes(), expected);
    assert_eq!(client.udp_buffer_sizes(), expected);
    let (server_connection, client_connection) = pair(&server, &client);
    let messages = [0, 1, 37, 4096].map(payload);
    let (sent, received) = moirai_executor::block_on(futures::future::join(
        async {
            let mut stream = client_connection
                .open_stream()
                .await
                .expect("authenticated stream");
            for message in &messages {
                stream.send_message(message).await.expect("complete send");
            }
            stream.finish_and_wait().await.expect("acknowledged FIN");
            stream.recv_message().await.expect("peer FIN")
        },
        async {
            let mut stream = server_connection
                .accept_stream()
                .await
                .expect("accepted stream");
            let mut actual = Vec::new();
            while let Some(message) = stream.recv_message().await.expect("complete receive") {
                actual.push(message);
            }
            stream
                .finish_and_wait()
                .await
                .expect("peer acknowledged FIN");
            actual
        },
    ));
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(sent, None);
        assert_eq!(received, messages);
        assert_eq!(client_connection.alpn(), ALPN);
        assert_eq!(
            client_connection
                .peer_certificates()
                .expect("trusted server")
                .len(),
            3
        );
        assert_eq!(server.udp_buffer_sizes(), expected);
        assert_eq!(client.udp_buffer_sizes(), expected);
    });
    server.close();
    client.close();
    let (server, client) = moirai_executor::block_on(futures::future::join(
        server.wait_idle(),
        client.wait_idle(),
    ));
    server.expect("server drained");
    client.expect("client drained");
}

#[test]
fn invalid_native_queue_request_preserves_io_error_context() {
    let _trace = trace();
    let oversized = usize::try_from(i32::MAX).expect("native signed maximum") + 1;
    for buffers in [
        UdpBufferConfig {
            receive: NonZeroUsize::new(oversized),
            send: None,
        },
        UdpBufferConfig {
            receive: None,
            send: NonZeroUsize::new(oversized),
        },
    ] {
        let error = config(4096, 2, DEADLINE)
            .with_udp_buffers(buffers)
            .expect_err("reject native overflow");
        match error {
            QuicError::Io(error) => {
                assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
                assert_eq!(
                    error.to_string(),
                    "UDP buffer exceeds native signed integer"
                );
            }
            other => panic!("native error context lost: {other}"),
        }
    }
}

use super::*;
use moirai_core::executor::TaskSpawner;
use std::future::Future;
use std::task::{Context, Poll};
use tracing::Instrument;

#[test]
fn upstream_peer_observes_exact_frame_bytes_and_eof() {
    let _trace = trace();
    let _arrange = tracing::info_span!("arrange").entered();
    let executor = executor();
    let config = config(65_536, 4, DEADLINE);
    let (server, client) = endpoints(&executor, &config);
    let (server_connection, client_connection) = pair(&server, &client);
    assert_eq!(server_connection.peer_certificates(), None);
    drop(_arrange);
    let receiver = executor
        .spawn_async(
            async move {
                let (_send, mut recv) = server_connection
                    .connection
                    .accept_bi()
                    .await
                    .expect("upstream stream");
                let mut messages = Vec::new();
                for length in [0, 1, 37, 65_536] {
                    let mut prefix = [0; 4];
                    recv.read_exact(&mut prefix).await.expect("wire prefix");
                    assert_eq!(u32::from_be_bytes(prefix), length);
                    let mut bytes = vec![0; length as usize];
                    recv.read_exact(&mut bytes).await.expect("wire bytes");
                    messages.push(bytes);
                }
                let mut byte = [0; 1];
                assert_eq!(recv.read(&mut byte).await.expect("clean EOF"), None);
                messages
            }
            .instrument(tracing::info_span!("act", peer = "upstream Quinn")),
        )
        .expect("receiver task");
    moirai_executor::block_on(async {
        let mut stream = client_connection.open_stream().await.expect("open stream");
        for length in [0, 1, 37, 65_536] {
            stream
                .send_message(&payload(length))
                .await
                .expect("send bounded frame");
        }
        stream.finish().expect("FIN");
    });
    let messages = receiver
        .join()
        .expect("task attached")
        .expect("computed messages");
    let _assert = tracing::info_span!("assert").entered();
    assert_eq!(messages, [0, 1, 37, 65_536].map(payload));
    assert_eq!(client_connection.alpn(), ALPN);
    assert_eq!(
        client_connection
            .peer_certificates()
            .expect("authenticated server chain")
            .len(),
        3
    );
}

#[test]
fn stalled_and_cancelled_stream_leaves_sibling_progress_intact() {
    let _trace = trace();
    let executor = executor();
    let (server, client) = endpoints(&executor, &config(4096, 2, DEADLINE));
    let (server_connection, client_connection) = pair(&server, &client);
    moirai_executor::block_on(async {
        let _act = tracing::info_span!("act").entered();
        let mut stalled = client_connection
            .open_stream()
            .await
            .expect("stalled stream");
        stalled
            .send
            .write_all(&4096_u32.to_be_bytes())
            .await
            .expect("prefix");
        stalled
            .send
            .write_all(&[0x37])
            .await
            .expect("partial payload");
        let mut receiving = server_connection
            .accept_stream()
            .await
            .expect("accept stalled");
        let mut read = Box::pin(receiving.recv_message());
        let waker = futures::task::noop_waker();
        assert!(matches!(
            read.as_mut().poll(&mut Context::from_waker(&waker)),
            Poll::Pending
        ));
        let mut sibling = client_connection
            .open_stream()
            .await
            .expect("sibling stream");
        let expected = payload(137);
        sibling.send_message(&expected).await.expect("sibling send");
        let mut sibling_peer = server_connection
            .accept_stream()
            .await
            .expect("sibling accept");
        assert_ne!(sibling.id(), stalled.id());
        assert_eq!(
            sibling_peer.recv_message().await.expect("sibling receive"),
            Some(expected.clone())
        );
        drop(read); // Cancellation resets only the partial stream.
        assert!(matches!(
            receiving.recv_message().await,
            Err(QuicError::Closed)
        ));
        sibling_peer
            .send_message(&expected)
            .await
            .expect("sibling echo");
        sibling_peer.finish().expect("sibling peer FIN");
        assert_eq!(
            sibling.recv_message().await.expect("sibling response"),
            Some(expected)
        );
        assert_eq!(sibling.recv_message().await.expect("sibling EOF"), None);
        sibling.finish().expect("sibling FIN");
    });
}

#[test]
fn exact_bound_round_trip_and_local_oversize_preserve_framing() {
    let _trace = trace();
    let executor = executor();
    let (server, client) = endpoints(&executor, &config(37, 1, DEADLINE));
    let (server_connection, client_connection) = pair(&server, &client);
    let server_session = server_connection.clone();
    let receiver = executor
        .spawn_async(
            async move {
                let mut stream = server_session.accept_stream().await.expect("accept");
                let message = stream
                    .recv_message()
                    .await
                    .expect("maximum receive")
                    .expect("message");
                stream.send_message(&message).await.expect("echo");
                stream.finish().expect("server FIN");
                message
            }
            .instrument(tracing::info_span!("act", side = "server")),
        )
        .expect("server task");
    let response = moirai_executor::block_on(async {
        let mut stream = client_connection.open_stream().await.expect("stream");
        assert!(matches!(
            stream.send_message(&payload(38)).await,
            Err(QuicError::MessageTooLarge {
                size: 38,
                limit: 37
            })
        ));
        stream
            .send_message(&payload(37))
            .await
            .expect("exact bound send");
        stream.finish().expect("client FIN");
        stream
            .recv_message()
            .await
            .expect("echo receive")
            .expect("echo bytes")
    });
    assert_eq!(response, payload(37));
    assert_eq!(
        receiver.join().expect("attached").expect("completed"),
        response
    );
}

#[test]
fn property_messages_preserve_arbitrary_bytes_across_real_udp() {
    use proptest::prelude::*;
    let _trace = trace();
    let executor = executor();
    let (server, client) = endpoints(&executor, &config(65_536, 2, DEADLINE));
    let (server_connection, client_connection) = pair(&server, &client);
    let mut runner = proptest::test_runner::TestRunner::default();
    runner
        .run(
            &proptest::collection::vec(any::<u8>(), 0..=65_536),
            |input| {
                let (received, echoed) = moirai_executor::block_on(futures::future::join(
                    async {
                        let mut stream = server_connection
                            .accept_stream()
                            .await
                            .expect("server stream");
                        let bytes = stream
                            .recv_message()
                            .await
                            .expect("server frame")
                            .expect("server message");
                        stream.send_message(&bytes).await.expect("server echo");
                        stream.finish().expect("server FIN");
                        bytes
                    },
                    async {
                        let mut stream = client_connection
                            .open_stream()
                            .await
                            .expect("client stream");
                        stream.send_message(&input).await.expect("client frame");
                        stream.finish().expect("client FIN");
                        stream
                            .recv_message()
                            .await
                            .expect("client frame")
                            .expect("client message")
                    },
                ));
                prop_assert_eq!(&received, &input);
                prop_assert_eq!(&echoed, &input);
                Ok(())
            },
        )
        .expect("arbitrary payloads preserve exact input bytes");
}

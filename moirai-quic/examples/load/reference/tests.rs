use super::error::Result as ReferenceResult;
use super::{Error, exchange, framing, material};
use crate::load::{Plan, report};
use std::net::SocketAddr;

fn runtime() -> tokio::runtime::Runtime {
    tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .expect("invariant: the host supports the Tokio test runtime")
}

fn loopback() -> SocketAddr {
    "127.0.0.1:0"
        .parse()
        .expect("invariant: the loopback literal is a socket address")
}

async fn endpoint_pair(
    plan: Plan,
    server_alpn: &[u8],
    client_alpn: &[u8],
) -> (quinn::Endpoint, quinn::Endpoint, SocketAddr) {
    let server = material::endpoint_with_alpn::<true>(loopback(), plan, server_alpn)
        .expect("invariant: fixture-backed server configuration is valid");
    let address = server
        .local_addr()
        .expect("invariant: a bound server has a local address");
    let client = material::endpoint_with_alpn::<false>(loopback(), plan, client_alpn)
        .expect("invariant: fixture-backed client configuration is valid");
    (server, client, address)
}

async fn connections(
    server: &quinn::Endpoint,
    client: &quinn::Endpoint,
    address: SocketAddr,
) -> (
    ReferenceResult<quinn::Connection>,
    ReferenceResult<quinn::Connection>,
) {
    futures::future::join(
        exchange::accept(server),
        exchange::connect(client, address, "localhost"),
    )
    .await
}

async fn authenticated_connections(
    plan: Plan,
) -> (
    quinn::Endpoint,
    quinn::Endpoint,
    quinn::Connection,
    quinn::Connection,
) {
    let (server, client, address) = endpoint_pair(plan, material::ALPN, material::ALPN).await;
    let (server_connection, client_connection) = connections(&server, &client, address).await;
    let server_connection = server_connection.expect("mutual-TLS server connection");
    let client_connection = client_connection.expect("mutual-TLS client connection");
    material::authorize(&server_connection).expect("authenticated client identity and ALPN");
    material::authorize(&client_connection).expect("authenticated server identity and ALPN");
    (server, client, server_connection, client_connection)
}

async fn streams(
    server: &quinn::Connection,
    client: &quinn::Connection,
) -> (
    (quinn::SendStream, quinn::RecvStream),
    (quinn::SendStream, quinn::RecvStream),
) {
    let mut client_stream = client
        .open_bi()
        .await
        .expect("client opens a bidirectional stream");
    client_stream
        .0
        .write_all(&0_u32.to_be_bytes())
        .await
        .expect("client activation frame reaches the wire");
    let server_stream = server
        .accept_bi()
        .await
        .expect("client-opened stream reaches server");
    (server_stream, client_stream)
}

async fn shutdown(server: quinn::Endpoint, client: quinn::Endpoint) {
    server.close(0_u8.into(), b"test complete");
    client.close(0_u8.into(), b"test complete");
    tokio::time::timeout(
        material::OPERATION_TIMEOUT,
        futures::future::join(server.wait_idle(), client.wait_idle()),
    )
    .await
    .expect("both test endpoints drain within the operation deadline");
}

#[test]
fn loopback_exchange_authenticates_frames_fin_and_report_values() {
    runtime().block_on(async {
        let plan = Plan::new(2, 3, 32).expect("positive bounded workload");
        let (server, client, address) = endpoint_pair(plan, material::ALPN, material::ALPN).await;
        let started = std::time::Instant::now();
        let (server_result, client_result) = futures::future::try_join(
            exchange::run::<true>(&server, address, plan, started),
            exchange::run::<false>(&client, address, plan, started),
        )
        .await
        .expect("authenticated reference exchange");
        let server_report = serde_json::to_value(
            report::<true>(plan, server_result).expect("server report contract"),
        )
        .expect("server report serialization");
        let client_report = serde_json::to_value(
            report::<false>(plan, client_result).expect("client report contract"),
        )
        .expect("client report serialization");
        for (report, responder) in [(server_report, true), (client_report, false)] {
            assert_eq!(report["schema"], 1);
            assert_eq!(report["responder"], responder);
            assert_eq!(report["streams"], 2);
            assert_eq!(report["messages_per_stream"], 3);
            assert_eq!(report["frame_bytes"], 32);
            assert_eq!(report["verified_payload_bytes_per_direction"], 192);
            assert_eq!(
                report["samples_ns"]
                    .as_array()
                    .expect("samples are an array")
                    .len(),
                6
            );
        }
        shutdown(server, client).await;
    });
}

#[test]
fn wrong_server_identity_is_a_typed_handshake_error() {
    runtime().block_on(async {
        let plan = Plan::new(1, 1, 16).expect("positive bounded workload");
        let (server, client, address) = endpoint_pair(plan, material::ALPN, material::ALPN).await;
        let (server_result, client_result) = futures::future::join(
            exchange::accept(&server),
            exchange::connect(&client, address, "wrong.invalid"),
        )
        .await;
        assert!(matches!(client_result, Err(Error::Connection(_))));
        assert!(matches!(server_result, Err(Error::Connection(_))));
        shutdown(server, client).await;
    });
}

#[test]
fn incompatible_alpn_is_a_typed_handshake_error() {
    runtime().block_on(async {
        let plan = Plan::new(1, 1, 16).expect("positive bounded workload");
        let (server, client, address) = endpoint_pair(plan, b"other/1", material::ALPN).await;
        let (server_result, client_result) = connections(&server, &client, address).await;
        assert!(matches!(client_result, Err(Error::Connection(_))));
        assert!(matches!(server_result, Err(Error::Connection(_))));
        shutdown(server, client).await;
    });
}

#[test]
fn corrupted_and_extra_frames_are_typed_protocol_errors() {
    runtime().block_on(async {
        let plan = Plan::new(1, 1, 16).expect("positive bounded workload");
        let (server, client, server_connection, client_connection) =
            authenticated_connections(plan).await;
        let ((mut server_send, server_recv), (client_send, client_recv)) =
            streams(&server_connection, &client_connection).await;
        let first = [7_u8; 16];
        let extra = [9_u8; 16];
        let mut client_stream = framing::Stream::new(client_send, client_recv, plan.bytes);
        let send_frames = async {
            server_send
                .write_all(&16_u32.to_be_bytes())
                .await
                .expect("first frame prefix");
            server_send.write_all(&first).await.expect("first frame");
            server_send
                .write_all(&16_u32.to_be_bytes())
                .await
                .expect("extra frame prefix");
            server_send.write_all(&extra).await.expect("extra frame");
            server_send.finish().expect("server FIN");
        };
        let receive_frames = async {
            let received = client_stream
                .recv_message()
                .await
                .expect("complete corrupted frame")
                .expect("frame precedes FIN");
            assert!(matches!(
                exchange::verify_payload(&received, &[3_u8; 16]),
                Err(Error::PayloadMismatch)
            ));
            let extra = client_stream
                .recv_message()
                .await
                .expect("complete extra frame");
            assert!(matches!(
                exchange::reject_extra(extra),
                Err(Error::ExtraData)
            ));
        };
        futures::future::join(send_frames, receive_frames).await;
        drop(server_send);
        drop(server_recv);
        drop(client_stream);
        drop(server_connection);
        drop(client_connection);
        shutdown(server, client).await;
    });
}

#[test]
fn truncated_frame_is_a_typed_exact_read_error() {
    runtime().block_on(async {
        let plan = Plan::new(1, 1, 32).expect("positive bounded workload");
        let (server, client, server_connection, client_connection) =
            authenticated_connections(plan).await;
        let ((mut server_send, server_recv), (client_send, client_recv)) =
            streams(&server_connection, &client_connection).await;
        server_send
            .write_all(&32_u32.to_be_bytes())
            .await
            .expect("truncated frame prefix");
        server_send
            .write_all(&[5_u8; 4])
            .await
            .expect("truncated frame body");
        server_send.finish().expect("server FIN");
        let mut client_stream = framing::Stream::new(client_send, client_recv, plan.bytes);
        let error = client_stream
            .recv_message()
            .await
            .expect_err("truncated frame must fail");
        assert!(matches!(
            error,
            Error::ReadExact(quinn::ReadExactError::FinishedEarly(4))
        ));
        drop(server_send);
        drop(server_recv);
        drop(client_stream);
        drop(server_connection);
        drop(client_connection);
        shutdown(server, client).await;
    });
}

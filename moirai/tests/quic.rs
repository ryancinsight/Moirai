#![cfg(all(feature = "quic", any(unix, windows)))]

use moirai::quic::{Identity, QuicConfig, QuicEndpoint, rustls};
use rustls::pki_types::{CertificateDer, PrivateKeyDer, pem::PemObject};
use std::sync::Arc;
use std::time::Duration;

#[test]
fn facade_runtime_drives_quic_without_a_second_scheduler() {
    let runtime = Arc::new(
        moirai::Moirai::builder()
            .worker_threads(2)
            .build()
            .expect("facade runtime"),
    );
    let certificates =
        CertificateDer::pem_slice_iter(include_bytes!("../../tests/fixtures/localhost-cert.pem"))
            .collect::<Result<Vec<_>, _>>()
            .expect("published certificate fixture");
    let mut roots = rustls::RootCertStore::empty();
    roots
        .add(certificates.last().expect("fixture root").clone())
        .expect("trust root");
    let key =
        PrivateKeyDer::from_pem_slice(include_bytes!("../../tests/fixtures/localhost-key.pem"))
            .expect("fixture key");
    let config = QuicConfig::new(
        b"facade-test/1",
        37,
        1,
        1,
        65_536,
        Duration::from_secs(2),
        Duration::from_secs(10),
    )
    .expect("bounded config");
    let server = QuicEndpoint::server(
        Arc::clone(&runtime),
        "127.0.0.1:0".parse().expect("loopback"),
        Identity::new(certificates, key),
        None,
        config.clone(),
    )
    .expect("server");
    let client = QuicEndpoint::client(
        Arc::clone(&runtime),
        "127.0.0.1:0".parse().expect("loopback"),
        roots,
        None,
        config,
    )
    .expect("client");
    let (server_connection, client_connection) = runtime.block_on(futures::future::join(
        server.accept(),
        client.connect(server.local_addr().expect("server address"), "localhost"),
    ));
    let server_connection = server_connection.expect("server handshake");
    let client_connection = client_connection.expect("client handshake");
    let receiving = server_connection.clone();
    let incoming = runtime.spawn_async(async move {
        let mut stream = receiving.accept_stream().await.expect("stream");
        let message = stream
            .recv_message()
            .await
            .expect("message")
            .expect("bytes");
        stream.send_message(&message).await.expect("echo");
        stream.finish_and_wait().await.expect("acknowledged output");
        message
    });
    let expected: Vec<u8> = (0..37).map(|index| index * 7 % 251).collect();
    let actual = runtime.block_on(async {
        let mut stream = client_connection
            .open_stream()
            .await
            .expect("client stream");
        stream.send_message(&expected).await.expect("input bytes");
        stream.finish().expect("client FIN");
        let bytes = stream
            .recv_message()
            .await
            .expect("echo")
            .expect("echo bytes");
        assert_eq!(stream.recv_message().await.expect("server FIN"), None);
        stream
            .finish_and_wait()
            .await
            .expect("client FIN acknowledged");
        bytes
    });
    assert_eq!(actual, expected);
    assert_eq!(
        incoming.join().expect("attached").expect("computed result"),
        expected
    );
    server.close();
    client.close();
    runtime.block_on(server.wait_idle()).expect("server drains");
    runtime.block_on(client.wait_idle()).expect("client drains");
}

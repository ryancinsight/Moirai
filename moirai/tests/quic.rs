#![cfg(any(unix, windows))]

use moirai::quic::{Identity, QuicConfig, QuicEndpoint, rustls};
use moirai_core::executor::TaskSpawner;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, pem::PemObject};
use std::sync::Arc;
use std::time::Duration;

fn endpoints() -> (Arc<moirai::Moirai>, QuicEndpoint, QuicEndpoint) {
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
    (runtime, server, client)
}

#[test]
fn facade_runtime_drives_quic_without_a_second_scheduler() {
    let _tracing = tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_test_writer()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .with_max_level(tracing::Level::TRACE)
            .finish(),
    );
    let (runtime, server, client) = {
        let _arrange = tracing::info_span!("arrange").entered();
        endpoints()
    };
    let act = tracing::info_span!("act").entered();
    let (server_connection, client_connection) = runtime.block_on(futures::future::join(
        server.accept(),
        client.connect(server.local_addr().expect("server address"), "localhost"),
    ));
    let server_connection = server_connection.expect("server handshake");
    let client_connection = client_connection.expect("client handshake");
    let receiving = server_connection.clone();
    let incoming = TaskSpawner::spawn_async(runtime.as_ref(), async move {
        let mut stream = receiving.accept_stream().await.expect("stream");
        let message = stream
            .recv_message()
            .await
            .expect("message")
            .expect("bytes");
        stream.send_message(&message).await.expect("echo");
        stream.finish_and_wait().await.expect("acknowledged output");
        message
    })
    .expect("facade scheduler admits receiver");
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
    let received = incoming.join().expect("attached").expect("computed result");
    drop(act);
    {
        let _assert = tracing::info_span!("assert").entered();
        assert_eq!(server_connection.alpn(), b"facade-test/1");
        assert_eq!(client_connection.alpn(), b"facade-test/1");
        assert_eq!(actual, expected);
        assert_eq!(received, expected);
    }
    let _teardown = tracing::info_span!("teardown").entered();
    server.close();
    client.close();
    runtime.block_on(server.wait_idle()).expect("server drains");
    runtime.block_on(client.wait_idle()).expect("client drains");
}

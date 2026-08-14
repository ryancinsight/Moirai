//! ADR-015 P1 verification: a real TLS 1.2/1.3 handshake + encrypted round-trip
//! between a `rustls` server and the `moirai-tls` client, both driven over Moirai
//! async sockets on the unified runtime. Plus the adversarial path: a client that
//! does NOT trust the server cert must fail the handshake (fail-closed).

use std::sync::Arc;

use moirai_async::io::{AsyncReadExt, AsyncWriteExt};
use moirai_async::net::{TcpListener, TcpStream};
use moirai_tls::rustls::ServerConfig;
use moirai_tls::{client_config_with_roots, ServerName, TlsConnector, ToFuturesIo, ToMoiraiIo};
use rustls_pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer};

/// Self-signed cert + key for "localhost" and a matching `rustls` server acceptor.
fn server_acceptor() -> (CertificateDer<'static>, futures_rustls::TlsAcceptor) {
    let ck = rcgen::generate_simple_self_signed(vec!["localhost".to_string()])
        .expect("self-signed cert generation");
    let cert_der = ck.cert.der().clone();
    let key_der = PrivateKeyDer::Pkcs8(PrivatePkcs8KeyDer::from(ck.key_pair.serialize_der()));

    let config = ServerConfig::builder_with_provider(Arc::new(moirai_crypto::provider()))
        .with_safe_default_protocol_versions()
        .expect("safe default protocol versions")
        .with_no_client_auth()
        .with_single_cert(vec![cert_der.clone()], key_der)
        .expect("server single cert");

    (
        cert_der,
        futures_rustls::TlsAcceptor::from(Arc::new(config)),
    )
}

#[test]
fn tls_round_trip_trusted_cert() {
    let rt = moirai::global();
    let (cert_der, acceptor) = server_acceptor();

    let listener = rt.block_on(TcpListener::bind("127.0.0.1:0")).expect("bind");
    let addr = listener.local_addr().expect("addr");

    // Server: accept, TLS-accept, echo a 5-byte plaintext frame.
    let server = rt.spawn_async(async move {
        let (sock, _peer) = listener.accept().await.expect("accept");
        let tls = acceptor
            .accept(ToFuturesIo(sock))
            .await
            .expect("server handshake");
        let mut stream = ToMoiraiIo(tls);
        let mut buf = [0u8; 5];
        stream.read_exact(&mut buf).await.expect("server read");
        stream.write_all(&buf).await.expect("server echo");
        stream.flush().await.expect("server flush");
        buf
    });

    // Client trusts the self-signed cert as a root.
    let mut roots = moirai_tls::rustls::RootCertStore::empty();
    roots.add(cert_der).expect("add trusted root");
    let connector = TlsConnector::from_config(Arc::new(client_config_with_roots(roots)));

    let echo = rt.block_on(async move {
        let sock = TcpStream::connect(&addr.to_string())
            .await
            .expect("connect");
        let domain = ServerName::try_from("localhost").expect("server name");
        let mut tls = connector
            .connect(domain, sock)
            .await
            .expect("client handshake");
        tls.write_all(b"hello").await.expect("client write");
        let mut echo = [0u8; 5];
        tls.read_exact(&mut echo).await.expect("client read echo");
        echo
    });

    assert_eq!(
        &echo, b"hello",
        "encrypted round-trip must preserve the frame"
    );
    assert_eq!(server.join().expect("server task"), Ok(*b"hello"));
}

#[test]
fn tls_handshake_fails_closed_for_untrusted_cert() {
    let rt = moirai::global();
    let (_server_cert, acceptor) = server_acceptor();

    let listener = rt.block_on(TcpListener::bind("127.0.0.1:0")).expect("bind");
    let addr = listener.local_addr().expect("addr");

    let server = rt.spawn_async(async move {
        let (sock, _peer) = listener.accept().await.expect("accept");
        // Handshake is expected to fail on the client's untrusted-root rejection.
        acceptor.accept(ToFuturesIo(sock)).await.map(|_| ())
    });

    // Client trusts only the unrelated Mozilla roots, NOT the server's self-signed cert.
    let connector = TlsConnector::with_webpki_roots();
    let result: std::io::Result<()> = rt.block_on(async move {
        let sock = TcpStream::connect(&addr.to_string())
            .await
            .expect("connect");
        let domain = ServerName::try_from("localhost").expect("server name");
        connector.connect(domain, sock).await.map(|_| ())
    });

    assert!(
        result.is_err(),
        "client must reject the untrusted self-signed certificate (fail closed)"
    );
    let _ = server.join();
}

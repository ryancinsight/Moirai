//! ADR-015 P1 verification: a real TLS 1.2/1.3 handshake + encrypted round-trip
//! between a `rustls` server and the `moirai-tls` client, both driven over Moirai
//! async sockets on the unified runtime. Plus the adversarial path: a client that
//! does NOT trust the server cert must fail the handshake (fail-closed).

use std::io;
use std::sync::Arc;
use std::time::Duration;

use moirai_async::io::{AsyncReadExt, AsyncWriteExt};
use moirai_async::net::{TcpListener, TcpStream};
use moirai_async::timer::timeout;
use moirai_tls::rustls::{CertificateError, Error as RustlsError, ServerConfig};
use moirai_tls::{client_config_with_roots, ServerName, TlsConnector, ToFuturesIo, ToMoiraiIo};
use rustls_pki_types::{pem::PemObject, CertificateDer, PrivateKeyDer};

const REJECTED_HANDSHAKE_DEADLINE: Duration = Duration::from_secs(5);

fn runtime() -> moirai::Moirai {
    moirai::Moirai::new().expect("test runtime")
}

fn certificate_error(error: &io::Error) -> &CertificateError {
    let rustls_error = error
        .get_ref()
        .and_then(|source| source.downcast_ref::<RustlsError>())
        .expect("TLS I/O error must preserve the rustls source");
    let RustlsError::InvalidCertificate(certificate_error) = rustls_error else {
        panic!("expected certificate validation failure, got {rustls_error:?}");
    };
    certificate_error
}

fn assert_rejected_server_terminated(server: moirai::TaskHandle<io::Result<()>>) {
    let result = server
        .join()
        .expect("server task remains attached")
        .expect("server task must not panic");
    if let Err(error) = result {
        assert_ne!(
            error.kind(),
            io::ErrorKind::TimedOut,
            "server-side rejected handshake must terminate before its deadline"
        );
    }
}

async fn accept_rejected_handshake(
    listener: TcpListener,
    acceptor: futures_rustls::TlsAcceptor,
) -> io::Result<()> {
    let (socket, _peer) = listener.accept().await?;
    match timeout(
        REJECTED_HANDSHAKE_DEADLINE,
        acceptor.accept(ToFuturesIo(socket)),
    )
    .await
    {
        Ok(result) => result.map(|_| ()),
        Err(error) => Err(io::Error::new(io::ErrorKind::TimedOut, error)),
    }
}

fn fixture_acceptor<const CHAIN_LEN: usize>(
    certificate_pem: &[u8],
    private_key_pem: &[u8],
) -> (CertificateDer<'static>, futures_rustls::TlsAcceptor) {
    let certs = CertificateDer::pem_slice_iter(certificate_pem)
        .collect::<Result<Vec<_>, _>>()
        .expect("certificate fixture parses");
    assert_eq!(certs.len(), CHAIN_LEN, "fixture certificate count");
    let root_der = certs.last().expect("root certificate exists").clone();

    let key_der =
        PrivateKeyDer::from_pem_slice(private_key_pem).expect("private-key fixture parses");

    let config = ServerConfig::builder_with_provider(Arc::new(moirai_crypto::provider()))
        .with_safe_default_protocol_versions()
        .expect("safe default protocol versions")
        .with_no_client_auth()
        .with_single_cert(certs, key_der)
        .expect("server single cert");

    (
        root_der,
        futures_rustls::TlsAcceptor::from(Arc::new(config)),
    )
}

/// Deterministic localhost certificate + key and a matching `rustls` server acceptor.
fn server_acceptor() -> (CertificateDer<'static>, futures_rustls::TlsAcceptor) {
    fixture_acceptor::<3>(
        include_bytes!("../../tests/fixtures/localhost-cert.pem"),
        include_bytes!("../../tests/fixtures/localhost-key.pem"),
    )
}

/// An acceptor presenting a chain whose leaf is permanently expired (2024-01-01
/// to 2024-12-31) under a valid root, so only the lifetime check can fail.
fn expired_server_acceptor() -> (CertificateDer<'static>, futures_rustls::TlsAcceptor) {
    fixture_acceptor::<2>(
        include_bytes!("../../tests/fixtures/expired-cert.pem"),
        include_bytes!("../../tests/fixtures/expired-key.pem"),
    )
}

#[test]
fn tls_round_trip_trusted_cert() {
    let rt = runtime();
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

    // Client trusts the fixture root CA.
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
    let rt = runtime();
    let (_server_cert, acceptor) = server_acceptor();

    let listener = rt.block_on(TcpListener::bind("127.0.0.1:0")).expect("bind");
    let addr = listener.local_addr().expect("addr");

    let server = rt.spawn_async(accept_rejected_handshake(listener, acceptor));

    // Client trusts only the unrelated Mozilla roots, NOT the fixture root.
    let connector = TlsConnector::with_webpki_roots();
    let result: std::io::Result<()> = rt.block_on(async move {
        let sock = TcpStream::connect(&addr.to_string())
            .await
            .expect("connect");
        let domain = ServerName::try_from("localhost").expect("server name");
        connector.connect(domain, sock).await.map(|_| ())
    });

    let error = result.expect_err("client must reject the untrusted certificate");
    let certificate_error = certificate_error(&error);
    assert!(
        matches!(certificate_error, CertificateError::UnknownIssuer),
        "untrusted chain must produce UnknownIssuer, got {certificate_error:?}"
    );
    assert_rejected_server_terminated(server);
}

/// A client that trusts the fixture root CA but connects for a name the
/// certificate does not cover must reject it with `NotValidForName` — the
/// value-semantic check that hostname validation is actually wired into the
/// socket-level client, not just the pure-Rust provider.
#[test]
fn tls_handshake_fails_closed_for_wrong_hostname() {
    let rt = runtime();
    let (cert_der, acceptor) = server_acceptor();

    let listener = rt.block_on(TcpListener::bind("127.0.0.1:0")).expect("bind");
    let addr = listener.local_addr().expect("addr");

    let server = rt.spawn_async(accept_rejected_handshake(listener, acceptor));

    // Client trusts the fixture root CA but connects for a name the
    // certificate does not cover.
    let mut roots = moirai_tls::rustls::RootCertStore::empty();
    roots.add(cert_der).expect("add trusted root");
    let connector = TlsConnector::from_config(Arc::new(client_config_with_roots(roots)));

    let result: std::io::Result<()> = rt.block_on(async move {
        let sock = TcpStream::connect(&addr.to_string())
            .await
            .expect("connect");
        let domain = ServerName::try_from("wrong.example").expect("server name");
        connector.connect(domain, sock).await.map(|_| ())
    });

    let error = result.expect_err("wrong hostname must fail the handshake");
    let certificate_error = certificate_error(&error);
    assert!(
        matches!(
            certificate_error,
            CertificateError::NotValidForName | CertificateError::NotValidForNameContext { .. }
        ),
        "wrong hostname must produce NotValidForName, got {certificate_error:?}"
    );
    assert_rejected_server_terminated(server);
}

/// A client that trusts the (valid) root but receives a leaf that expired in
/// 2024 must reject the handshake with `Expired` — the value-semantic check
/// that certificate lifetime validation is wired into the socket-level client,
/// not just present in the pure-Rust provider.
#[test]
fn tls_handshake_fails_closed_for_expired_cert() {
    let rt = runtime();
    let (root_der, acceptor) = expired_server_acceptor();

    let listener = rt.block_on(TcpListener::bind("127.0.0.1:0")).expect("bind");
    let addr = listener.local_addr().expect("addr");

    let server = rt.spawn_async(accept_rejected_handshake(listener, acceptor));

    // Client trusts the fixture root CA; only the leaf's lifetime is invalid.
    let mut roots = moirai_tls::rustls::RootCertStore::empty();
    roots.add(root_der).expect("add trusted root");
    let connector = TlsConnector::from_config(Arc::new(client_config_with_roots(roots)));

    let result: std::io::Result<()> = rt.block_on(async move {
        let sock = TcpStream::connect(&addr.to_string())
            .await
            .expect("connect");
        let domain = ServerName::try_from("localhost").expect("server name");
        connector.connect(domain, sock).await.map(|_| ())
    });

    let error = result.expect_err("expired certificate must fail the handshake");
    let certificate_error = certificate_error(&error);
    assert!(
        matches!(
            certificate_error,
            CertificateError::Expired | CertificateError::ExpiredContext { .. }
        ),
        "expired certificate must produce Expired, got {certificate_error:?}"
    );
    assert_rejected_server_terminated(server);
}

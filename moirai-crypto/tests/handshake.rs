//! End-to-end validation of the pure-Rust provider: a full TLS 1.3 handshake
//! and encrypted application-data round-trip driven entirely in memory, using
//! `moirai_crypto::provider()` on **both** the client and server sides.
//!
//! This exercises key exchange (X25519/P-256), the AEAD suites, HKDF key
//! schedule, certificate signature verification, and software signing.

use std::io::{Read, Write};
use std::sync::Arc;

use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName};
use rustls::{ClientConfig, ClientConnection, RootCertStore, ServerConfig, ServerConnection};

fn certificate_error(error: rustls::Error) -> rustls::CertificateError {
    let rustls::Error::InvalidCertificate(certificate_error) = error else {
        panic!("expected certificate validation failure, got {error:?}");
    };
    certificate_error
}

/// Pump handshake/data bytes from `from` into `to`, surfacing any
/// `process_new_packets` error (e.g. certificate verification failure).
fn transfer(
    from: &mut rustls::Connection,
    to: &mut rustls::Connection,
) -> Result<(), rustls::Error> {
    let mut buf = [0u8; 16 * 1024];
    while from.wants_write() {
        let n = from.write_tls(&mut &mut buf[..]).unwrap();
        if n == 0 {
            break;
        }
        let mut offset = 0;
        while offset < n {
            offset += to.read_tls(&mut &buf[offset..n]).unwrap();
            to.process_new_packets()?;
        }
    }
    Ok(())
}

fn test_material() -> (Vec<CertificateDer<'static>>, PrivateKeyDer<'static>) {
    let mut cert_pem = &include_bytes!("../../tests/fixtures/localhost-cert.pem")[..];
    let certs = rustls_pemfile::certs(&mut cert_pem)
        .collect::<Result<Vec<_>, _>>()
        .expect("certificate fixture parses");
    assert_eq!(
        certs.len(),
        3,
        "fixture contains leaf, intermediate, and root"
    );

    let mut key_pem = &include_bytes!("../../tests/fixtures/localhost-key.pem")[..];
    let key_der = rustls_pemfile::private_key(&mut key_pem)
        .expect("private-key fixture parses")
        .expect("private-key fixture exists");

    (certs, key_der)
}

fn round_trip_with(server_name: &'static str) {
    let (certs, key_der) = test_material();
    let root_der = certs.last().expect("root certificate exists").clone();

    let provider = Arc::new(moirai_crypto::provider());

    let server_config = ServerConfig::builder_with_provider(provider.clone())
        .with_safe_default_protocol_versions()
        .expect("server protocol versions")
        .with_no_client_auth()
        .with_single_cert(certs.clone(), key_der)
        .expect("server single cert");

    let mut roots = RootCertStore::empty();
    roots.add(root_der).expect("trust fixture root");
    let client_config = ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .expect("client protocol versions")
        .with_root_certificates(roots)
        .with_no_client_auth();

    let mut client: rustls::Connection = ClientConnection::new(
        Arc::new(client_config),
        ServerName::try_from(server_name).unwrap(),
    )
    .expect("client connection")
    .into();
    let mut server: rustls::Connection = ServerConnection::new(Arc::new(server_config))
        .expect("server connection")
        .into();

    // Drive the handshake to completion.
    for _ in 0..16 {
        transfer(&mut client, &mut server).expect("client->server");
        transfer(&mut server, &mut client).expect("server->client");
        if !client.is_handshaking() && !server.is_handshaking() {
            break;
        }
    }

    assert!(
        !client.is_handshaking() && !server.is_handshaking(),
        "handshake must complete"
    );

    // TLS 1.3 must have been negotiated.
    assert_eq!(
        client.protocol_version(),
        Some(rustls::ProtocolVersion::TLSv1_3)
    );

    // Client -> server encrypted application data.
    client.writer().write_all(b"ping").unwrap();
    transfer(&mut client, &mut server).expect("client->server data");
    let mut got = Vec::new();
    server.reader().read_to_end(&mut got).ok();
    assert_eq!(&got, b"ping", "server must decrypt client data");

    // Server -> client encrypted application data.
    server.writer().write_all(b"pong").unwrap();
    transfer(&mut server, &mut client).expect("server->client data");
    let mut got = Vec::new();
    client.reader().read_to_end(&mut got).ok();
    assert_eq!(&got, b"pong", "client must decrypt server data");
}

#[test]
fn tls13_in_memory_round_trip() {
    round_trip_with("localhost");
}

/// A client using the wrong server name must fail certificate validation.
#[test]
fn wrong_server_name_is_rejected() {
    let (certs, key_der) = test_material();
    let root_der = certs.last().expect("root certificate exists").clone();

    let provider = Arc::new(moirai_crypto::provider());
    let server_config = ServerConfig::builder_with_provider(provider.clone())
        .with_safe_default_protocol_versions()
        .expect("server protocol versions")
        .with_no_client_auth()
        .with_single_cert(certs, key_der)
        .expect("server single cert");

    let mut roots = RootCertStore::empty();
    roots.add(root_der).expect("trust fixture root");
    let client_config = ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .expect("client protocol versions")
        .with_root_certificates(roots)
        .with_no_client_auth();

    let mut client: rustls::Connection = ClientConnection::new(
        Arc::new(client_config),
        ServerName::try_from("wrong.example").unwrap(),
    )
    .expect("client connection")
    .into();
    let mut server: rustls::Connection = ServerConnection::new(Arc::new(server_config))
        .expect("server connection")
        .into();

    let mut client_err = None;
    for _ in 0..16 {
        if let Err(error) = transfer(&mut client, &mut server) {
            client_err = Some(error);
            break;
        }
        if let Err(error) = transfer(&mut server, &mut client) {
            client_err = Some(error);
            break;
        }
        if !client.is_handshaking() && !server.is_handshaking() {
            break;
        }
    }

    let error = certificate_error(client_err.expect("wrong hostname must fail the handshake"));
    assert!(
        matches!(
            error,
            rustls::CertificateError::NotValidForName
                | rustls::CertificateError::NotValidForNameContext { .. }
        ),
        "wrong hostname must produce NotValidForName, got {error:?}"
    );
}

#[test]
fn untrusted_cert_is_rejected() {
    let (certs, key_der) = test_material();

    let provider = Arc::new(moirai_crypto::provider());

    let server_config = ServerConfig::builder_with_provider(provider.clone())
        .with_safe_default_protocol_versions()
        .expect("server protocol versions")
        .with_no_client_auth()
        .with_single_cert(certs, key_der)
        .expect("server single cert");

    // Client trusts an *empty* root store, so the server cert is untrusted.
    let client_config = ClientConfig::builder_with_provider(provider)
        .with_safe_default_protocol_versions()
        .expect("client protocol versions")
        .with_root_certificates(RootCertStore::empty())
        .with_no_client_auth();

    let mut client: rustls::Connection = ClientConnection::new(
        Arc::new(client_config),
        ServerName::try_from("localhost").unwrap(),
    )
    .expect("client connection")
    .into();
    let mut server: rustls::Connection = ServerConnection::new(Arc::new(server_config))
        .expect("server connection")
        .into();

    // Drive the handshake; the client must surface a certificate error.
    let mut client_err = None;
    for _ in 0..16 {
        if let Err(e) = transfer(&mut client, &mut server) {
            // A fatal alert from the client also counts as rejection.
            client_err = Some(e);
            break;
        }
        if let Err(e) = transfer(&mut server, &mut client) {
            client_err = Some(e);
            break;
        }
        if !client.is_handshaking() && !server.is_handshaking() {
            break;
        }
    }

    let error = certificate_error(client_err.expect("untrusted certificate must fail handshake"));
    assert!(
        matches!(error, rustls::CertificateError::UnknownIssuer),
        "untrusted chain must produce UnknownIssuer, got {error:?}"
    );
}

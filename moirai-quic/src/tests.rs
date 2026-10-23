mod acceptance;
mod congestion;
mod loss;
mod rejection;
mod socket;

use super::*;
use moirai_executor::HybridExecutor;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, pem::PemObject};
use std::sync::Arc;
use std::time::Duration;

const DEADLINE: Duration = Duration::from_secs(2);
const IDLE: Duration = Duration::from_secs(10);
const ALPN: &[u8] = b"moirai-test/1";

fn executor() -> Arc<HybridExecutor> {
    Arc::new(
        moirai_executor::ExecutorBuilder::new()
            .worker_threads(2)
            .async_threads(1)
            .build()
            .expect("Moirai executor"),
    )
}

fn config(frame_bytes: u32, streams: u32, deadline: Duration) -> QuicConfig {
    QuicConfig::new(ALPN, frame_bytes, streams, 4, 65_536, deadline, IDLE)
        .expect("finite QUIC config")
}

fn material() -> (Identity, rustls::RootCertStore) {
    let certificates =
        CertificateDer::pem_slice_iter(include_bytes!("../../tests/fixtures/localhost-cert.pem"))
            .collect::<Result<Vec<_>, _>>()
            .expect("published certificate fixture");
    let mut roots = rustls::RootCertStore::empty();
    roots
        .add(certificates.last().expect("fixture root").clone())
        .expect("trusted root");
    let key =
        PrivateKeyDer::from_pem_slice(include_bytes!("../../tests/fixtures/localhost-key.pem"))
            .expect("published fixture key");
    (Identity::new(certificates, key), roots)
}

fn endpoints(executor: &Arc<HybridExecutor>, config: &QuicConfig) -> (QuicEndpoint, QuicEndpoint) {
    let (identity, roots) = material();
    let server = QuicEndpoint::server(
        Arc::clone(executor),
        "127.0.0.1:0".parse().expect("loopback"),
        identity,
        None,
        config.clone(),
    )
    .expect("server endpoint");
    let client = QuicEndpoint::client(
        Arc::clone(executor),
        "127.0.0.1:0".parse().expect("loopback"),
        roots,
        None,
        config.clone(),
    )
    .expect("client endpoint");
    (server, client)
}

fn pair(server: &QuicEndpoint, client: &QuicEndpoint) -> (QuicConnection, QuicConnection) {
    let address = server.local_addr().expect("bound server address");
    // Both handshakes must progress concurrently through real UDP.
    let (server, client) = moirai_executor::block_on(futures::future::join(
        server.accept(),
        client.connect(address, "localhost"),
    ));
    (
        server.expect("server handshake"),
        client.expect("client handshake"),
    )
}

fn trace() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_test_writer()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .with_max_level(tracing::Level::TRACE)
            .finish(),
    )
}

pub(crate) fn payload(length: usize) -> Vec<u8> {
    (0..length)
        .map(|index| u8::try_from((index * 37 + index / 251) % 251).expect("modular byte"))
        .collect()
}

//! Raw Quinn endpoint and mutual-TLS construction.

use super::super::{Plan, identity};
use super::error::{Error, Result};
use moirai_quic::rustls;
use quinn::crypto::rustls::{QuicClientConfig, QuicServerConfig};
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

pub(super) const ALPN: &[u8] = b"moirai-load/1";
pub(crate) const OPERATION_TIMEOUT: Duration = Duration::from_secs(5);

pub(crate) fn endpoint<const RESPONDER: bool>(
    address: SocketAddr,
    plan: Plan,
) -> Result<quinn::Endpoint> {
    let bind_address = if RESPONDER {
        address
    } else if address.is_ipv6() {
        "[::]:0"
            .parse()
            .expect("invariant: IPv6 wildcard literal is a socket address")
    } else {
        "0.0.0.0:0"
            .parse()
            .expect("invariant: IPv4 wildcard literal is a socket address")
    };
    endpoint_with_alpn::<RESPONDER>(bind_address, plan, ALPN)
}

pub(super) fn endpoint_with_alpn<const RESPONDER: bool>(
    address: SocketAddr,
    plan: Plan,
    alpn: &[u8],
) -> Result<quinn::Endpoint> {
    let transport = transport(plan)?;
    let server = if RESPONDER {
        Some(server_config(Arc::clone(&transport), alpn)?)
    } else {
        None
    };
    let socket = std::net::UdpSocket::bind(address).map_err(Error::Io)?;
    socket.set_nonblocking(true).map_err(Error::Io)?;
    let mut endpoint = quinn::Endpoint::new(
        quinn::EndpointConfig::default(),
        server,
        socket,
        Arc::new(quinn::TokioRuntime),
    )
    .map_err(Error::Io)?;
    if !RESPONDER {
        endpoint.set_default_client_config(client_config(transport, alpn)?);
    }
    Ok(endpoint)
}

fn transport(plan: Plan) -> Result<Arc<quinn::TransportConfig>> {
    let stream_window = u64::from(plan.bytes)
        .checked_add(4)
        .ok_or(Error::InvalidConfig("stream window overflows"))?;
    let connection_window = stream_window
        .checked_mul(u64::from(plan.streams))
        .ok_or(Error::InvalidConfig("connection window overflows"))?;
    let mut transport = quinn::TransportConfig::default();
    match plan.policy {
        moirai_quic::CongestionPolicy::Cubic => {
            transport
                .congestion_controller_factory(Arc::new(quinn::congestion::CubicConfig::default()));
        }
        moirai_quic::CongestionPolicy::NewReno => {
            transport.congestion_controller_factory(Arc::new(
                quinn::congestion::NewRenoConfig::default(),
            ));
        }
        moirai_quic::CongestionPolicy::ExperimentalBbr => {
            transport
                .congestion_controller_factory(Arc::new(quinn::congestion::BbrConfig::default()));
        }
    }
    transport.max_concurrent_bidi_streams(plan.streams.into());
    transport.max_concurrent_uni_streams(0_u8.into());
    transport.stream_receive_window(
        quinn::VarInt::from_u64(stream_window)
            .map_err(|_| Error::InvalidConfig("stream window exceeds QUIC varint"))?,
    );
    transport.receive_window(
        quinn::VarInt::from_u64(connection_window)
            .map_err(|_| Error::InvalidConfig("connection window exceeds QUIC varint"))?,
    );
    transport.send_window(connection_window);
    transport.crypto_buffer_size(65_536);
    transport.datagram_receive_buffer_size(None);
    transport.max_idle_timeout(Some(
        Duration::from_secs(60)
            .try_into()
            .map_err(|_| Error::InvalidConfig("idle timeout exceeds QUIC varint"))?,
    ));
    Ok(Arc::new(transport))
}

fn client_config(
    transport: Arc<quinn::TransportConfig>,
    alpn: &[u8],
) -> Result<quinn::ClientConfig> {
    let builder = rustls::ClientConfig::builder_with_provider(Arc::new(
        rustls::crypto::ring::default_provider(),
    ))
    .with_protocol_versions(&[&rustls::version::TLS13])
    .map_err(Error::Tls)?
    .with_root_certificates(identity::roots().map_err(Error::Fixture)?);
    let mut tls = builder
        .with_client_auth_cert(
            identity::certificates().map_err(Error::Fixture)?,
            identity::private_key().map_err(Error::Fixture)?,
        )
        .map_err(Error::Tls)?;
    tls.alpn_protocols = vec![alpn.to_vec()];
    tls.enable_early_data = false;
    let crypto = QuicClientConfig::try_from(tls)
        .map_err(|_| Error::InvalidConfig("QUIC Initial cipher suite is absent"))?;
    let mut client = quinn::ClientConfig::new(Arc::new(crypto));
    client.transport_config(transport);
    Ok(client)
}

fn server_config(
    transport: Arc<quinn::TransportConfig>,
    alpn: &[u8],
) -> Result<quinn::ServerConfig> {
    let provider = Arc::new(rustls::crypto::ring::default_provider());
    let verifier = rustls::server::WebPkiClientVerifier::builder_with_provider(
        Arc::new(identity::roots().map_err(Error::Fixture)?),
        Arc::clone(&provider),
    )
    .build()
    .map_err(|error| Error::Io(std::io::Error::new(std::io::ErrorKind::InvalidInput, error)))?;
    let mut tls = rustls::ServerConfig::builder_with_provider(provider)
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map_err(Error::Tls)?
        .with_client_cert_verifier(verifier)
        .with_single_cert(
            identity::certificates().map_err(Error::Fixture)?,
            identity::private_key().map_err(Error::Fixture)?,
        )
        .map_err(Error::Tls)?;
    tls.alpn_protocols = vec![alpn.to_vec()];
    tls.max_early_data_size = 0;
    let crypto = QuicServerConfig::try_from(tls)
        .map_err(|_| Error::InvalidConfig("QUIC Initial cipher suite is absent"))?;
    let mut server = quinn::ServerConfig::with_crypto(Arc::new(crypto));
    server.transport_config(transport);
    server.max_incoming(1);
    server.incoming_buffer_size(65_536);
    server.incoming_buffer_size_total(65_536);
    Ok(server)
}

pub(super) fn authorize(connection: &quinn::Connection) -> Result<()> {
    let identity = connection.peer_identity().ok_or(Error::PeerIdentity)?;
    let identity = identity
        .downcast::<Vec<rustls::pki_types::CertificateDer<'static>>>()
        .map_err(|_| Error::PeerIdentity)?;
    if identity.first() != Some(&identity::leaf().map_err(Error::Fixture)?) {
        return Err(Error::PeerIdentity);
    }
    let protocol = connection
        .handshake_data()
        .ok_or(Error::ApplicationProtocol)?
        .downcast::<quinn::crypto::rustls::HandshakeData>()
        .map_err(|_| Error::ApplicationProtocol)?
        .protocol
        .ok_or(Error::ApplicationProtocol)?;
    if protocol.as_slice() != ALPN {
        return Err(Error::ApplicationProtocol);
    }
    Ok(())
}

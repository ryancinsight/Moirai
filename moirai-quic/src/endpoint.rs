use super::admission::StreamAdmission;
use super::config::Identity;
use super::error::bounded;
use super::runtime::MoiraiRuntime;
use super::runtime::RuntimeHealth;
use super::{QuicConfig, QuicError, QuicStream, rustls};
use crate::lock_mutex;
use moirai_core::executor::TaskSpawner;
use rustls::pki_types::CertificateDer;
use std::net::{SocketAddr, UdpSocket};
use std::sync::{Arc, Mutex};

/// A bound native UDP endpoint with authenticated connection admission.
///
/// Keep the supplied Moirai executor live until endpoints and connections drain.
/// Dropping this endpoint closes its connections. Endpoint count is application
/// policy; per-endpoint connection and pending-handshake counts are bounded.
pub struct QuicEndpoint {
    pub(super) endpoint: quinn::Endpoint,
    config: QuicConfig,
    admission: Mutex<()>,
    health: Arc<RuntimeHealth>,
}

impl QuicEndpoint {
    /// Bind a client trusting exactly `roots`, with optional client identity.
    ///
    /// TLS authenticates the server name supplied to [`Self::connect`]. An
    /// identity allows mutual TLS where the server requires client certificates.
    /// All sessions negotiate the config's ALPN and disable early data.
    ///
    /// # Errors
    /// Returns TLS configuration, socket or protocol configuration errors.
    pub fn client<E: TaskSpawner + Send + Sync + 'static>(
        executor: Arc<E>,
        address: SocketAddr,
        roots: rustls::RootCertStore,
        identity: Option<Identity>,
        config: QuicConfig,
    ) -> Result<Self, QuicError> {
        let builder = rustls::ClientConfig::builder_with_provider(Arc::new(
            rustls::crypto::ring::default_provider(),
        ))
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map_err(QuicError::Tls)?
        .with_root_certificates(roots);
        let mut tls = match identity {
            Some(identity) => builder
                .with_client_auth_cert(identity.certificates, identity.key)
                .map_err(QuicError::Tls)?,
            None => builder.with_no_client_auth(),
        };
        tls.alpn_protocols = vec![config.alpn.to_vec()];
        tls.enable_early_data = false;
        let crypto = quinn::crypto::rustls::QuicClientConfig::try_from(tls)
            .map_err(|_| QuicError::InvalidConfig("QUIC Initial cipher suite is absent"))?;
        let mut client = quinn::ClientConfig::new(Arc::new(crypto));
        client.transport_config(config.transport()?);
        let mut endpoint = Self::bind(executor, address, None, config)?;
        endpoint.endpoint.set_default_client_config(client);
        Ok(endpoint)
    }

    /// Bind a server using `identity`; `client_roots` requires mutual TLS.
    ///
    /// `None` authenticates only the server, so application policy must
    /// authenticate/authorize each client independently. No remote address,
    /// stream ID or client-declared principal establishes identity.
    ///
    /// # Errors
    /// Returns key/certificate, trust-store, socket or configuration errors.
    pub fn server<E: TaskSpawner + Send + Sync + 'static>(
        executor: Arc<E>,
        address: SocketAddr,
        identity: Identity,
        client_roots: Option<rustls::RootCertStore>,
        config: QuicConfig,
    ) -> Result<Self, QuicError> {
        let provider = Arc::new(rustls::crypto::ring::default_provider());
        let builder = rustls::ServerConfig::builder_with_provider(Arc::clone(&provider))
            .with_protocol_versions(&[&rustls::version::TLS13])
            .map_err(QuicError::Tls)?;
        let builder = match client_roots {
            Some(roots) => {
                let verifier = rustls::server::WebPkiClientVerifier::builder_with_provider(
                    Arc::new(roots),
                    provider,
                )
                .build()
                .map_err(|error| {
                    QuicError::Io(std::io::Error::new(std::io::ErrorKind::InvalidInput, error))
                })?;
                builder.with_client_cert_verifier(verifier)
            }
            None => builder.with_no_client_auth(),
        };
        let mut tls = builder
            .with_single_cert(identity.certificates, identity.key)
            .map_err(QuicError::Tls)?;
        tls.alpn_protocols = vec![config.alpn.to_vec()];
        tls.max_early_data_size = 0;
        let crypto = quinn::crypto::rustls::QuicServerConfig::try_from(tls)
            .map_err(|_| QuicError::InvalidConfig("QUIC Initial cipher suite is absent"))?;
        let mut server = quinn::ServerConfig::with_crypto(Arc::new(crypto));
        server.transport_config(config.transport()?);
        server.max_incoming(config.max_connections as usize);
        server.incoming_buffer_size(u64::from(config.max_handshake_bytes));
        server.incoming_buffer_size_total(
            u64::from(config.max_handshake_bytes) * u64::from(config.max_connections),
        );
        Self::bind(executor, address, Some(server), config)
    }

    fn bind<E: TaskSpawner + Send + Sync + 'static>(
        executor: Arc<E>,
        address: SocketAddr,
        server: Option<quinn::ServerConfig>,
        config: QuicConfig,
    ) -> Result<Self, QuicError> {
        let socket = UdpSocket::bind(address).map_err(QuicError::Io)?;
        // Condition 1: Quinn's endpoint constructor requires an erased Runtime.
        let health = Arc::new(RuntimeHealth::default());
        let runtime = Arc::new(MoiraiRuntime {
            executor,
            health: Arc::clone(&health),
        });
        let endpoint =
            quinn::Endpoint::new(quinn::EndpointConfig::default(), server, socket, runtime)
                .map_err(QuicError::Io)?;
        health.check()?;
        Ok(Self {
            endpoint,
            config,
            admission: Mutex::new(()),
            health,
        })
    }

    /// Bound address, suitable for publication before clients connect.
    pub fn local_addr(&self) -> Result<SocketAddr, QuicError> {
        self.endpoint.local_addr().map_err(QuicError::Io)
    }

    /// Connect with server-name validation within one finite handshake deadline.
    ///
    /// # Errors
    /// Returns full admission, invalid address/name, timeout or TLS errors.
    pub async fn connect(
        &self,
        address: SocketAddr,
        server_name: &str,
    ) -> Result<QuicConnection, QuicError> {
        let connecting = {
            let _guard = lock_mutex(&self.admission);
            self.check_capacity()?;
            self.endpoint
                .connect(address, server_name)
                .map_err(QuicError::Connect)?
        };
        let connection = bounded(self.config.operation_timeout, "handshake", connecting).await;
        self.health.check()?;
        let connection = connection?.map_err(QuicError::Connection)?;
        QuicConnection::authenticated(connection, self.config.clone(), Arc::clone(&self.health))
    }

    /// Accept one authenticated connection within a finite admission deadline.
    ///
    /// # Errors
    /// Returns closed endpoint, full admission, timeout or TLS errors.
    pub async fn accept(&self) -> Result<QuicConnection, QuicError> {
        self.health.check()?;
        let operation = async {
            let incoming = self.endpoint.accept().await.ok_or(QuicError::Closed)?;
            let connecting = {
                let _guard = lock_mutex(&self.admission);
                if let Err(error) = self.check_capacity() {
                    incoming.refuse();
                    return Err(error);
                }
                incoming.accept().map_err(QuicError::Connection)?
            };
            connecting.await.map_err(QuicError::Connection)
        };
        let connection = bounded(self.config.operation_timeout, "accept", operation).await;
        self.health.check()?;
        QuicConnection::authenticated(connection??, self.config.clone(), Arc::clone(&self.health))
    }

    fn check_capacity(&self) -> Result<(), QuicError> {
        self.health.check()?;
        if self.endpoint.open_connections() >= self.config.max_connections as usize {
            Err(QuicError::ConnectionCapacity)
        } else {
            Ok(())
        }
    }

    /// Close all connections and stop new admission.
    pub fn close(&self) {
        self.endpoint.close(0_u8.into(), b"endpoint closed");
    }

    /// Wait for closing connections to drain within the operation deadline.
    ///
    /// # Errors
    /// Returns a deadline error if connections remain alive.
    pub async fn wait_idle(&self) -> Result<(), QuicError> {
        bounded(
            self.config.operation_timeout,
            "endpoint drain",
            self.endpoint.wait_idle(),
        )
        .await
    }
}

impl Drop for QuicEndpoint {
    fn drop(&mut self) {
        self.close();
    }
}

/// Authenticated connection with bounded independent bidirectional streams.
#[derive(Clone)]
pub struct QuicConnection {
    pub(super) connection: quinn::Connection,
    config: QuicConfig,
    health: Arc<RuntimeHealth>,
    streams: Arc<StreamAdmission>,
}

impl QuicConnection {
    fn authenticated(
        connection: quinn::Connection,
        mut config: QuicConfig,
        health: Arc<RuntimeHealth>,
    ) -> Result<Self, QuicError> {
        // Condition 1: Quinn's TLS handshake ABI exposes protocol metadata as Any.
        let protocol = connection
            .handshake_data()
            .and_then(|data| data.downcast::<quinn::crypto::rustls::HandshakeData>().ok())
            .and_then(|data| data.protocol);
        if protocol.as_deref() != Some(config.alpn.as_ref()) {
            connection.close(
                super::error::PROTOCOL_REJECTED,
                b"application protocol rejected",
            );
            return Err(QuicError::ApplicationProtocol);
        }
        let Some(protocol) = protocol else {
            return Err(QuicError::ApplicationProtocol);
        };
        config.alpn = Arc::from(protocol);
        let streams = StreamAdmission::new(config.max_streams);
        Ok(Self {
            connection,
            config,
            health,
            streams,
        })
    }

    /// Open an ordered message stream within the configured admission deadline.
    ///
    /// # Errors
    /// Returns connection failure or stream-admission timeout.
    pub async fn open_stream(&self) -> Result<QuicStream, QuicError> {
        self.health.check()?;
        let permit = StreamAdmission::reserve(&self.streams)?;
        let streams = bounded(
            self.config.operation_timeout,
            "open stream",
            self.connection.open_bi(),
        )
        .await;
        self.health.check()?;
        let streams = streams?.map_err(QuicError::Connection)?;
        Ok(QuicStream::new(
            streams,
            &self.config,
            Arc::clone(&self.health),
            permit,
        ))
    }

    /// Accept an ordered message stream within the configured deadline.
    ///
    /// # Errors
    /// Returns connection failure or stream-admission timeout.
    pub async fn accept_stream(&self) -> Result<QuicStream, QuicError> {
        self.health.check()?;
        let permit = StreamAdmission::reserve(&self.streams)?;
        let streams = bounded(
            self.config.operation_timeout,
            "accept stream",
            self.connection.accept_bi(),
        )
        .await;
        self.health.check()?;
        let streams = streams?.map_err(QuicError::Connection)?;
        Ok(QuicStream::new(
            streams,
            &self.config,
            Arc::clone(&self.health),
            permit,
        ))
    }

    /// Peer certificate chain validated during TLS, if the peer supplied one.
    ///
    /// Server certificates exist on client connections. Server connections have
    /// client certificates only when the endpoint requires mutual TLS.
    pub fn peer_certificates(&self) -> Option<Vec<CertificateDer<'static>>> {
        // Condition 1: Quinn exposes negotiated TLS identity through Any.
        self.connection
            .peer_identity()?
            .downcast::<Vec<CertificateDer<'static>>>()
            .ok()
            .map(|certificates| *certificates)
    }

    /// TLS-negotiated application protocol for this authenticated connection.
    pub fn alpn(&self) -> &[u8] {
        &self.config.alpn
    }

    /// Close this connection without affecting other endpoint connections.
    pub fn close(&self) {
        self.connection
            .close(0_u8.into(), b"session connection closed");
    }
}

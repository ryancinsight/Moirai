//! Async TLS client for the Moirai runtime.
//!
//! Drives the sans-I/O [`rustls`] state machine (via the runtime-agnostic
//! [`futures_rustls`] adapter) over a Moirai async socket — **no Tokio**. Moirai's
//! [`moirai_async::io::AsyncRead`]/[`AsyncWrite`](moirai_async::io::AsyncWrite)
//! traits are signature-identical to `futures_io`, so the bridge is two zero-cost
//! newtypes; all TLS cryptography and record framing are delegated to the audited
//! `rustls` stack (ADR-015 — moirai owns transport, not crypto reimplementation).
//!
//! ```no_run
//! # async fn ex<S: moirai_async::io::AsyncRead + moirai_async::io::AsyncWrite + Unpin>(sock: S) {
//! use moirai_tls::{TlsConnector, ServerName};
//! let connector = TlsConnector::with_webpki_roots();
//! let domain = ServerName::try_from("example.com").unwrap();
//! let _tls = connector.connect(domain, sock).await.unwrap();
//! # }
//! ```

#![forbid(unsafe_code)]
#![deny(missing_docs)]

use std::io;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};

use futures_rustls::rustls::{ClientConfig, RootCertStore};
use futures_rustls::TlsConnector as RustlsConnector;
use moirai_async::io::{AsyncRead as MoiraiRead, AsyncWrite as MoiraiWrite};

/// Re-export of the underlying `rustls` types (config, crypto, errors).
pub use futures_rustls::rustls;
/// Server name used for SNI and certificate validation.
pub use rustls_pki_types::ServerName;

/// Zero-cost adapter exposing a Moirai async stream as a [`futures_io`] stream.
///
/// Requires `S: Unpin` (Moirai sockets and the `rustls` stream are `Unpin`), which
/// keeps the projection safe without `unsafe`.
#[derive(Debug)]
pub struct ToFuturesIo<S>(
    /// The wrapped Moirai async stream.
    pub S,
);

impl<S: MoiraiRead + Unpin> futures_io::AsyncRead for ToFuturesIo<S> {
    #[inline]
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        MoiraiRead::poll_read(Pin::new(&mut self.get_mut().0), cx, buf)
    }
}

impl<S: MoiraiWrite + Unpin> futures_io::AsyncWrite for ToFuturesIo<S> {
    #[inline]
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        MoiraiWrite::poll_write(Pin::new(&mut self.get_mut().0), cx, buf)
    }

    #[inline]
    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        MoiraiWrite::poll_flush(Pin::new(&mut self.get_mut().0), cx)
    }

    #[inline]
    fn poll_close(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        MoiraiWrite::poll_shutdown(Pin::new(&mut self.get_mut().0), cx)
    }
}

/// Zero-cost adapter exposing a [`futures_io`] stream as a Moirai async stream.
#[derive(Debug)]
pub struct ToMoiraiIo<S>(
    /// The wrapped `futures_io` stream.
    pub S,
);

impl<S: futures_io::AsyncRead + Unpin> MoiraiRead for ToMoiraiIo<S> {
    #[inline]
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        futures_io::AsyncRead::poll_read(Pin::new(&mut self.get_mut().0), cx, buf)
    }
}

impl<S: futures_io::AsyncWrite + Unpin> MoiraiWrite for ToMoiraiIo<S> {
    #[inline]
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        futures_io::AsyncWrite::poll_write(Pin::new(&mut self.get_mut().0), cx, buf)
    }

    #[inline]
    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        futures_io::AsyncWrite::poll_flush(Pin::new(&mut self.get_mut().0), cx)
    }

    #[inline]
    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        futures_io::AsyncWrite::poll_close(Pin::new(&mut self.get_mut().0), cx)
    }
}

/// A negotiated client-side TLS stream over a Moirai async socket `S`.
///
/// Implements Moirai's [`AsyncRead`](moirai_async::io::AsyncRead) and
/// [`AsyncWrite`](moirai_async::io::AsyncWrite), so upper layers (HTTP) treat it
/// exactly like a plaintext socket.
pub type TlsStream<S> = ToMoiraiIo<futures_rustls::client::TlsStream<ToFuturesIo<S>>>;

/// Client TLS connector wrapping a shared [`rustls::ClientConfig`].
#[derive(Clone)]
pub struct TlsConnector {
    inner: RustlsConnector,
}

impl TlsConnector {
    /// Build a connector from an existing `rustls` client configuration.
    #[must_use]
    pub fn from_config(config: Arc<ClientConfig>) -> Self {
        Self {
            inner: RustlsConnector::from(config),
        }
    }

    /// Build a connector trusting the Mozilla root CA set ([`webpki_roots`]),
    /// using the `ring` crypto provider and safe default protocol versions.
    #[must_use]
    pub fn with_webpki_roots() -> Self {
        let mut roots = RootCertStore::empty();
        roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
        Self::from_config(Arc::new(client_config_with_roots(roots)))
    }

    /// Perform the TLS handshake over `stream`, validating the peer certificate
    /// against `domain`. Returns a [`TlsStream`] exposing Moirai async I/O.
    ///
    /// # Errors
    /// Propagates handshake, certificate-validation, and underlying socket errors.
    pub async fn connect<S>(
        &self,
        domain: ServerName<'static>,
        stream: S,
    ) -> io::Result<TlsStream<S>>
    where
        S: MoiraiRead + MoiraiWrite + Unpin,
    {
        let raw = self.inner.connect(domain, ToFuturesIo(stream)).await?;
        Ok(ToMoiraiIo(raw))
    }
}

/// Construct a `ring`-backed [`ClientConfig`] with safe default protocol versions
/// trusting `roots`. Shared by [`TlsConnector::with_webpki_roots`] and tests that
/// supply a custom root (self-signed) store.
#[must_use]
pub fn client_config_with_roots(roots: RootCertStore) -> ClientConfig {
    ClientConfig::builder_with_provider(Arc::new(
        futures_rustls::rustls::crypto::ring::default_provider(),
    ))
    .with_safe_default_protocol_versions()
    .expect("ring provider supports the safe default protocol versions")
    .with_root_certificates(roots)
    .with_no_client_auth()
}

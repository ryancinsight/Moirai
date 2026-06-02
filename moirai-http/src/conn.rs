//! Transport connection: a plain or TLS-wrapped Moirai TCP stream, unified as a
//! single Moirai async stream via a static enum (no `dyn` on the hot path).

use std::io;
use std::pin::Pin;
use std::task::{Context, Poll};

use moirai_async::io::{AsyncRead, AsyncWrite};
use moirai_async::net::TcpStream;
use moirai_tls::{ServerName, TlsConnector, TlsStream};

/// Connection target: scheme/host/port identifying a pool bucket.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Origin {
    /// True for `https`, false for `http`.
    pub secure: bool,
    /// Host name (used for TCP connect and TLS SNI / cert validation).
    pub host: String,
    /// TCP port.
    pub port: u16,
}

impl Origin {
    /// Address string `host:port` for [`TcpStream::connect`].
    #[must_use]
    pub fn authority(&self) -> String {
        format!("{}:{}", self.host, self.port)
    }

    /// Value for the `Host` header: `host`, or `host:port` when the port is
    /// non-default for the scheme (443 for https, 80 for http).
    #[must_use]
    pub fn host_header(&self) -> String {
        let default = if self.secure { 443 } else { 80 };
        if self.port == default {
            self.host.clone()
        } else {
            format!("{}:{}", self.host, self.port)
        }
    }
}

/// An established HTTP transport connection.
pub enum Conn {
    /// Plaintext HTTP over TCP.
    Plain(TcpStream),
    /// HTTPS: TLS over TCP (boxed — the rustls stream is large).
    Tls(Box<TlsStream<TcpStream>>),
}

impl Conn {
    /// Open a new connection to `origin`, performing the TLS handshake when secure.
    ///
    /// # Errors
    /// Propagates DNS/connect and TLS handshake failures.
    pub async fn connect(origin: &Origin, tls: &TlsConnector) -> io::Result<Self> {
        let tcp = TcpStream::connect(&origin.authority()).await?;
        if origin.secure {
            let domain = ServerName::try_from(origin.host.clone())
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e))?;
            let stream = tls.connect(domain, tcp).await?;
            Ok(Conn::Tls(Box::new(stream)))
        } else {
            Ok(Conn::Plain(tcp))
        }
    }
}

impl AsyncRead for Conn {
    #[inline]
    fn poll_read(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &mut [u8],
    ) -> Poll<io::Result<usize>> {
        match self.get_mut() {
            Conn::Plain(s) => AsyncRead::poll_read(Pin::new(s), cx, buf),
            Conn::Tls(s) => AsyncRead::poll_read(Pin::new(s.as_mut()), cx, buf),
        }
    }
}

impl AsyncWrite for Conn {
    #[inline]
    fn poll_write(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
        buf: &[u8],
    ) -> Poll<io::Result<usize>> {
        match self.get_mut() {
            Conn::Plain(s) => AsyncWrite::poll_write(Pin::new(s), cx, buf),
            Conn::Tls(s) => AsyncWrite::poll_write(Pin::new(s.as_mut()), cx, buf),
        }
    }

    #[inline]
    fn poll_flush(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        match self.get_mut() {
            Conn::Plain(s) => AsyncWrite::poll_flush(Pin::new(s), cx),
            Conn::Tls(s) => AsyncWrite::poll_flush(Pin::new(s.as_mut()), cx),
        }
    }

    #[inline]
    fn poll_shutdown(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        match self.get_mut() {
            Conn::Plain(s) => AsyncWrite::poll_shutdown(Pin::new(s), cx),
            Conn::Tls(s) => AsyncWrite::poll_shutdown(Pin::new(s.as_mut()), cx),
        }
    }
}

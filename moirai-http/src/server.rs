//! Bounded one-shot HTTP/1.1 server transport over Moirai TCP sockets.

use std::io;
use std::net::SocketAddr;

use moirai_async::net::{TcpListener, TcpServerConfig};

mod config;
mod connection;
mod response;

pub use config::{
    DEFAULT_MAX_BODY_BYTES, DEFAULT_MAX_CONNECTIONS, DEFAULT_MAX_HEADER_BYTES,
    DEFAULT_MAX_HEADER_COUNT, DEFAULT_MAX_RESPONSE_BYTES, DEFAULT_REQUEST_TIMEOUT, ServerConfig,
};
pub use connection::{AwaitingRequest, AwaitingResponse, HttpConnection, HttpRequest};
pub use response::HttpResponse;

/// A bound HTTP/1.1 server listener.
pub struct HttpServer {
    listener: TcpListener,
    config: ServerConfig,
}

impl HttpServer {
    /// Bind a one-shot HTTP server to the first address resolved from `addr`.
    ///
    /// # Errors
    /// Returns invalid configuration, address resolution, or socket-bind
    /// failures.
    pub async fn bind(addr: &str, config: ServerConfig) -> io::Result<Self> {
        config.validate()?;
        let tcp_config = TcpServerConfig {
            max_connections: Some(config.max_connections),
            nodelay: true,
            keep_alive: None,
            timeout: Some(config.request_timeout),
        };
        let listener = TcpListener::bind_with_config(addr, tcp_config).await?;
        Ok(Self { listener, config })
    }

    /// Return the local address assigned to the listener.
    ///
    /// # Errors
    /// Propagates the underlying socket query failure.
    pub fn local_addr(&self) -> io::Result<SocketAddr> {
        self.listener.local_addr()
    }

    /// Accept one connection and enter the request-reading typestate.
    ///
    /// # Errors
    /// Returns connection-pool exhaustion or an accept failure.
    pub async fn accept(&self) -> io::Result<HttpConnection<AwaitingRequest>> {
        let (stream, _) = self.listener.accept().await?;
        Ok(HttpConnection {
            stream,
            config: self.config,
            prefix: Vec::new(),
            state: AwaitingRequest,
        })
    }
}

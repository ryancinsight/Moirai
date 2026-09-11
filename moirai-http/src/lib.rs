//! Minimal async HTTP/1.1 client and one-shot server transport for the Moirai runtime.
//!
//! Runs over Moirai async sockets and [`moirai_tls`] without Tokio. The client
//! provides Content-Length and chunked response framing, a bounded keep-alive
//! pool with idle eviction, RFC 9110 redirects, and one deadline across each
//! logical request. Its optional server boundary uses bounded request bodies,
//! typed one-shot connection typestates, and the same Moirai socket/deadline
//! primitives. HTTP/2 and vendor protocols remain outside this crate.

// Response bytes and redirect locations cross a trust boundary. Panicking
// arithmetic and indexing remain denied for every implementation leaf.
#![deny(clippy::indexing_slicing, clippy::arithmetic_side_effects)]
#![forbid(unsafe_code)]
#![deny(missing_docs)]

mod client;
pub mod codec;
pub mod conn;
mod pool;
mod redirect;
pub mod request;
pub mod server;
pub mod upgrade;
pub mod websocket;

pub use client::HttpClient;
pub use codec::Response;
pub use conn::Origin;
pub use request::HttpRequestHead;
pub use server::{
    AwaitingRequest, AwaitingResponse, HttpConnection, HttpRequest, HttpResponse, HttpServer,
    ServerConfig,
};
pub use upgrade::{
    WebSocketConfig, WebSocketUpgrade, accept_websocket, accept_websocket_with_validator,
};
pub use websocket::WebSocketStream;

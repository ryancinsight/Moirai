# ADR 0054: Bounded HTTP server transport

- Status: Accepted
- Date: 2026-09-11
- Item: [MOI-HTTP-SERVER-2026-09-11](../backlog.md#MOI-HTTP-SERVER-2026-09-11)

## Context

Metis needs an optional HTTP deployment boundary for typed browser fragments
and local service demonstrations. `moirai-http` already owns bounded HTTP/1.1
request parsing for WebSocket upgrades and response parsing for its client, but
it does not expose a server transport. Adding Axum or Tokio downstream would
duplicate the socket and deadline policy and would make the first-party Moirai
stack less useful to Atlas consumers.

The transport must provide a small protocol surface. It must not decide which
routes, origins, sessions, capabilities, markup, or domain records are valid;
those decisions belong to Metis and its consumers. It must also keep a hostile
peer from driving unbounded header, body, response, connection, or wait state.

## Decision

Add a `moirai-http` server boundary over `moirai_async::net::TcpListener` and
`TcpStream`:

- `ServerConfig` validates non-zero header, header-count, body, response, and
  request-deadline limits.
- `HttpServer::bind` and `HttpServer::accept` own the listener. Accepted
  connections use `HttpConnection<AwaitingRequest>` and
  `HttpConnection<AwaitingResponse>` typestates, so a response cannot be
  written before a request is read and a connection is terminal after one
  response.
- The request parser accepts one origin-form HTTP/1.1 request with a bounded
  header block and optional bounded `Content-Length` body. Transfer-encoded
  request bodies and bytes beyond the declared body are rejected; the server
  closes after the response, so keep-alive and pipelining are not implied.
- Responses validate a three-digit status, token-safe header names, and
  control-byte-free values. The transport writes its own `Content-Length` and
  `Connection: close` fields, bounds the encoded response, applies HEAD body
  suppression, flushes, and shuts down the write side. Partial I/O or deadline
  failure terminalizes the connection and is never retried.

The server layer uses Moirai's async traits and timer. It does not add a
third-party HTTP runtime, expose `web-sys`, generate HTML, or inspect DICOM.

## Alternatives rejected

1. Adding Axum or Hyper to Metis would introduce a second async/socket stack
   and violate first-party provider ownership for this capability.
2. Exposing a generic callback server in Moirai would move route and authority
   policy into the transport and require an erased hot-path handler.
3. Reusing the client response parser for server requests would accept an
   unbounded or ambiguous request body and would not express the one-shot
   connection lifecycle.
4. Keeping the existing private request-head parser forces every consumer to
   duplicate framing, validation, and deadline behavior.

## Threat model and limits

The peer controls request bytes and timing. Header, header-count, body, and
response limits bound allocation; token and control-byte validation blocks
header injection; origin-form validation prevents an absolute-form target from
silently changing authority; transfer encoding is rejected rather than
interpreted ambiguously. A finite deadline bounds slow reads and writes. The
one-response lifecycle avoids request-smuggling and retry-after-partial-write
states. The listener's existing connection pool bounds concurrent sockets.

The boundary does not authenticate, authorize, select routes, sanitize HTML,
or apply application policy. Metis must perform those checks before emitting a
fragment, and RITK remains the DICOM parser and viewer owner.

## Verification

`moirai-http` integration tests drive a real loopback listener and client for a
valid request/response, malformed and transfer-encoded requests, body and
response limits, HEAD suppression, and terminal connection closure. Unit tests
cover configuration and response validation. The crate's native nextest,
warning-denied Clippy, documentation, and metadata gates are required before
the item closes.

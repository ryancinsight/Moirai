//! ADR-015 P0 foundation spike: prove async TCP works end-to-end driven by the
//! unified Moirai runtime (`block_on` + `spawn_async`), i.e. the reactor path the
//! HTTP/TLS stack will build on — not the `futures::executor` self-wake fallback.
//!
//! Gate: must pass on Linux (epoll/kqueue) and Windows (IOCP, the flagged risk).

#![cfg(feature = "async")]

use moirai_async::io::{AsyncReadExt, AsyncWriteExt};
use moirai_async::net::{TcpListener, TcpStream};

#[test]
fn tcp_echo_round_trip_on_unified_runtime() {
    let rt = moirai::global();

    // Bind on an ephemeral loopback port, driven by the Moirai runtime.
    let listener = rt
        .block_on(TcpListener::bind("127.0.0.1:0"))
        .expect("bind must succeed");
    let addr = listener.local_addr().expect("local_addr must resolve");

    // Server: accept one connection and echo a 4-byte frame, on the worker pool.
    let server = rt.spawn_async(async move {
        let (mut stream, _peer) = listener.accept().await.expect("accept must succeed");
        let mut buf = [0u8; 4];
        stream.read_exact(&mut buf).await.expect("server read");
        stream.write_all(&buf).await.expect("server echo");
        stream.flush().await.expect("server flush");
        buf
    });

    // Client: connect, send, read the echo back — value-semantic assertion.
    let echo = rt.block_on(async move {
        let mut client = TcpStream::connect(&addr.to_string())
            .await
            .expect("client connect");
        client.write_all(b"ping").await.expect("client write");
        let mut echo = [0u8; 4];
        client.read_exact(&mut echo).await.expect("client read echo");
        echo
    });

    assert_eq!(&echo, b"ping", "client must receive the exact echoed frame");
    let served = server.join().expect("server task must complete");
    assert_eq!(served, Ok(*b"ping"), "server must have observed the frame");
}

use super::*;
use std::io::{Read as _, Write as _};
use std::net::TcpListener;

fn read_request_head(stream: &mut std::net::TcpStream) {
    let mut buffer = Vec::new();
    let mut byte = [0u8; 1];
    while !buffer.ends_with(b"\r\n\r\n") {
        match stream.read(&mut byte) {
            Ok(0) | Err(_) => break,
            Ok(_) => buffer.push(byte[0]),
        }
    }
}

fn seed_stale_pooled_connection(client: &HttpClient, origin: &Origin, listener: &TcpListener) {
    let connection = moirai::block_on(Conn::connect(origin, &client.tls))
        .expect("pooled connection must establish");
    let (stale, _) = listener
        .accept()
        .expect("server must accept pooled connection");
    drop(stale);
    client.pool.put(origin, connection, 1);
}

#[test]
fn idempotent_get_retries_stale_pooled_connection_and_succeeds() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("address").port();
    let url = format!("http://127.0.0.1:{port}/x");
    let origin = Origin {
        secure: false,
        host: "127.0.0.1".to_owned(),
        port,
    };
    let client = HttpClient::new();
    seed_stale_pooled_connection(&client, &origin, &listener);

    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("retry connection must arrive");
        read_request_head(&mut stream);
        stream
            .write_all(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok")
            .expect("response write");
    });

    let response = moirai::block_on(client.get(&url, &[]))
        .expect("GET must retry the stale pooled connection");
    assert_eq!(response.status, 200);
    assert_eq!(response.body, b"ok");
    server.join().expect("server thread");
}

#[test]
fn non_idempotent_post_does_not_open_a_retry_connection() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("address").port();
    let url = format!("http://127.0.0.1:{port}/submit");
    let origin = Origin {
        secure: false,
        host: "127.0.0.1".to_owned(),
        port,
    };
    let client = HttpClient::new();
    seed_stale_pooled_connection(&client, &origin, &listener);
    listener
        .set_nonblocking(true)
        .expect("nonblocking listener");

    let error = moirai::block_on(client.request("POST", &url, &[], Some(b"payload")))
        .expect_err("POST over a stale connection must not be retried");
    assert!(
        matches!(
            error.kind(),
            io::ErrorKind::UnexpectedEof
                | io::ErrorKind::ConnectionReset
                | io::ErrorKind::ConnectionAborted
                | io::ErrorKind::BrokenPipe
        ),
        "unexpected error kind {:?}: {error}",
        error.kind()
    );
    let retry = listener
        .accept()
        .expect_err("a retry would already be queued before request returned");
    assert_eq!(retry.kind(), io::ErrorKind::WouldBlock);
}

#[test]
fn zero_timeout_bounds_the_entire_request_without_waiting() {
    let listener = TcpListener::bind("127.0.0.1:0").expect("bind");
    let port = listener.local_addr().expect("address").port();
    let mut client = HttpClient::new();
    client.set_timeout(Duration::ZERO);

    let error = moirai::block_on(client.get(&format!("http://127.0.0.1:{port}/pending"), &[]))
        .expect_err("a pending exchange must observe the zero deadline");
    assert_eq!(error.kind(), io::ErrorKind::TimedOut);
    assert_eq!(error.to_string(), "logical HTTP request timed out");
}

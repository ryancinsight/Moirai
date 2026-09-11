//! Real loopback coverage for the bounded HTTP/1.1 server transport.

use moirai_async::io::AsyncWriteExt;
use moirai_async::net::TcpStream;
use moirai_http::{HttpResponse, HttpServer, ServerConfig};
use std::io;

fn test_config() -> ServerConfig {
    ServerConfig::new(8, 4096, 16, 64, 256, std::time::Duration::from_secs(2))
        .expect("test limits are valid")
}

async fn read_to_close(stream: &mut TcpStream) -> io::Result<Vec<u8>> {
    let mut output = Vec::with_capacity(256);
    let mut chunk = [0_u8; 128];
    loop {
        let count = stream.read(&mut chunk).await?;
        if count == 0 {
            return Ok(output);
        }
        output.extend_from_slice(
            chunk
                .get(..count)
                .ok_or_else(|| io::Error::other("test read exceeded its buffer"))?,
        );
        if output.len() > 256 {
            return Err(io::Error::other("test response exceeded its bound"));
        }
    }
}

#[test]
fn server_round_trip_is_bounded_and_connection_scoped() {
    let runtime = moirai::global();
    let server = runtime
        .block_on(HttpServer::bind("127.0.0.1:0", test_config()))
        .expect("server bind");
    let address = server.local_addr().expect("server address");
    let task = runtime.spawn_async(async move {
        let connection = server.accept().await?;
        let (request, connection) = connection.read_request().await?;
        assert_eq!(request.method(), "POST");
        assert_eq!(request.target(), "/fragment");
        assert_eq!(request.header("origin"), Some("http://127.0.0.1:8080"));
        assert_eq!(request.body(), b"payload");
        let mut response = HttpResponse::new(200, b"ok".to_vec())?;
        response.set_header("Content-Type", "text/plain")?;
        connection.write_response(response).await
    });

    let response = runtime.block_on(async move {
        let mut client = TcpStream::connect(&address.to_string()).await?;
        client
            .write_all(
                b"POST /fragment HTTP/1.1\r\nHost: 127.0.0.1\r\nOrigin: http://127.0.0.1:8080\r\nContent-Length: 7\r\n\r\npayload",
            )
            .await?;
        client.flush().await?;
        read_to_close(&mut client).await
    });
    let response = response.expect("client response");
    assert_eq!(
        response,
        b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok"
    );
    task.join()
        .expect("server task join")
        .expect("server task result")
        .expect("server response");
}

#[test]
fn head_response_preserves_length_and_suppresses_body() {
    let runtime = moirai::global();
    let server = runtime
        .block_on(HttpServer::bind("127.0.0.1:0", test_config()))
        .expect("server bind");
    let address = server.local_addr().expect("server address");
    let task = runtime.spawn_async(async move {
        let connection = server.accept().await?;
        let (request, connection) = connection.read_request().await?;
        assert_eq!(request.method(), "HEAD");
        let response = HttpResponse::new(200, b"hidden".to_vec())?;
        connection.write_response(response).await
    });

    let response = runtime.block_on(async move {
        let mut client = TcpStream::connect(&address.to_string()).await?;
        client
            .write_all(b"HEAD /status HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
            .await?;
        client.flush().await?;
        read_to_close(&mut client).await
    });
    let response = response.expect("client response");
    assert_eq!(
        response,
        b"HTTP/1.1 200 OK\r\nContent-Length: 6\r\nConnection: close\r\n\r\n"
    );
    task.join()
        .expect("server task join")
        .expect("server task result")
        .expect("server response");
}

#[test]
fn server_rejects_transfer_encoding_and_oversized_body() {
    let runtime = moirai::global();
    let server = runtime
        .block_on(HttpServer::bind("127.0.0.1:0", test_config()))
        .expect("server bind");
    let address = server.local_addr().expect("server address");
    let task = runtime.spawn_async(async move {
        let connection = server.accept().await?;
        connection.read_request().await.map(|_| ())
    });
    runtime
        .block_on(async move {
            let mut client = TcpStream::connect(&address.to_string()).await?;
            client
            .write_all(
                b"POST /fragment HTTP/1.1\r\nHost: 127.0.0.1\r\nTransfer-Encoding: chunked\r\n\r\n",
            )
            .await
        })
        .expect("malformed request write");
    let error = task
        .join()
        .expect("transfer-encoding task join")
        .expect("transfer-encoding server task result")
        .expect_err("transfer encoding must be rejected");
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);

    let server = runtime
        .block_on(HttpServer::bind(
            "127.0.0.1:0",
            ServerConfig::new(8, 4096, 16, 3, 256, std::time::Duration::from_secs(2))
                .expect("body limit"),
        ))
        .expect("second server bind");
    let address = server.local_addr().expect("second server address");
    let task = runtime.spawn_async(async move {
        let connection = server.accept().await?;
        connection.read_request().await.map(|_| ())
    });
    runtime
        .block_on(async move {
            let mut client = TcpStream::connect(&address.to_string()).await?;
            client
                .write_all(
                    b"POST /fragment HTTP/1.1\r\nHost: 127.0.0.1\r\nContent-Length: 4\r\n\r\ntest",
                )
                .await
        })
        .expect("oversized request write");
    let error = task
        .join()
        .expect("body-limit task join")
        .expect("body-limit server task result")
        .expect_err("oversized body must be rejected");
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
}

#[test]
fn server_rejects_absolute_form_targets() {
    let runtime = moirai::global();
    let server = runtime
        .block_on(HttpServer::bind("127.0.0.1:0", test_config()))
        .expect("server bind");
    let address = server.local_addr().expect("server address");
    let task = runtime.spawn_async(async move {
        let connection = server.accept().await?;
        connection.read_request().await.map(|_| ())
    });
    runtime
        .block_on(async move {
            let mut client = TcpStream::connect(&address.to_string()).await?;
            client
                .write_all(
                    b"GET http://example.test/fragment HTTP/1.1\r\nHost: example.test\r\n\r\n",
                )
                .await
        })
        .expect("request write");
    let error = task
        .join()
        .expect("target-validation task join")
        .expect("target-validation server task result")
        .expect_err("absolute-form target must be rejected");
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
}

#[test]
fn server_request_deadline_terminalizes_an_idle_peer() {
    let runtime = moirai::global();
    let config = ServerConfig::new(8, 4096, 16, 64, 256, std::time::Duration::from_millis(50))
        .expect("deadline");
    let server = runtime
        .block_on(HttpServer::bind("127.0.0.1:0", config))
        .expect("server bind");
    let address = server.local_addr().expect("server address");
    let task = runtime.spawn_async(async move {
        let connection = server.accept().await?;
        connection.read_request().await.map(|_| ())
    });
    let client = runtime
        .block_on(TcpStream::connect(&address.to_string()))
        .expect("idle peer connect");
    let error = task
        .join()
        .expect("deadline task join")
        .expect("deadline server task result")
        .expect_err("idle peer must hit the request deadline");
    drop(client);
    assert_eq!(error.kind(), io::ErrorKind::TimedOut);
}

#[test]
fn server_rejects_an_oversized_response_before_writing() {
    let runtime = moirai::global();
    let config = ServerConfig::new(8, 4096, 16, 64, 32, std::time::Duration::from_secs(2))
        .expect("response limit");
    let server = runtime
        .block_on(HttpServer::bind("127.0.0.1:0", config))
        .expect("server bind");
    let address = server.local_addr().expect("server address");
    let task = runtime.spawn_async(async move {
        let connection = server.accept().await?;
        let (_, connection) = connection.read_request().await?;
        let response = HttpResponse::new(200, vec![b'x'; 64])?;
        connection.write_response(response).await
    });
    runtime
        .block_on(async move {
            let mut client = TcpStream::connect(&address.to_string()).await?;
            client
                .write_all(b"GET /fragment HTTP/1.1\r\nHost: 127.0.0.1\r\n\r\n")
                .await
        })
        .expect("request write");
    let error = task
        .join()
        .expect("response-limit task join")
        .expect("response-limit server task result")
        .expect_err("oversized response must be rejected");
    assert_eq!(error.kind(), io::ErrorKind::InvalidData);
}

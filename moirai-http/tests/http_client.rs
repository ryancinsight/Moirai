//! ADR-015 P2 verification against local HTTP/1.1 servers: framing, keep-alive,
//! bounded redirects, method/body policy, and destination-aware fields.

use moirai_async::io::AsyncWriteExt;
use moirai_async::net::{TcpListener, TcpStream};
use moirai_http::HttpClient;
use std::io::{Read as _, Write as _};
use std::net::{TcpListener as StdTcpListener, TcpStream as StdTcpStream};

/// Read an HTTP request head; return `(method, path)` or `None` at EOF.
async fn read_request_head(stream: &mut TcpStream) -> std::io::Result<Option<(String, String)>> {
    let mut buf = Vec::new();
    let mut tmp = [0u8; 1024];
    loop {
        if let Some(p) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            let line_end = buf.windows(2).position(|w| w == b"\r\n").unwrap_or(p);
            let line = String::from_utf8_lossy(&buf[..line_end]).into_owned();
            let mut it = line.split_whitespace();
            let method = it.next().unwrap_or("").to_string();
            let path = it.next().unwrap_or("").to_string();
            return Ok(Some((method, path)));
        }
        let n = stream.read(&mut tmp).await?;
        if n == 0 {
            return Ok(None);
        }
        buf.extend_from_slice(&tmp[..n]);
    }
}

/// Canned response for a path. `HEAD` keeps headers but omits the body.
fn response_for(method: &str, path: &str) -> Vec<u8> {
    let is_head = method == "HEAD";
    match path {
        "/fixed" => {
            let mut r = b"HTTP/1.1 200 OK\r\nContent-Length: 11\r\n\r\n".to_vec();
            if !is_head {
                r.extend_from_slice(b"hello world");
            }
            r
        }
        "/chunked" => {
            // "hello " (6 = 0x6) + "world" (5 = 0x5) + terminator.
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n6\r\nhello \r\n5\r\nworld\r\n0\r\n\r\n"
                .to_vec()
        }
        "/range" => {
            let mut r = b"HTTP/1.1 206 Partial Content\r\nContent-Length: 5\r\n\r\n".to_vec();
            if !is_head {
                r.extend_from_slice(b"hello");
            }
            r
        }
        _ => b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n".to_vec(),
    }
}

#[test]
fn http_framing_header_passthrough_and_keepalive() {
    let rt = moirai::global();
    let listener = rt.block_on(TcpListener::bind("127.0.0.1:0")).expect("bind");
    let addr = listener.local_addr().expect("addr");

    // Server: accept connections; each handles requests until the client closes it
    // (keep-alive). Runs until the client drops its pooled connections.
    rt.spawn_async(async move {
        loop {
            let (mut stream, _peer) = match listener.accept().await {
                Ok(c) => c,
                Err(_) => break,
            };
            moirai::global().spawn_async(async move {
                while let Ok(Some((method, path))) = read_request_head(&mut stream).await {
                    let resp = response_for(&method, &path);
                    if stream.write_all(&resp).await.is_err() {
                        break;
                    }
                    let _ = stream.flush().await;
                }
            });
        }
    });

    let base = format!("http://{addr}");
    rt.block_on(async move {
        let client = HttpClient::new();

        let r = client
            .get(&format!("{base}/fixed"), &[])
            .await
            .expect("fixed");
        assert_eq!(r.status, 200);
        assert_eq!(r.body, b"hello world", "Content-Length body");
        assert!(r.keep_alive, "framed HTTP/1.1 response keeps alive");

        let r = client
            .get(&format!("{base}/chunked"), &[])
            .await
            .expect("chunked");
        assert_eq!(r.status, 200);
        assert_eq!(r.body, b"hello world", "chunked decode must reassemble");

        let r = client
            .get(&format!("{base}/range"), &[("Range", "bytes=0-4")])
            .await
            .expect("range");
        assert_eq!(r.status, 206, "Range header passed through to server");
        assert_eq!(r.body, b"hello");

        let r = client
            .head(&format!("{base}/fixed"), &[])
            .await
            .expect("head");
        assert_eq!(r.status, 200);
        assert!(r.body.is_empty(), "HEAD must not read a body");
        assert_eq!(r.header("content-length"), Some("11"));
    });
}

#[derive(Debug)]
struct ObservedRequest {
    method: String,
    path: String,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl ObservedRequest {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(candidate, _)| candidate.eq_ignore_ascii_case(name))
            .map(|(_, value)| value.as_str())
    }
}

fn read_observed_request(stream: &mut StdTcpStream) -> ObservedRequest {
    let mut head = Vec::new();
    let mut byte = [0u8; 1];
    while !head.ends_with(b"\r\n\r\n") {
        stream.read_exact(&mut byte).expect("request head byte");
        head.push(byte[0]);
    }
    let head = String::from_utf8(head).expect("ASCII request head");
    let mut lines = head.lines();
    let mut request_line = lines.next().expect("request line").split_whitespace();
    let method = request_line.next().expect("request method").to_owned();
    let path = request_line.next().expect("request path").to_owned();
    let headers: Vec<_> = lines
        .filter(|line| !line.is_empty())
        .map(|line| {
            let (name, value) = line.split_once(':').expect("header delimiter");
            (name.to_owned(), value.trim().to_owned())
        })
        .collect();
    let content_length = headers
        .iter()
        .find(|(name, _)| name.eq_ignore_ascii_case("content-length"))
        .map_or(0, |(_, value)| {
            value.parse::<usize>().expect("content length")
        });
    let mut body = vec![0; content_length];
    stream.read_exact(&mut body).expect("request body");
    ObservedRequest {
        method,
        path,
        headers,
        body,
    }
}

fn write_response(stream: &mut StdTcpStream, response: &[u8]) {
    stream.write_all(response).expect("response write");
    stream.flush().expect("response flush");
}

#[test]
fn cross_origin_post_redirect_becomes_bodyless_get_without_credentials() {
    let source = StdTcpListener::bind("127.0.0.1:0").expect("source bind");
    let destination = StdTcpListener::bind("127.0.0.1:0").expect("destination bind");
    let source_address = source.local_addr().expect("source address");
    let destination_address = destination.local_addr().expect("destination address");

    let source_server = std::thread::spawn(move || {
        let (mut stream, _) = source.accept().expect("source request");
        let request = read_observed_request(&mut stream);
        assert_eq!(request.method, "POST");
        assert_eq!(request.path, "/submit");
        assert_eq!(request.body, b"payload");
        assert_eq!(request.header("authorization"), Some("Bearer secret"));
        write_response(
            &mut stream,
            format!(
                "HTTP/1.1 302 Found\r\nLocation: http://{destination_address}/final\r\nContent-Length: 0\r\nConnection: close\r\n\r\n"
            )
            .as_bytes(),
        );
    });
    let destination_server = std::thread::spawn(move || {
        let (mut stream, _) = destination.accept().expect("destination request");
        let request = read_observed_request(&mut stream);
        assert_eq!(request.method, "GET");
        assert_eq!(request.path, "/final");
        assert!(request.body.is_empty());
        assert_eq!(
            request.header("host"),
            Some(destination_address.to_string().as_str())
        );
        assert_eq!(request.header("x-test"), Some("kept"));
        for removed in [
            "authorization",
            "cookie",
            "content-length",
            "content-type",
            "connection",
            "x-private",
        ] {
            assert_eq!(request.header(removed), None, "forwarded {removed}");
        }
        write_response(
            &mut stream,
            b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\nConnection: close\r\n\r\nok",
        );
    });

    let client = HttpClient::new();
    let response = moirai::block_on(client.request(
        "POST",
        &format!("http://{source_address}/submit"),
        &[
            ("Authorization", "Bearer secret"),
            ("Cookie", "session=secret"),
            ("Content-Type", "text/plain"),
            ("Connection", "X-Private"),
            ("X-Private", "secret"),
            ("X-Test", "kept"),
        ],
        Some(b"payload"),
    ))
    .expect("redirected request");
    assert_eq!(response.status, 200);
    assert_eq!(response.body, b"ok");
    source_server.join().expect("source server");
    destination_server.join().expect("destination server");
}

#[test]
fn temporary_redirect_preserves_method_body_and_same_origin_credentials() {
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind");
    let address = listener.local_addr().expect("address");
    let server = std::thread::spawn(move || {
        let (mut first, _) = listener.accept().expect("first request");
        let first_request = read_observed_request(&mut first);
        assert_eq!(first_request.method, "PUT");
        assert_eq!(first_request.body, b"preserved");
        write_response(
            &mut first,
            b"HTTP/1.1 307 Temporary Redirect\r\nLocation: /final\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        );
        drop(first);

        let (mut second, _) = listener.accept().expect("redirect request");
        let second_request = read_observed_request(&mut second);
        assert_eq!(second_request.method, "PUT");
        assert_eq!(second_request.path, "/final");
        assert_eq!(second_request.body, b"preserved");
        assert_eq!(second_request.header("authorization"), Some("Bearer same"));
        assert_eq!(second_request.header("content-type"), Some("text/plain"));
        write_response(
            &mut second,
            b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        );
    });

    let response = moirai::block_on(HttpClient::new().request(
        "PUT",
        &format!("http://{address}/start"),
        &[
            ("Authorization", "Bearer same"),
            ("Content-Type", "text/plain"),
        ],
        Some(b"preserved"),
    ))
    .expect("307 request");
    assert_eq!(response.status, 204);
    server.join().expect("server");
}

#[test]
fn zero_redirect_budget_rejects_the_first_location() {
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind");
    let address = listener.local_addr().expect("address");
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("request");
        let request = read_observed_request(&mut stream);
        assert_eq!(request.path, "/loop");
        write_response(
            &mut stream,
            b"HTTP/1.1 302 Found\r\nLocation: /loop\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        );
    });

    let mut client = HttpClient::new();
    client.set_max_redirects(0);
    let error = moirai::block_on(client.get(&format!("http://{address}/loop"), &[]))
        .expect_err("redirect must exceed the zero budget");
    assert_eq!(error.kind(), std::io::ErrorKind::InvalidData);
    assert_eq!(error.to_string(), "redirect limit of 0 exceeded");
    server.join().expect("server");
}

#[test]
fn redirect_without_location_is_returned_unchanged() {
    let listener = StdTcpListener::bind("127.0.0.1:0").expect("bind");
    let address = listener.local_addr().expect("address");
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().expect("request");
        let request = read_observed_request(&mut stream);
        assert_eq!(request.path, "/choice");
        write_response(
            &mut stream,
            b"HTTP/1.1 302 Found\r\nContent-Length: 4\r\nConnection: close\r\n\r\nstay",
        );
    });

    let response =
        moirai::block_on(HttpClient::new().get(&format!("http://{address}/choice"), &[]))
            .expect("redirect response without Location");
    assert_eq!(response.status, 302);
    assert_eq!(response.body, b"stay");
    server.join().expect("server");
}

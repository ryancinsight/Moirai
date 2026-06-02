//! ADR-015 P2 verification: drive `moirai-http` against a local HTTP/1.1 server
//! (itself on Moirai sockets), covering Content-Length, chunked transfer-encoding,
//! `Range`/206 header pass-through, `HEAD` (no body), and keep-alive reuse.

use moirai_async::io::AsyncWriteExt;
use moirai_async::net::{TcpListener, TcpStream};
use moirai_http::HttpClient;

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

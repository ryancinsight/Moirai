//! A separately compiled, unmodified upstream Quinn peer is a wire reference.
//! Its Tokio dev runtime is excluded from native consumer normal/build graphs.

use super::*;

#[test]
fn native_provider_interoperates_with_unmodified_upstream_quinn() {
    let _trace = trace();
    let _arrange = tracing::info_span!("arrange").entered();
    let executor = executor();
    let (identity, roots) = material();
    let server = QuicEndpoint::server(
        Arc::clone(&executor),
        "127.0.0.1:0".parse().expect("loopback"),
        identity,
        None,
        config(37, 1, DEADLINE),
    )
    .expect("Moirai server");
    let address = server.local_addr().expect("native address");
    let expected = payload(37);
    let reference_runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("upstream dev runtime");
    drop(_arrange);
    let _act = tracing::info_span!("act").entered();
    let (actual, received) = reference_runtime.block_on(async {
        let mut tls = rustls::ClientConfig::builder_with_provider(Arc::new(
            rustls::crypto::ring::default_provider(),
        ))
        .with_protocol_versions(&[&rustls::version::TLS13])
        .expect("TLS 1.3")
        .with_root_certificates(roots)
        .with_no_client_auth();
        tls.alpn_protocols = vec![ALPN.to_vec()];
        let crypto =
            quinn::crypto::rustls::QuicClientConfig::try_from(tls).expect("upstream QUIC TLS");
        let mut reference = quinn::Endpoint::client("127.0.0.1:0".parse().expect("loopback"))
            .expect("unmodified upstream endpoint");
        reference.set_default_client_config(quinn::ClientConfig::new(Arc::new(crypto)));
        let peer = async {
            let connection = reference
                .connect(address, "localhost")
                .expect("upstream connect")
                .await
                .expect("verified handshake");
            let (mut send, mut receive) = connection.open_bi().await.expect("reference stream");
            send.write_all(&37_u32.to_be_bytes())
                .await
                .expect("wire length");
            send.write_all(&expected).await.expect("wire payload");
            send.finish().expect("reference FIN");
            let mut prefix = [0; 4];
            receive.read_exact(&mut prefix).await.expect("reply prefix");
            assert_eq!(u32::from_be_bytes(prefix), 37);
            let mut reply = vec![0; 37];
            receive.read_exact(&mut reply).await.expect("reply payload");
            assert_eq!(receive.read(&mut [0; 1]).await.expect("native FIN"), None);
            assert_eq!(
                send.stopped().await.expect("reference FIN acknowledged"),
                None
            );
            reply
        };
        let native = async {
            let connection = server.accept().await.expect("Moirai handshake");
            assert_eq!(connection.alpn(), ALPN);
            let mut stream = connection.accept_stream().await.expect("native stream");
            let message = stream
                .recv_message()
                .await
                .expect("native read")
                .expect("frame");
            assert_eq!(stream.recv_message().await.expect("reference EOF"), None);
            stream.send_message(&message).await.expect("native reply");
            stream
                .finish_and_wait()
                .await
                .expect("native FIN acknowledged");
            message
        };
        let values = futures::future::join(peer, native).await;
        reference.close(0_u32.into(), b"reference complete");
        server.close();
        let (native_idle, ()) =
            futures::future::join(server.wait_idle(), reference.wait_idle()).await;
        native_idle.expect("native endpoint drained");
        values
    });
    drop(_act);
    let _assert = tracing::info_span!("assert").entered();
    assert_eq!(actual, expected);
    assert_eq!(received, expected);
}

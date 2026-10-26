use super::*;

#[tokio::test]
async fn rebind_recv() {
    let _guard = subscribe();

    let cert = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
    let key = PrivatePkcs8KeyDer::from(cert.signing_key.serialize_der());
    let cert = CertificateDer::from(cert.cert);

    let mut roots = rustls::RootCertStore::empty();
    roots.add(cert.clone()).unwrap();

    let mut client = Endpoint::client(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)).unwrap();
    let mut client_config = ClientConfig::with_root_certificates(Arc::new(roots)).unwrap();
    client_config.transport_config(Arc::new({
        let mut cfg = TransportConfig::default();
        cfg.max_concurrent_uni_streams(1u32.into());
        cfg
    }));
    client.set_default_client_config(client_config);

    let server_config =
        crate::ServerConfig::with_single_cert(vec![cert.clone()], key.into()).unwrap();
    let server = {
        let _guard = tracing::error_span!("server").entered();
        Endpoint::server(
            server_config,
            SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0),
        )
        .unwrap()
    };
    let server_addr = server.local_addr().unwrap();

    const MSG: &[u8; 5] = b"hello";

    let write_send = Arc::new(tokio::sync::Notify::new());
    let write_recv = write_send.clone();
    let connected_send = Arc::new(tokio::sync::Notify::new());
    let connected_recv = connected_send.clone();
    let server = tokio::spawn(async move {
        let connection = server.accept().await.unwrap().await.unwrap();
        info!("got conn");
        connected_send.notify_one();
        write_recv.notified().await;
        let mut stream = connection.open_uni().await.unwrap();
        stream.write_all(MSG).await.unwrap();
        stream.finish().unwrap();
        // Wait for the stream to be closed, one way or another.
        _ = stream.stopped().await;
    });

    let connection = {
        let _guard = tracing::error_span!("client").entered();
        client
            .connect(server_addr, "localhost")
            .unwrap()
            .await
            .unwrap()
    };
    info!("connected");
    connected_recv.notified().await;
    client
        .rebind(UdpSocket::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)).unwrap())
        .unwrap();
    info!("rebound");
    write_send.notify_one();
    let mut stream = connection.accept_uni().await.unwrap();
    assert_eq!(stream.read_to_end(MSG.len()).await.unwrap(), MSG);
    server.await.unwrap();
}

#[tokio::test]
async fn two_datagram_readers() {
    let _guard = subscribe();
    let endpoint = endpoint();

    let (client, server) = tokio::join!(
        endpoint
            .connect(endpoint.local_addr().unwrap(), "localhost")
            .unwrap(),
        async { endpoint.accept().await.unwrap().await }
    );
    let client = client.unwrap();
    let server = server.unwrap();

    let done = tokio::sync::Notify::new();
    let (a, b, ()) = tokio::join!(
        async {
            let x = client.read_datagram().await.unwrap();
            done.notify_waiters();
            x
        },
        async {
            let x = client.read_datagram().await.unwrap();
            done.notify_waiters();
            x
        },
        async {
            server.send_datagram(b"one"[..].into()).unwrap();
            done.notified().await;
            server.send_datagram_wait(b"two"[..].into()).await.unwrap();
        }
    );
    assert!(*a == *b"one" || *b == *b"one");
    assert!(*a == *b"two" || *b == *b"two");
}

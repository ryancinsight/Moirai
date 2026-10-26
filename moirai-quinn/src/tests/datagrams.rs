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
            let mut second = server.send_datagram_wait(b"two"[..].into());
            (&mut second).await.unwrap();
            // A completed future retains its result; polling it again must not panic or enqueue
            // a second copy of the datagram.
            (&mut second).await.unwrap();
        }
    );
    assert!(*a == *b"one" || *b == *b"one");
    assert!(*a == *b"two" || *b == *b"two");
}

#[tokio::test]
async fn repeated_rebind_keeps_two_live_connections_routable() {
    let _guard = subscribe();
    let factory = EndpointFactory::new();
    let client = factory.endpoint();
    let server = factory.endpoint();
    let server_addr = server.local_addr().expect("real server address");
    let mut pairs = Vec::new();
    for _ in 0..2 {
        let (outgoing, incoming) = tokio::join!(
            client.connect(server_addr, "localhost").expect("connect"),
            async { server.accept().await.expect("incoming").await }
        );
        pairs.push((
            outgoing.expect("client handshake"),
            incoming.expect("server handshake"),
        ));
    }

    for generation in 0..2 {
        let new_socket = UdpSocket::bind("127.0.0.1:0").expect("new real UDP socket");
        client.rebind(new_socket).expect("socket migration");
        for (index, (outgoing, incoming)) in pairs.iter().enumerate() {
            let request = Bytes::from(format!("generation-{generation}-connection-{index}"));
            let response = Bytes::from(format!("ack-{generation}-{index}"));
            timeout(Duration::from_secs(5), async {
                outgoing
                    .send_datagram_wait(request.clone())
                    .await
                    .expect("new-path datagram admission");
                assert_eq!(incoming.read_datagram().await.expect("request"), request);
                incoming
                    .send_datagram_wait(response.clone())
                    .await
                    .expect("reply admission");
                assert_eq!(outgoing.read_datagram().await.expect("reply"), response);
            })
            .await
            .expect("both connections must remain routable after rebind");
        }
    }
}

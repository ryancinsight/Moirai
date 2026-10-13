use super::*;

#[test]
fn configuration_rejects_invalid_bounds_and_deadlines() {
    let _trace = trace();
    for (alpn, frame, streams, connections, handshake, deadline, idle) in [
        (&b""[..], 37, 2, 1, 1200, DEADLINE, IDLE),
        (ALPN, 0, 2, 1, 1200, DEADLINE, IDLE),
        (ALPN, 37, 0, 1, 1200, DEADLINE, IDLE),
        (ALPN, 37, 2, 0, 1200, DEADLINE, IDLE),
        (ALPN, 37, 2, 1, 1199, DEADLINE, IDLE),
        (ALPN, 37, 2, 1, 1200, Duration::ZERO, IDLE),
        (ALPN, 37, 2, 1, 1200, IDLE, DEADLINE),
        (ALPN, u32::MAX, u32::MAX, 1, 1200, DEADLINE, IDLE),
        (ALPN, 37, 2, 1, 1200, DEADLINE, Duration::MAX),
    ] {
        let QuicError::InvalidConfig(reason) =
            QuicConfig::new(alpn, frame, streams, connections, handshake, deadline, idle)
                .expect_err("invalid bounds must be rejected")
        else {
            panic!("wrong configuration error")
        };
        assert!(!reason.is_empty());
    }
    let config = config(37, 2, DEADLINE);
    assert_eq!(config.max_frame_bytes(), 37);
    assert_eq!(config.max_streams(), 2);
    assert_eq!(config.operation_timeout(), DEADLINE);
}

#[test]
fn untrusted_certificate_wrong_hostname_and_alpn_fail_closed() {
    let _trace = trace();
    // RFC 9000 section 20.1: CRYPTO_ERROR = 0x100 + TLS alert code.
    for (name, trusted, alpn, alert) in [
        (
            "localhost",
            false,
            ALPN,
            rustls::AlertDescription::UnknownCA,
        ),
        (
            "wrong.example",
            true,
            ALPN,
            rustls::AlertDescription::BadCertificate,
        ),
        (
            "localhost",
            true,
            &b"wrong-protocol/1"[..],
            rustls::AlertDescription::NoApplicationProtocol,
        ),
    ] {
        let executor = executor();
        let (identity, roots) = material();
        let config = config(4096, 2, DEADLINE);
        let server = QuicEndpoint::server(
            Arc::clone(&executor),
            "127.0.0.1:0".parse().expect("loopback"),
            identity,
            None,
            config,
        )
        .expect("server");
        let client = QuicEndpoint::client(
            Arc::clone(&executor),
            "127.0.0.1:0".parse().expect("loopback"),
            if trusted {
                roots
            } else {
                rustls::RootCertStore::empty()
            },
            None,
            QuicConfig::new(alpn, 4096, 2, 4, 65_536, DEADLINE, IDLE).expect("client config"),
        )
        .expect("client");
        let (server_result, client_result) = moirai_executor::block_on(futures::future::join(
            server.accept(),
            client.connect(server.local_addr().expect("server address"), name),
        ));
        let client_error = match client_result {
            Ok(_) => panic!("invalid handshake accepted"),
            Err(error) => error,
        };
        let code = match client_error {
            QuicError::Connection(quinn::ConnectionError::TransportError(error)) => {
                u64::from(error.code)
            }
            QuicError::Connection(quinn::ConnectionError::ConnectionClosed(error)) => {
                u64::from(error.error_code)
            }
            error => panic!("expected local or peer TLS transport alert, got {error}"),
        };
        assert_eq!(code, 0x100 + u64::from(u8::from(alert)));
        match server_result {
            Err(QuicError::Connection(_)) => {}
            result => panic!(
                "server must reject the handshake, got {}",
                result
                    .err()
                    .map_or_else(|| "accepted".to_owned(), |error| error.to_string())
            ),
        }
    }
}

#[test]
fn hostile_prefix_and_truncation_abort_the_affected_stream() {
    let _trace = trace();
    let executor = executor();
    let (server, client) = endpoints(&executor, &config(37, 3, DEADLINE));
    let (server_connection, client_connection) = pair(&server, &client);
    moirai_executor::block_on(async {
        for wire in [
            vec![0, 0],
            38_u32.to_be_bytes().to_vec(),
            [37_u32.to_be_bytes().as_slice(), &[1, 2, 3]].concat(),
        ] {
            let mut sending = client_connection.open_stream().await.expect("stream");
            sending.send.write_all(&wire).await.expect("hostile bytes");
            sending.finish().expect("FIN");
            let mut receiving = server_connection.accept_stream().await.expect("accept");
            let error = receiving.recv_message().await.expect_err("hostile frame");
            if wire == 38_u32.to_be_bytes() {
                assert!(matches!(
                    error,
                    QuicError::MessageTooLarge {
                        size: 38,
                        limit: 37
                    }
                ));
            } else {
                assert!(matches!(
                    error,
                    QuicError::ReadExact(quinn::ReadExactError::FinishedEarly(_))
                ));
            }
            assert!(matches!(
                receiving.recv_message().await,
                Err(QuicError::Closed)
            ));
        }
    });
}

#[test]
fn stalled_read_deadline_aborts_frame_and_connection_admission_is_bounded() {
    let _trace = trace();
    let executor = executor();
    let config = QuicConfig::new(ALPN, 37, 1, 1, 65_536, Duration::from_millis(100), IDLE)
        .expect("bounded config");
    let (server, client) = endpoints(&executor, &config);
    let (server_connection, client_connection) = pair(&server, &client);
    moirai_executor::block_on(async {
        assert!(matches!(
            client
                .connect(server.local_addr().expect("server address"), "localhost")
                .await,
            Err(QuicError::ConnectionCapacity)
        ));
        let mut sending = client_connection.open_stream().await.expect("stream");
        sending
            .send
            .write_all(&37_u32.to_be_bytes())
            .await
            .expect("prefix only");
        let mut receiving = server_connection.accept_stream().await.expect("accept");
        assert!(matches!(
            receiving.recv_message().await,
            Err(QuicError::TimedOut("receive message"))
        ));
        assert!(matches!(
            receiving.recv_message().await,
            Err(QuicError::Closed)
        ));
        client.close();
        server.close();
        // Drain is separately bounded; QUIC's closing period can exceed this
        // deliberately short operation deadline and must report that timeout.
        if let Err(error) = client.wait_idle().await {
            assert!(matches!(error, QuicError::TimedOut("endpoint drain")));
        }
    });
}

#[test]
fn application_stream_admission_releases_only_when_its_owner_drops() {
    let _trace = trace();
    let executor = executor();
    let (server, client) = endpoints(&executor, &config(37, 1, DEADLINE));
    let (server_connection, client_connection) = pair(&server, &client);
    moirai_executor::block_on(async {
        let mut stream = client_connection
            .open_stream()
            .await
            .expect("first admitted stream");
        stream
            .send_message(&[])
            .await
            .expect("publish stream on wire");
        let peer = server_connection
            .accept_stream()
            .await
            .expect("peer owns the first stream");
        assert_eq!(peer.id(), 0);
        assert!(matches!(
            client_connection.open_stream().await,
            Err(QuicError::StreamCapacity)
        ));
        // Both owners retire: local application admission and the peer's QUIC
        // stream-credit contract are independent bounds, and both must release.
        drop(peer);
        drop(stream);
        let stream = client_connection
            .open_stream()
            .await
            .expect("owner releases permit");
        assert_eq!(stream.id(), 4); // Client bidirectional IDs are 0, 4, 8, ...
    });
}

#[test]
fn mutual_tls_admits_only_a_client_with_the_required_certificate() {
    let _trace = trace();
    for authenticated in [true, false] {
        let executor = executor();
        let (identity, roots) = material();
        let server = QuicEndpoint::server(
            Arc::clone(&executor),
            "127.0.0.1:0".parse().expect("loopback"),
            identity,
            Some(roots.clone()),
            config(37, 1, DEADLINE),
        )
        .expect("mutual TLS server");
        let client = QuicEndpoint::client(
            Arc::clone(&executor),
            "127.0.0.1:0".parse().expect("loopback"),
            roots,
            authenticated.then(|| material().0),
            config(37, 1, DEADLINE),
        )
        .expect("client");
        let (server_result, client_result) = moirai_executor::block_on(futures::future::join(
            server.accept(),
            client.connect(server.local_addr().expect("server address"), "localhost"),
        ));
        if authenticated {
            let server_connection = server_result.expect("authenticated client admitted");
            let client_connection = client_result.expect("authenticated server admitted");
            assert_eq!(
                server_connection
                    .peer_certificates()
                    .expect("verified client chain")
                    .len(),
                3
            );
            assert_eq!(
                client_connection
                    .peer_certificates()
                    .expect("verified server chain")
                    .len(),
                3
            );
        } else {
            let error = match server_result {
                Ok(_) => panic!("anonymous client admitted"),
                Err(error) => error,
            };
            let QuicError::Connection(quinn::ConnectionError::TransportError(error)) = error else {
                panic!("server must retain the missing-client-certificate alert");
            };
            assert_eq!(
                u64::from(error.code),
                0x100 + u64::from(u8::from(rustls::AlertDescription::CertificateRequired))
            );
        }
    }
}

#[test]
fn expired_leaf_is_rejected_even_when_its_root_is_trusted() {
    let _trace = trace();
    let executor = executor();
    let certificates =
        CertificateDer::pem_slice_iter(include_bytes!("../../../tests/fixtures/expired-cert.pem"))
            .collect::<Result<Vec<_>, _>>()
            .expect("permanently expired published fixture");
    let mut roots = rustls::RootCertStore::empty();
    roots
        .add(certificates.last().expect("valid root").clone())
        .expect("trust expired fixture root");
    let key =
        PrivateKeyDer::from_pem_slice(include_bytes!("../../../tests/fixtures/expired-key.pem"))
            .expect("expired fixture public key material");
    let server = QuicEndpoint::server(
        Arc::clone(&executor),
        "127.0.0.1:0".parse().expect("loopback"),
        Identity::new(certificates, key),
        None,
        config(37, 1, DEADLINE),
    )
    .expect("server identity");
    let client = QuicEndpoint::client(
        Arc::clone(&executor),
        "127.0.0.1:0".parse().expect("loopback"),
        roots,
        None,
        config(37, 1, DEADLINE),
    )
    .expect("client trust");
    let (_server_result, client_result) = moirai_executor::block_on(futures::future::join(
        server.accept(),
        client.connect(server.local_addr().expect("server address"), "localhost"),
    ));
    let error = match client_result {
        Ok(_) => panic!("expired leaf admitted"),
        Err(error) => error,
    };
    let QuicError::Connection(quinn::ConnectionError::TransportError(error)) = error else {
        panic!("certificate expiry error lost");
    };
    assert_eq!(
        u64::from(error.code),
        0x100 + u64::from(u8::from(rustls::AlertDescription::CertificateExpired))
    );
}

#[test]
fn stopped_real_executor_rejects_endpoint_without_panicking_or_masking_error() {
    let _trace = trace();
    let mut executor = moirai_executor::ExecutorBuilder::new()
        .worker_threads(1)
        .build()
        .expect("executor");
    executor.shutdown().expect("stop real executor");
    let (_, roots) = material();
    let error = match QuicEndpoint::client(
        Arc::new(executor),
        "127.0.0.1:0".parse().expect("loopback"),
        roots,
        None,
        config(37, 1, DEADLINE),
    ) {
        Ok(_) => panic!("stopped spawner must reject protocol driver admission"),
        Err(error) => error,
    };
    let QuicError::Executor(error) = error else {
        panic!("executor rejection context was lost")
    };
    assert_eq!(*error, moirai_core::error::ExecutorError::ShuttingDown);
}

#[test]
fn exhausted_send_window_times_out_without_unbounded_queue_growth() {
    let _trace = trace();
    let executor = executor();
    let (server, client) = endpoints(&executor, &config(37, 1, DEADLINE));
    let (server_connection, client_connection) = pair(&server, &client);
    moirai_executor::block_on(async {
        let mut sender = client_connection.open_stream().await.expect("sender");
        sender
            .send_message(&payload(37))
            .await
            .expect("first frame fills peer receive window");
        let _receiving = server_connection
            .accept_stream()
            .await
            .expect("peer holds unread frame");
        // ACK retires packets but does not increase MAX_STREAM_DATA. The first
        // 37-byte message plus its four-byte prefix exhausts all 41 credits.
        let error = sender
            .send_message(&payload(37))
            .await
            .expect_err("peer never advances stream receive credit");
        assert!(matches!(error, QuicError::TimedOut("send message")));
        assert!(matches!(
            sender.send_message(&payload(37)).await,
            Err(QuicError::Closed)
        ));
    });
}

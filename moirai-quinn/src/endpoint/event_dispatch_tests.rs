use super::*;

#[test]
fn drained_connection_event_is_not_delivered_or_resurrected() {
    let (endpoint_sender, _endpoint_receiver) = mpsc::unbounded_channel();
    let mut connections = ConnectionSet {
        senders: FxHashMap::default(),
        sender: endpoint_sender,
        close: None,
    };
    let handle = ConnectionHandle(42);
    let event = || ConnectionEvent::Close {
        error_code: VarInt::from_u32(9),
        reason: Bytes::from_static(b"drained"),
    };

    assert!(!connections.dispatch_if_live(handle, event()));
    assert!(connections.senders.is_empty());

    let (sender, mut receiver) = mpsc::unbounded_channel();
    connections.senders.insert(handle, sender);
    assert!(connections.dispatch_if_live(handle, event()));
    match receiver.try_recv() {
        Some(ConnectionEvent::Close { error_code, reason }) => {
            assert_eq!(error_code, VarInt::from_u32(9));
            assert_eq!(reason, Bytes::from_static(b"drained"));
        }
        other => panic!("live connection did not receive its exact event: {other:?}"),
    }
    connections.senders.remove(&handle);
    assert!(!connections.dispatch_if_live(handle, event()));
    assert!(receiver.try_recv().is_none());
}

#[tokio::test]
async fn poisoned_endpoint_fails_closed_and_can_teardown() {
    let socket = std::net::UdpSocket::bind("127.0.0.1:0").expect("local UDP socket");
    let endpoint = Endpoint::new(
        EndpointConfig::default(),
        None,
        socket,
        Arc::new(crate::runtime::TokioRuntime),
    )
    .expect("local endpoint");
    assert!(endpoint.local_addr().expect("bound local socket").port() > 0);

    let original_panic = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _state = endpoint.inner.state.lock().expect("unpoisoned setup");
        panic!("deliberate endpoint-state poison");
    }))
    .expect_err("the original panic must propagate");
    assert_eq!(
        original_panic.downcast_ref::<&str>(),
        Some(&"deliberate endpoint-state poison")
    );
    assert!(endpoint.inner.state.is_poisoned());
    assert_eq!(
        endpoint.local_addr().expect_err("poisoned endpoint").kind(),
        io::ErrorKind::Other
    );
    assert!(endpoint.accept().await.is_none());
    drop(endpoint);
}

#[tokio::test]
async fn retired_sockets_wait_for_each_connection_and_each_generation() {
    let h1 = ConnectionHandle(1);
    let h2 = ConnectionHandle(2);
    let mut retiring = Vec::new();
    for _ in 0..3 {
        let socket = std::net::UdpSocket::bind("127.0.0.1:0").expect("old local UDP socket");
        retiring.push(RetiringSocket {
            socket: crate::runtime::TokioRuntime
                .wrap_udp_socket(socket)
                .expect("wrapped old socket"),
            pending: FxHashSet::from_iter([h1, h2]),
        });
    }
    assert_eq!(retiring.len(), 3);
    assert!(retiring.iter().all(|old| old.pending.len() == 2));

    // A packet on the middle generation retires only older sockets for h1.
    observe_connection_on(&mut retiring, 1, h1);
    assert!(!retiring[0].pending.contains(&h1));
    assert!(retiring[1].pending.contains(&h1));
    assert!(retiring[2].pending.contains(&h1));
    assert!(retiring.iter().all(|old| old.pending.contains(&h2)));

    // Reaching the current socket for h2 must not discard any socket still needed by h1.
    let current_generation = retiring.len();
    observe_connection_on(&mut retiring, current_generation, h2);
    assert!(retiring.iter().all(|old| !old.pending.contains(&h2)));
    assert!(retiring[1].pending.contains(&h1));
    assert!(retiring[2].pending.contains(&h1));

    observe_connection_on(&mut retiring, current_generation, h1);
    assert!(retiring.iter().all(|old| old.pending.is_empty()));
}

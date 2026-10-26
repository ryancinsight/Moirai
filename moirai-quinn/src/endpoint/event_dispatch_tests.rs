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

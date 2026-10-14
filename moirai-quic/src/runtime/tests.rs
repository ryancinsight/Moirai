use super::*;
use moirai_pal::reactor::IoReactor;
use std::net::UdpSocket;
use std::task::Waker;
use std::time::Duration;

#[test]
fn writable_hint_survives_successful_datagrams_without_per_packet_rearming() {
    let subscriber = tracing_subscriber::fmt()
        .with_test_writer()
        .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
        .finish();
    let _trace = tracing::subscriber::set_default(subscriber);
    let arrange = tracing::info_span!("arrange").entered();
    let executor = Arc::new(
        moirai_executor::ExecutorBuilder::new()
            .worker_threads(1)
            .build()
            .expect("executor"),
    );
    let runtime = MoiraiRuntime {
        executor,
        health: Arc::new(RuntimeHealth::default()),
    };
    let reactor = IoReactor::new().expect("reactor");
    let peer = UdpSocket::bind("127.0.0.1:0").expect("real UDP peer");
    peer.set_read_timeout(Some(Duration::from_secs(1)))
        .expect("finite peer deadline");
    let socket = runtime
        .wrap_udp_socket(UdpSocket::bind("127.0.0.1:0").expect("real UDP sender"))
        .expect("Moirai socket adapter");
    let mut poller = Arc::clone(&socket).create_io_poller();
    let mut cx = Context::from_waker(Waker::noop());
    drop(arrange);
    reactor.with_active(|| {
        let _act = tracing::info_span!("act").entered();
        assert!(poller.as_mut().poll_writable(&mut cx).is_pending());
        reactor
            .run_iteration(Some(Duration::from_secs(1)))
            .expect("native writable event");
        assert!(matches!(
            poller.as_mut().poll_writable(&mut cx),
            Poll::Ready(Ok(()))
        ));
        for length in [0, 1, 37, 1200] {
            let bytes = crate::tests::payload(length);
            // Readiness stays cached until actual kernel I/O returns WouldBlock.
            // The preceding implementation returns Pending here on every packet.
            assert!(matches!(
                poller.as_mut().poll_writable(&mut cx),
                Poll::Ready(Ok(()))
            ));
            socket
                .try_send(&Transmit {
                    destination: peer.local_addr().expect("peer address"),
                    ecn: None,
                    contents: &bytes,
                    segment_size: None,
                    src_ip: None,
                })
                .expect("actual datagram send");
            let mut received = vec![0; 1200];
            let (received_len, source) =
                peer.recv_from(&mut received).expect("actual peer datagram");
            let _assert = tracing::info_span!("assert", length).entered();
            assert_eq!(&received[..received_len], bytes);
            assert_eq!(source, socket.local_addr().expect("sender address"));
        }
    });
}

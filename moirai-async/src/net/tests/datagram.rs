use super::*;

#[test]
fn test_udp_loopback_send_recv_and_stats_values() {
    futures::executor::block_on(async {
        let sender = UdpSocket::bind("127.0.0.1:0")
            .await
            .expect("sender bind must succeed");
        let receiver = UdpSocket::bind("127.0.0.1:0")
            .await
            .expect("receiver bind must succeed");
        let target = receiver.local_addr().expect("receiver address must exist");
        let source = sender.local_addr().expect("sender address must exist");

        let sent = sender
            .send_to(b"datagram", target)
            .await
            .expect("udp send must succeed");
        assert_eq!(sent, 8);

        let deadline = Instant::now() + Duration::from_secs(2);
        let mut buf = [0_u8; 16];
        let (received, peer) = loop {
            match receiver.recv_from(&mut buf).await {
                Ok(received) => break received,
                Err(error)
                    if error.kind() == io::ErrorKind::WouldBlock && Instant::now() < deadline =>
                {
                    std::thread::yield_now();
                }
                Err(error) => panic!("udp receive failed: {error}"),
            }
        };

        assert_eq!(received, 8);
        assert_eq!(peer, source);
        assert_eq!(&buf[..received], b"datagram");

        let send_stats = sender.stats();
        assert_eq!(send_stats.packets_sent, 1);
        assert_eq!(send_stats.bytes_sent, 8);
        assert_eq!(send_stats.packets_received, 0);
        assert_eq!(send_stats.bytes_received, 0);

        let recv_stats = receiver.stats();
        assert_eq!(recv_stats.packets_sent, 0);
        assert_eq!(recv_stats.bytes_sent, 0);
        assert_eq!(recv_stats.packets_received, 1);
        assert_eq!(recv_stats.bytes_received, 8);
    });
}

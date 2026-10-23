use super::*;
use moirai_pal::net::{UdpBufferConfig, UdpBufferSizes};
use socket2::SockRef;
use std::net::UdpSocket as NativeSocket;
use std::num::NonZeroUsize;
use std::time::Duration;

fn trace() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_test_writer()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .finish(),
    )
}

#[test]
fn configured_unicast_preserves_empty_datagrams_bytes_sources_and_counters() {
    let _trace = trace();
    let _test = tracing::info_span!("configured_udp_exchange").entered();
    let payload: Vec<_> = (0..=u8::MAX).collect();
    for address in ["127.0.0.1:0", "[::1]:0"] {
        futures::executor::block_on(async {
            let config = UdpConfig {
                buffers: UdpBufferConfig {
                    receive: NonZeroUsize::new(8192),
                    send: NonZeroUsize::new(4096),
                },
                ..UdpConfig::default()
            };
            let sender = UdpSocket::bind_with_config(address, config)
                .await
                .expect("configured sender");
            let receiver = UdpSocket::bind_with_config(address, config)
                .await
                .expect("configured receiver");
            assert_eq!(*receiver.config(), config);
            let target = receiver.local_addr().expect("receiver address");
            let source = sender.local_addr().expect("sender address");
            for message in [&[][..], &payload[..1], &payload[..]] {
                let sent = sender
                    .send_to(message, target)
                    .await
                    .expect("complete datagram send");
                let mut actual = [0x37; 256];
                let (received, peer) =
                    crate::timer::timeout(Duration::from_secs(2), receiver.recv_from(&mut actual))
                        .await
                        .expect("finite receive deadline")
                        .expect("complete datagram receive");
                tracing::info_span!("assert").in_scope(|| {
                    assert_eq!(sent, message.len());
                    assert_eq!(received, message.len());
                    assert_eq!(peer, source);
                    assert_eq!(&actual[..received], message);
                    if message.is_empty() {
                        assert_eq!(actual, [0x37; 256]);
                    }
                });
            }
            let sent = sender.stats();
            let received = receiver.stats();
            assert_eq!(sent.packets_sent, 3);
            assert_eq!(sent.bytes_sent, 257);
            assert_eq!(received.packets_received, sent.packets_sent);
            assert_eq!(received.bytes_received, sent.bytes_sent);
            assert_eq!(sent.packets_received, 0);
            assert_eq!(received.packets_sent, 0);
        });
    }
}

#[test]
fn rejected_datagram_retains_native_error_and_does_not_count_success() {
    let _trace = trace();
    futures::executor::block_on(async {
        let sender = UdpSocket::bind_with_config(
            "127.0.0.1:0",
            UdpConfig {
                buffers: UdpBufferConfig {
                    send: NonZeroUsize::new(8192),
                    receive: None,
                },
                ..UdpConfig::default()
            },
        )
        .await
        .expect("configured socket");
        let receiver = NativeSocket::bind("127.0.0.1:0").expect("real receiver");
        // UDP's 16-bit length includes its header; 2^16 payload bytes exceed it.
        let payload = vec![0x37; usize::from(u16::MAX) + 1];
        let error = sender
            .send_to(&payload, receiver.local_addr().expect("target"))
            .await
            .expect_err("oversized datagram");
        assert_ne!(error.raw_os_error().expect("native datagram error"), 0);
        let stats = sender.stats();
        assert_eq!(stats.packets_sent, 0);
        assert_eq!(stats.bytes_sent, 0);
        let sent = sender
            .send_to(b"after-rejection", receiver.local_addr().expect("target"))
            .await
            .expect("next valid datagram");
        let mut actual = [0x37; 15];
        receiver
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("finite receive");
        let (received, peer) = receiver.recv_from(&mut actual).expect("computed datagram");
        assert_eq!(sent, 15);
        assert_eq!(received, 15);
        assert_eq!(actual, *b"after-rejection");
        assert_eq!(peer, sender.local_addr().expect("source"));
        assert_eq!(sender.stats().packets_sent, 1);
        assert_eq!(sender.stats().bytes_sent, 15);
    });
}

#[test]
fn configured_buffers_match_native_socket_options() {
    let _trace = trace();
    let _test = tracing::info_span!("configured_udp_buffers").entered();
    for address in ["127.0.0.1:0", "[::1]:0"] {
        let native = tracing::info_span!("arrange").in_scope(|| {
            let native = NativeSocket::bind(address).expect("native reference socket");
            let options = SockRef::from(&native);
            options
                .set_recv_buffer_size(4096)
                .expect("native receive option");
            options
                .set_send_buffer_size(4096)
                .expect("native send option");
            native
        });
        let actual = tracing::info_span!("act").in_scope(|| {
            futures::executor::block_on(UdpSocket::bind_with_config(
                address,
                UdpConfig {
                    buffers: UdpBufferConfig {
                        receive: NonZeroUsize::new(4096),
                        send: NonZeroUsize::new(4096),
                    },
                    ..UdpConfig::default()
                },
            ))
            .expect("configured async socket")
        });
        tracing::info_span!("assert").in_scope(|| {
            let expected = SockRef::from(&native);
            let observed = SockRef::from(actual.inner.as_std());
            assert_eq!(
                observed.recv_buffer_size().expect("receive readback"),
                expected.recv_buffer_size().expect("reference receive")
            );
            assert_eq!(
                observed.send_buffer_size().expect("send readback"),
                expected.send_buffer_size().expect("reference send")
            );
            assert_eq!(
                actual.buffer_sizes().expect("public readback"),
                UdpBufferSizes::read(&native).expect("reference options")
            );
        });
    }
}

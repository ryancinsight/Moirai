use super::*;
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
fn independent_buffers_and_defaults_match_native_options() {
    let _trace = trace();
    let _test = tracing::info_span!("udp_native_buffer_contract").entered();
    for address in ["127.0.0.1:0", "[::1]:0"] {
        for (receive, send) in [
            (None, None),
            (Some(4096), None),
            (None, Some(8192)),
            (Some(4096), Some(8192)),
        ] {
            let config = UdpConfig {
                buffers: UdpBufferConfig {
                    receive: receive.and_then(NonZeroUsize::new),
                    send: send.and_then(NonZeroUsize::new),
                },
                ..UdpConfig::default()
            };
            let reference = tracing::info_span!("arrange").in_scope(|| {
                let socket = UdpSocket::bind(address).expect("native reference");
                let options = SockRef::from(&socket);
                if let Some(size) = receive {
                    options.set_recv_buffer_size(size).expect("receive request");
                }
                if let Some(size) = send {
                    options.set_send_buffer_size(size).expect("send request");
                }
                socket
            });
            let socket = tracing::info_span!("act").in_scope(|| {
                config
                    .bind(address.parse().expect("address"))
                    .expect("configured socket")
            });
            tracing::info_span!("assert").in_scope(|| {
                assert_eq!(
                    UdpBufferSizes::read(&socket).expect("effective options"),
                    UdpBufferSizes::read(&reference).expect("native options")
                );
                assert!(!socket.broadcast().expect("broadcast readback"));
                let mut byte = [0x37];
                assert_eq!(
                    socket
                        .recv_from(&mut byte)
                        .expect_err("nonblocking empty queue")
                        .kind(),
                    io::ErrorKind::WouldBlock
                );
                assert_eq!(byte, [0x37]);
            });
        }
    }
}

#[test]
fn native_integer_overflow_is_rejected_before_socket_options() {
    let _trace = trace();
    let _test = tracing::info_span!("udp_buffer_representation").entered();
    let maximum = usize::try_from(i32::MAX).expect("native signed maximum");
    for size in [maximum + 1, usize::MAX] {
        for buffers in [
            UdpBufferConfig {
                receive: NonZeroUsize::new(size),
                send: None,
            },
            UdpBufferConfig {
                receive: None,
                send: NonZeroUsize::new(size),
            },
        ] {
            let error = tracing::info_span!("act").in_scope(|| {
                UdpConfig {
                    buffers,
                    ..UdpConfig::default()
                }
                .bind("127.0.0.1:0".parse().expect("address"))
                .expect_err("overflow must be rejected")
            });
            tracing::info_span!("assert").in_scope(|| {
                assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
                assert_eq!(
                    error.to_string(),
                    "UDP buffer exceeds native signed integer"
                );
            });
        }
    }
    // Boundary validation performs no allocation or socket syscall.
    UdpBufferConfig {
        receive: NonZeroUsize::new(maximum),
        send: NonZeroUsize::new(maximum),
    }
    .validate()
    .expect("inclusive boundary");
}

#[test]
fn invalid_group_family_interface_and_broadcast_are_rejected() {
    let _trace = trace();
    let group = Ipv4Addr::new(239, 255, 0, 77);
    let group_ipv6 = "ff02::77".parse().expect("IPv6 multicast");
    for (address, multicast) in [
        (
            "127.0.0.1:0",
            MulticastConfig::V4 {
                group: Ipv4Addr::LOCALHOST,
                interface: Ipv4Addr::LOCALHOST,
            },
        ),
        (
            "127.0.0.1:0",
            MulticastConfig::V4 {
                group,
                interface: group,
            },
        ),
        (
            "127.0.0.1:0",
            MulticastConfig::V4 {
                group,
                interface: Ipv4Addr::BROADCAST,
            },
        ),
        (
            "127.0.0.1:0",
            MulticastConfig::V6 {
                group: group_ipv6,
                interface: 0,
            },
        ),
        (
            "[::1]:0",
            MulticastConfig::V4 {
                group,
                interface: Ipv4Addr::LOCALHOST,
            },
        ),
        (
            "[::1]:0",
            MulticastConfig::V6 {
                group: Ipv6Addr::LOCALHOST,
                interface: 0,
            },
        ),
    ] {
        let error = tracing::info_span!("act").in_scope(|| {
            UdpConfig {
                multicast: Some(multicast),
                ..UdpConfig::default()
            }
            .bind(address.parse().expect("address"))
            .expect_err("invalid group contract")
        });
        tracing::info_span!("assert")
            .in_scope(|| assert_eq!(error.kind(), io::ErrorKind::InvalidInput));
    }
    let error = UdpConfig {
        broadcast: true,
        ..UdpConfig::default()
    }
    .bind("[::1]:0".parse().expect("address"))
    .expect_err("IPv6 has no broadcast");
    assert_eq!(error.kind(), io::ErrorKind::InvalidInput);
    assert_eq!(error.to_string(), "IPv6 has no broadcast");
    let socket = UdpConfig {
        broadcast: true,
        ..UdpConfig::default()
    }
    .bind("127.0.0.1:0".parse().expect("address"))
    .expect("IPv4 broadcast option");
    assert!(socket.broadcast().expect("native broadcast option"));
}

#[test]
fn native_membership_error_releases_the_bound_socket() {
    let _trace = trace();
    let reserved = UdpSocket::bind("[::1]:0").expect("reserve address");
    let address = reserved.local_addr().expect("reserved address");
    drop(reserved);
    let error = tracing::info_span!("act").in_scope(|| {
        UdpConfig {
            multicast: Some(MulticastConfig::V6 {
                group: "ff02::77".parse().expect("multicast"),
                interface: u32::MAX,
            }),
            ..UdpConfig::default()
        }
        .bind(address)
        .expect_err("invalid native interface index")
    });
    tracing::info_span!("assert").in_scope(|| {
        assert_ne!(error.raw_os_error().expect("preserved OS failure"), 0);
        let socket = UdpSocket::bind(address).expect("failed configuration closes its socket");
        assert_eq!(socket.local_addr().expect("rebound address"), address);
    });
}

#[test]
fn multicast_membership_delivers_exact_datagrams_on_the_selected_interface() {
    let _trace = trace();
    let _test = tracing::info_span!("udp_multicast_delivery").entered();
    let group = Ipv4Addr::new(239, 255, 0, 77);
    let config = UdpConfig {
        multicast: Some(MulticastConfig::V4 {
            group,
            interface: Ipv4Addr::LOCALHOST,
        }),
        ..UdpConfig::default()
    };
    let (receiver, sender) = tracing::info_span!("arrange").in_scope(|| {
        let receiver = config
            .bind("0.0.0.0:0".parse().expect("wildcard address"))
            .expect("joined receiver");
        let sender = config
            .bind("127.0.0.1:0".parse().expect("interface address"))
            .expect("joined sender");
        receiver
            .set_nonblocking(false)
            .expect("bounded blocking reference receive");
        receiver
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("finite receive deadline");
        assert_eq!(
            SockRef::from(&sender)
                .multicast_if_v4()
                .expect("outgoing interface"),
            Ipv4Addr::LOCALHOST
        );
        (receiver, sender)
    });
    let target = SocketAddr::from((group, receiver.local_addr().expect("receiver port").port()));
    let payload: Vec<_> = (0..=u8::MAX).collect();
    for message in [&[][..], &payload[..]] {
        let (sent, received, peer, actual) = tracing::info_span!("act").in_scope(|| {
            let sent = sender.send_to(message, target).expect("multicast send");
            let mut actual = [0x37; 256];
            let (received, peer) = receiver
                .recv_from(&mut actual)
                .expect("joined multicast receive");
            (sent, received, peer, actual)
        });
        tracing::info_span!("assert").in_scope(|| {
            assert_eq!(sent, message.len());
            assert_eq!(received, message.len());
            assert_eq!(peer, sender.local_addr().expect("sender address"));
            assert_eq!(&actual[..received], message);
            if message.is_empty() {
                assert_eq!(actual, [0x37; 256]);
            }
        });
    }
}

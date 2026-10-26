#![cfg(any(feature = "rustls-aws-lc-rs", feature = "rustls-ring"))]

#[cfg(all(feature = "rustls-aws-lc-rs", not(feature = "rustls-ring")))]
use rustls::crypto::aws_lc_rs::default_provider;
#[cfg(feature = "rustls-ring")]
use rustls::crypto::ring::default_provider;

use std::{
    convert::TryInto,
    future::Future,
    io,
    net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr, UdpSocket},
    pin::pin,
    str,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    task::{Context, Poll, RawWaker, RawWakerVTable, Waker},
};

use crate::runtime::TokioRuntime;
use crate::{Duration, Instant};
use bytes::Bytes;
use proto::{crypto::rustls::QuicClientConfig, RandomConnectionIdGenerator};
use rand::{rngs::StdRng, Rng, SeedableRng};
use rustls::{
    pki_types::{CertificateDer, PrivateKeyDer, PrivatePkcs8KeyDer},
    RootCertStore,
};
use tokio::time::{sleep, timeout};
use tokio::{
    join,
    runtime::{Builder, Runtime},
};
use tracing::{error_span, info};
use tracing_futures::Instrument as _;
use tracing_subscriber::EnvFilter;

use super::{ClientConfig, Endpoint, EndpointConfig, RecvStream, SendStream, TransportConfig};

/// Construct an endpoint suitable for connecting to itself
fn endpoint() -> Endpoint {
    EndpointFactory::new().endpoint()
}

fn endpoint_with_config(transport_config: TransportConfig) -> Endpoint {
    EndpointFactory::new().endpoint_with_config(transport_config)
}

/// Constructs endpoints suitable for connecting to themselves and each other
struct EndpointFactory {
    cert: rcgen::CertifiedKey<rcgen::KeyPair>,
    endpoint_config: EndpointConfig,
}

impl EndpointFactory {
    fn new() -> Self {
        Self {
            cert: rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap(),
            endpoint_config: EndpointConfig::default(),
        }
    }

    fn endpoint(&self) -> Endpoint {
        self.endpoint_with_config(TransportConfig::default())
    }

    fn endpoint_with_config(&self, transport_config: TransportConfig) -> Endpoint {
        let key = PrivateKeyDer::Pkcs8(self.cert.signing_key.serialize_der().into());
        let transport_config = Arc::new(transport_config);
        let mut server_config =
            crate::ServerConfig::with_single_cert(vec![self.cert.cert.der().clone()], key).unwrap();
        server_config.transport_config(transport_config.clone());

        let mut roots = rustls::RootCertStore::empty();
        roots.add(self.cert.cert.der().clone()).unwrap();
        let mut endpoint = Endpoint::new(
            self.endpoint_config.clone(),
            Some(server_config),
            UdpSocket::bind(SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), 0)).unwrap(),
            Arc::new(TokioRuntime),
        )
        .unwrap();
        let mut client_config = ClientConfig::with_root_certificates(Arc::new(roots)).unwrap();
        client_config.transport_config(transport_config);
        endpoint.set_default_client_config(client_config);

        endpoint
    }
}

fn run_echo(args: EchoArgs) {
    let _guard = subscribe();
    let runtime = rt_basic();
    let handle = {
        // Use small receive windows
        let mut transport_config = TransportConfig::default();
        if let Some(receive_window) = args.receive_window {
            transport_config.receive_window(receive_window.try_into().unwrap());
        }
        if let Some(stream_receive_window) = args.stream_receive_window {
            transport_config.stream_receive_window(stream_receive_window.try_into().unwrap());
        }
        transport_config.max_concurrent_bidi_streams(1_u8.into());
        transport_config.max_concurrent_uni_streams(1_u8.into());
        let transport_config = Arc::new(transport_config);

        // We don't use the `endpoint` helper here because we want two different endpoints with
        // different addresses.
        let cert = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
        let key = PrivatePkcs8KeyDer::from(cert.signing_key.serialize_der());
        let cert = CertificateDer::from(cert.cert);
        let mut server_config =
            crate::ServerConfig::with_single_cert(vec![cert.clone()], key.into()).unwrap();

        server_config.transport = transport_config.clone();
        let server_sock = UdpSocket::bind(args.server_addr).unwrap();
        let server_addr = server_sock.local_addr().unwrap();
        let server = {
            let _guard = runtime.enter();
            let _guard = error_span!("server").entered();
            Endpoint::new(
                Default::default(),
                Some(server_config),
                server_sock,
                Arc::new(TokioRuntime),
            )
            .unwrap()
        };

        let mut roots = rustls::RootCertStore::empty();
        roots.add(cert).unwrap();
        let mut client_crypto =
            rustls::ClientConfig::builder_with_provider(default_provider().into())
                .with_safe_default_protocol_versions()
                .unwrap()
                .with_root_certificates(roots)
                .with_no_client_auth();
        client_crypto.key_log = Arc::new(rustls::KeyLogFile::new());

        let mut client = {
            let _guard = runtime.enter();
            let _guard = error_span!("client").entered();
            Endpoint::client(args.client_addr).unwrap()
        };
        let mut client_config =
            ClientConfig::new(Arc::new(QuicClientConfig::try_from(client_crypto).unwrap()));
        client_config.transport_config(transport_config);
        client.set_default_client_config(client_config);

        let handle = runtime.spawn(async move {
            let incoming = server.accept().await.unwrap();

            // Note for anyone modifying the platform support in this test:
            // If `local_ip` gets available on additional platforms - which
            // requires modifying this test - please update the list of supported
            // platforms in the doc comment of `quinn_udp::RecvMeta::dst_ip`.
            if cfg!(target_os = "linux")
                || cfg!(target_os = "android")
                || cfg!(target_os = "freebsd")
                || cfg!(target_os = "openbsd")
                || cfg!(target_os = "netbsd")
                || cfg!(target_os = "macos")
                || cfg!(target_os = "windows")
            {
                let local_ip = incoming.local_ip().expect("Local IP must be available");
                assert!(local_ip.is_loopback());
            } else {
                assert_eq!(None, incoming.local_ip());
            }

            let new_conn = incoming.await.unwrap();
            tokio::spawn(async move {
                while let Ok(stream) = new_conn.accept_bi().await {
                    tokio::spawn(echo(stream));
                }
            });
            server.wait_idle().await;
        });

        info!("connecting from {} to {}", args.client_addr, server_addr);
        runtime.block_on(
            async move {
                let new_conn = client
                    .connect(server_addr, "localhost")
                    .unwrap()
                    .await
                    .expect("connect");

                /// This is just an arbitrary number to generate deterministic test data
                const SEED: u64 = 0x12345678;

                for i in 0..args.nr_streams {
                    println!("Opening stream {i}");
                    let (mut send, mut recv) = new_conn.open_bi().await.expect("stream open");
                    let msg = gen_data(args.stream_size, SEED);

                    let send_task = async {
                        send.write_all(&msg).await.expect("write");
                        send.finish().unwrap();
                    };
                    let recv_task = async { recv.read_to_end(usize::MAX).await.expect("read") };

                    let (_, data) = tokio::join!(send_task, recv_task);

                    assert_eq!(data[..], msg[..], "Data mismatch");
                }
                new_conn.close(0u32.into(), b"done");
                client.wait_idle().await;
            }
            .instrument(error_span!("client")),
        );
        handle
    };
    runtime.block_on(handle).unwrap();
}

struct EchoArgs {
    client_addr: SocketAddr,
    server_addr: SocketAddr,
    nr_streams: usize,
    stream_size: usize,
    receive_window: Option<u64>,
    stream_receive_window: Option<u64>,
}

async fn echo((mut send, mut recv): (SendStream, RecvStream)) {
    loop {
        // These are 32 buffers, for reading approximately 32kB at once
        #[rustfmt::skip]
        let mut bufs = [
            Bytes::new(), Bytes::new(), Bytes::new(), Bytes::new(),
            Bytes::new(), Bytes::new(), Bytes::new(), Bytes::new(),
            Bytes::new(), Bytes::new(), Bytes::new(), Bytes::new(),
            Bytes::new(), Bytes::new(), Bytes::new(), Bytes::new(),
            Bytes::new(), Bytes::new(), Bytes::new(), Bytes::new(),
            Bytes::new(), Bytes::new(), Bytes::new(), Bytes::new(),
            Bytes::new(), Bytes::new(), Bytes::new(), Bytes::new(),
            Bytes::new(), Bytes::new(), Bytes::new(), Bytes::new(),
        ];

        match recv.read_chunks(&mut bufs).await.expect("read chunks") {
            Some(n) => {
                send.write_all_chunks(&mut bufs[..n])
                    .await
                    .expect("write chunks");
            }
            None => break,
        }
    }

    let _ = send.finish();
}

fn gen_data(size: usize, seed: u64) -> Vec<u8> {
    let mut rng: StdRng = SeedableRng::seed_from_u64(seed);
    let mut buf = vec![0; size];
    rng.fill_bytes(&mut buf);
    buf
}

fn subscribe() -> tracing::subscriber::DefaultGuard {
    let sub = tracing_subscriber::FmtSubscriber::builder()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(|| TestWriter)
        .finish();
    tracing::subscriber::set_default(sub)
}

struct TestWriter;

impl std::io::Write for TestWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        print!(
            "{}",
            str::from_utf8(buf).expect("tried to log invalid UTF-8")
        );
        Ok(buf.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        io::stdout().flush()
    }
}

fn rt_basic() -> Runtime {
    Builder::new_current_thread().enable_all().build().unwrap()
}

fn rt_threaded() -> Runtime {
    Builder::new_multi_thread().enable_all().build().unwrap()
}

#[derive(Default)]
struct WakeCounter {
    wakes: AtomicUsize,
}

impl WakeCounter {
    fn wakes(&self) -> usize {
        self.wakes.load(Ordering::SeqCst)
    }
}

fn new_count_waker() -> (Waker, Arc<WakeCounter>) {
    // instance of WakeCounter
    let counter = Arc::new(WakeCounter::default());

    // convert
    let waker = unsafe { Waker::from_raw(raw_waker(counter.clone())) };
    (waker, counter)
}

fn raw_waker(counter: Arc<WakeCounter>) -> RawWaker {
    // Store an Arc<WakeCounter> behind the raw pointer.
    let ptr = Arc::into_raw(counter) as *const ();
    RawWaker::new(ptr, &VTABLE)
}

static VTABLE: RawWakerVTable =
    RawWakerVTable::new(clone_waker, wake_waker, wake_by_ref_waker, drop_waker);

unsafe fn clone_waker(data: *const ()) -> RawWaker {
    let arc = Arc::<WakeCounter>::from_raw(data as *const WakeCounter);
    let cloned = arc.clone();
    std::mem::forget(arc);
    raw_waker(cloned)
}

unsafe fn wake_waker(data: *const ()) {
    let arc = Arc::<WakeCounter>::from_raw(data as *const WakeCounter);
    arc.wakes.fetch_add(1, Ordering::SeqCst);
    // arc drops here
}

unsafe fn wake_by_ref_waker(data: *const ()) {
    let arc = Arc::<WakeCounter>::from_raw(data as *const WakeCounter);
    arc.wakes.fetch_add(1, Ordering::SeqCst);
    std::mem::forget(arc);
}

unsafe fn drop_waker(data: *const ()) {
    drop(Arc::<WakeCounter>::from_raw(data as *const WakeCounter));
}

mod datagrams;
mod echo;
mod handshake;
mod streams;

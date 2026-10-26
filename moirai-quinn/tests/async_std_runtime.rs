#![cfg(all(feature = "runtime-async-std", feature = "rustls-ring"))]

use std::{net::UdpSocket, sync::Arc, time::Duration};

use moirai_quinn::{AsyncStdRuntime, ClientConfig, Endpoint, EndpointConfig, ServerConfig};
use rustls::{pki_types::PrivateKeyDer, RootCertStore};

#[test]
fn async_std_compatible_executor_exchanges_authenticated_quic_bytes() {
    let exchange = async {
        let certified = rcgen::generate_simple_self_signed(vec!["localhost".into()]).unwrap();
        let certificate = certified.cert.der().clone();
        let key = PrivateKeyDer::Pkcs8(certified.signing_key.serialize_der().into());
        let server_config = ServerConfig::with_single_cert(vec![certificate.clone()], key).unwrap();
        let runtime = Arc::new(AsyncStdRuntime);
        let server = Endpoint::new(
            EndpointConfig::default(),
            Some(server_config),
            UdpSocket::bind("127.0.0.1:0").unwrap(),
            runtime.clone(),
        )
        .unwrap();
        let server_address = server.local_addr().unwrap();
        let mut roots = RootCertStore::empty();
        roots.add(certificate).unwrap();
        let mut client = Endpoint::new(
            EndpointConfig::default(),
            None,
            UdpSocket::bind("127.0.0.1:0").unwrap(),
            runtime,
        )
        .unwrap();
        client.set_default_client_config(
            ClientConfig::with_root_certificates(Arc::new(roots)).unwrap(),
        );

        let server_task = async_global_executor::spawn(async move {
            let connection = server.accept().await.unwrap().await.unwrap();
            let mut stream = connection.accept_uni().await.unwrap();
            let received = stream.read_to_end(1024).await.unwrap();
            connection.close(0u32.into(), b"received");
            received
        });
        let connection = client
            .connect(server_address, "localhost")
            .unwrap()
            .await
            .unwrap();
        let mut stream = connection.open_uni().await.unwrap();
        stream
            .write_all(b"moirai async-std compatibility")
            .await
            .unwrap();
        stream.finish().unwrap();
        assert_eq!(server_task.await, b"moirai async-std compatibility");
        client.close(0u32.into(), b"complete");
    };
    async_global_executor::block_on(futures_lite::future::or(exchange, async {
        async_io::Timer::after(Duration::from_secs(10)).await;
        panic!("async-std compatibility QUIC exchange exceeded 10 seconds");
    }));
}

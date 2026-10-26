//! Native buffer and I/O behavior through real Quinn protocol drivers and UDP.

use super::*;
use moirai_async::io::ReadBuffer;
use std::mem::MaybeUninit;
use std::pin::Pin;

#[derive(Clone, Copy, Debug)]
enum ReadMode {
    Initialized,
    Uninitialized,
    NativeTrait,
}

#[test]
fn native_uninitialized_partial_reads_preserve_bytes_and_fin() {
    let _trace = trace();
    for (length, mode) in [0, 1, 7, 257].into_iter().flat_map(|length| {
        [
            ReadMode::Initialized,
            ReadMode::Uninitialized,
            ReadMode::NativeTrait,
        ]
        .map(|mode| (length, mode))
    }) {
        let _case = tracing::info_span!("native_read_buffer", length, ?mode).entered();
        let _arrange = tracing::info_span!("arrange").entered();
        let executor = executor();
        let config = config(1024, 1, DEADLINE);
        let (server, client) = endpoints(&executor, &config);
        let (server_connection, client_connection) = pair(&server, &client);
        let expected = payload(length);
        drop(_arrange);
        let _act = tracing::info_span!("act").entered();
        let actual = moirai_executor::block_on(moirai_async::timer::timeout(DEADLINE, async {
            let (mut sender, _client_receiver) = client_connection
                .connection
                .open_bi()
                .await
                .expect("admitted stream");
            let send = async {
                moirai_async::io::AsyncWriteExt::write_all(&mut sender, &expected)
                    .await
                    .expect("complete native trait write");
                moirai_async::io::AsyncWriteExt::flush(&mut sender)
                    .await
                    .expect("native flush");
                moirai_async::io::AsyncWriteExt::shutdown(&mut sender)
                    .await
                    .expect("queue native FIN");
                assert_eq!(sender.stopped().await.expect("FIN acknowledged"), None);
            };
            let receive = async {
                let (_server_sender, mut receiver) = server_connection
                    .connection
                    .accept_bi()
                    .await
                    .expect("peer stream");
                let mut uninitialized = [MaybeUninit::uninit(); 7];
                let mut initialized = [0xa5; 7];
                let capacity = initialized.len();
                let mut actual = Vec::new();
                loop {
                    let chunk = if matches!(mode, ReadMode::NativeTrait) {
                        let count = std::future::poll_fn(|cx| {
                            moirai_async::io::AsyncRead::poll_read(
                                Pin::new(&mut receiver),
                                cx,
                                &mut initialized,
                            )
                        })
                        .await
                        .expect("native trait read");
                        initialized[..count].to_vec()
                    } else {
                        let mut buffer = if matches!(mode, ReadMode::Uninitialized) {
                            ReadBuffer::uninit(&mut uninitialized)
                        } else {
                            ReadBuffer::new(&mut initialized)
                        };
                        std::future::poll_fn(|cx| receiver.poll_read_buf(cx, &mut buffer))
                            .await
                            .expect("native buffer read");
                        buffer.filled().to_vec()
                    };
                    if chunk.is_empty() {
                        break;
                    }
                    assert!(chunk.len() <= capacity);
                    actual.extend_from_slice(&chunk);
                }
                actual
            };
            let ((), actual) = futures::future::join(send, receive).await;
            actual
        }))
        .expect("unchanged transport deadline");
        drop(_act);
        let _assert = tracing::info_span!("assert").entered();
        assert_eq!(actual, expected);
        server_connection.close();
        client_connection.close();
    }
}

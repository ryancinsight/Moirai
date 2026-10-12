use super::*;
use crate::instance::{MAX_INSTANCE_MESSAGE_BYTES, read_frame, write_frame};
use std::io::Write;

fn capture() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(
        tracing_subscriber::fmt()
            .with_test_writer()
            .with_span_events(tracing_subscriber::fmt::format::FmtSpan::FULL)
            .finish(),
    )
}

#[test]
fn closed_senders_preserve_complete_buffered_messages() {
    let _trace = capture();
    let _test = tracing::info_span!("closed_instance_sender").entered();
    for bytes in [
        b"org.example://study/7".to_vec(),
        (u8::MIN..=u8::MAX)
            .cycle()
            .take(MAX_INSTANCE_MESSAGE_BYTES)
            .collect(),
        Vec::new(),
    ] {
        let (mut sender, mut receiver) = tracing::info_span!("arrange").in_scope(|| {
            let pair = UnixStream::pair().expect("real local sockets");
            // A complete maximum frame must fit before the receiver starts:
            // macOS socket pairs otherwise default to an 8 KiB send buffer.
            let capacity =
                libc::c_int::try_from(MAX_INSTANCE_MESSAGE_BYTES + std::mem::size_of::<u32>())
                    .expect("frame capacity fits the socket ABI");
            let length = libc::socklen_t::try_from(std::mem::size_of_val(&capacity))
                .expect("socket option length");
            // SAFETY: the socket stays owned; capacity is a live c_int and
            // length names its exact initialized representation.
            let result = unsafe {
                libc::setsockopt(
                    pair.0.as_raw_fd(),
                    libc::SOL_SOCKET,
                    libc::SO_SNDBUF,
                    std::ptr::from_ref(&capacity).cast(),
                    length,
                )
            };
            assert_eq!(
                result,
                0,
                "reserve complete frame: {}",
                io::Error::last_os_error()
            );
            pair.1.set_nonblocking(true).expect("nonblocking receiver");
            pair
        });
        write_frame(&mut sender, &bytes).expect("complete frame");
        sender
            .shutdown(std::net::Shutdown::Write)
            .expect("shutdown sender");
        drop(sender);
        let received = tracing::info_span!("act")
            .in_scope(|| read_frame(&mut Receiver::new(&mut receiver)).expect("buffered frame"));
        tracing::info_span!("assert").in_scope(|| assert_eq!(received, bytes));
    }
}

#[test]
fn partial_closed_frames_and_live_stalls_keep_distinct_errors() {
    let _trace = capture();
    let _test = tracing::info_span!("instance_receive_failure_contract").entered();
    let (mut sender, mut receiver) = tracing::info_span!("arrange").in_scope(|| {
        let pair = UnixStream::pair().expect("real local sockets");
        pair.1.set_nonblocking(true).expect("nonblocking receiver");
        pair
    });
    sender.write_all(&[3, 0, 0, 0, 1]).expect("truncated frame");
    drop(sender);
    let error = tracing::info_span!("act").in_scope(|| {
        read_frame(&mut Receiver::new(&mut receiver)).expect_err("partial closed frame")
    });
    tracing::info_span!("assert").in_scope(|| assert_eq!(error.kind(), ErrorKind::UnexpectedEof));

    let (_sender, mut receiver) = tracing::info_span!("arrange").in_scope(|| {
        let pair = UnixStream::pair().expect("real local sockets");
        pair.1.set_nonblocking(true).expect("nonblocking receiver");
        pair
    });
    let started = Instant::now();
    let error = tracing::info_span!("act")
        .in_scope(|| read_frame(&mut Receiver::new(&mut receiver)).expect_err("live sender stall"));
    tracing::info_span!("assert").in_scope(|| {
        assert_eq!(error.kind(), ErrorKind::TimedOut);
        assert!(started.elapsed() >= INSTANCE_IO_TIMEOUT);
    });
}

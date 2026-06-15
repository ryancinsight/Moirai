#[test]
fn result_handle_diagnostics_track_async_state_primitives() {
    let source = read_benchmark("benches/result_handle_diagnostics/async_state.rs");
    let benchmark = read_result_handle_diagnostics();

    for required in [
        "struct DiagnosticAsyncWaker",
        "impl Wake for DiagnosticAsyncWaker",
        "fn wake_by_ref(self: &Arc<Self>)",
        "compare_exchange(",
        "DIAGNOSTIC_ASYNC_IDLE",
        "DIAGNOSTIC_ASYNC_QUEUED",
        "DIAGNOSTIC_ASYNC_POLLING",
        "DIAGNOSTIC_ASYNC_NOTIFIED",
        "direct_async_idle_to_queued_state_claim",
        "direct_async_polling_to_notified_state_claim",
        "direct_async_notified_to_polling_state_claim",
        "direct_async_polling_to_idle_state_release",
        "direct_async_waker_from_arc",
        "direct_async_wake_by_ref_polling_notification",
        "direct_async_completed_state_store",
        "direct_async_future_present_drop_flag",
        "direct_async_lifecycle_complete",
        "direct_async_sender_cell_take_send_join",
        "direct_async_ready_completion_components",
        "UnsafeCell::new(true)",
        "UnsafeCell::new(Some(sender))",
        "metrics.record_task_completed",
    ] {
        assert!(
            source.contains(required) && benchmark.contains(required),
            "async state diagnostics must retain {required}"
        );
    }

    for prohibited in ["Box<dyn", "Mutex<", "dyn Future<Output", "AtomicBool"] {
        assert!(
            !source.contains(prohibited),
            "async state diagnostics must not introduce {prohibited}"
        );
    }
}

#[test]
fn timer_wheel_cancellation_is_real_and_lazy() {
    let source = format!(
        "{}\n{}",
        read_benchmark("../moirai-async/src/timer.rs"),
        read_benchmark("../moirai-async/src/timer/wheel.rs")
    );

    for required in [
        "mod wheel;",
        "pub use wheel::{TimerCommand, TimerWheel};",
        "cancelled: HashSet<u64>",
        "self.cancelled.insert(timer_id)",
        "self.cancelled.remove(&expired.id)",
        "fn timer_wheel_cancelled_timer_does_not_wake()",
        "fn timer_wheel_poll_wakes_only_uncancelled_expired_timers()",
    ] {
        assert!(
            source.contains(required),
            "timer wheel cancellation must retain {required}"
        );
    }

    for prohibited in [
        "cancel(&mut self, _timer_id",
        "false // Simplified",
        "This is a simplified implementation",
        "For now, we'll mark",
    ] {
        assert!(
            !source.contains(prohibited),
            "timer wheel cancellation must not reintroduce placeholder path {prohibited}"
        );
    }
}

#[test]
fn pal_timer_future_waits_until_deadline() {
    let source = format!(
        "{}\n{}",
        read_benchmark("../moirai-pal/src/timer.rs"),
        read_benchmark("../moirai-pal/src/timer/tests.rs")
    );
    let audit = read_benchmark("../docs/rayon_tokio_gap_audit.md");

    for required in [
        "struct TimerState",
        "completed: AtomicBool",
        "sleeper_started: AtomicBool",
        "waker: Mutex<Option<std::task::Waker>>",
        "fn spawn_sleeper",
        "std::thread::sleep(deadline.duration_since(now))",
        "Poll::Pending",
        "pal_timer_is_pending_before_deadline_and_wakes",
        "pal_timer_zero_duration_completes_immediately",
    ] {
        assert!(
            source.contains(required),
            "PAL timer future must retain real deadline marker {required}"
        );
    }

    assert!(
        audit.contains("PAL platform timer future"),
        "Rayon/Tokio audit must retain PAL timer boundary"
    );

    for prohibited in [
        "Placeholder for platform-agnostic timer operations",
        "This will be fully implemented",
        "In a real implementation",
        "For now, yield once",
    ] {
        assert!(
            !source.contains(prohibited),
            "PAL timer future must not reintroduce placeholder marker {prohibited}"
        );
    }
}

#[test]
fn async_file_facade_is_value_semantic_and_benchmarked_against_tokio() {
    let fs_source = format!(
        "{}\n{}",
        read_benchmark("../moirai-async/src/fs.rs"),
        read_benchmark("../moirai-async/src/fs/tests.rs")
    );
    let pal_fs_source = read_benchmark("../moirai-pal/src/fs.rs");
    let fs_benchmark = read_benchmark("benches/async_fs_comparison.rs");
    let fs_dir_benchmark = read_benchmark("benches/async_fs_dir_comparison.rs");
    let benchmark_manifest = read_benchmark("Cargo.toml");
    let audit = read_benchmark("../docs/rayon_tokio_gap_audit.md");

    for required in [
        "pub async fn read_to_string<P: AsRef<Path>>(path: P) -> io::Result<String>",
        "pub async fn read<P: AsRef<Path>>(path: P) -> io::Result<Vec<u8>>",
        "pub async fn write<P: AsRef<Path>, C: AsRef<[u8]>>(path: P, contents: C) -> io::Result<()>",
        "pub async fn append<P: AsRef<Path>, C: AsRef<[u8]>>(path: P, contents: C) -> io::Result<()>",
        "pub async fn copy<P: AsRef<Path>, Q: AsRef<Path>>(from: P, to: Q) -> io::Result<u64>",
        "pub async fn metadata<P: AsRef<Path>>(path: P) -> io::Result<std::fs::Metadata>",
        "pub async fn rename<P: AsRef<Path>, Q: AsRef<Path>>(from: P, to: Q) -> io::Result<()>",
        "pub async fn remove_file<P: AsRef<Path>>(path: P) -> io::Result<()>",
        "pub async fn create_dir<P: AsRef<Path>>(path: P) -> io::Result<()>",
        "pub async fn create_dir_all<P: AsRef<Path>>(path: P) -> io::Result<()>",
        "pub async fn remove_dir<P: AsRef<Path>>(path: P) -> io::Result<()>",
        "pub async fn remove_dir_all<P: AsRef<Path>>(path: P) -> io::Result<()>",
        "moirai_pal::fs::write(path, contents).await",
        "moirai_pal::fs::append(path, contents).await",
        "moirai_pal::fs::copy(from, to).await",
        "moirai_pal::fs::metadata(path).await",
        "moirai_pal::fs::rename(from, to).await",
        "moirai_pal::fs::remove_file(path).await",
        "moirai_pal::fs::create_dir(path).await",
        "moirai_pal::fs::create_dir_all(path).await",
        "moirai_pal::fs::remove_dir(path).await",
        "moirai_pal::fs::remove_dir_all(path).await",
        "test_file_write_read_append_and_stats_values",
        "test_file_copy_and_directory_values",
        "test_recursive_directory_values",
        "assert_eq!(contents, \"alpha-beta\")",
        "assert_eq!(file.stats().bytes_read, 5)",
        "assert_eq!(copied, 10)",
        "assert_eq!(dest_bytes, b\"0123456789\")",
    ] {
        assert!(
            fs_source.contains(required),
            "async file facade source must retain marker {required}"
        );
    }

    for required in [
        "pub async fn write<P: AsRef<Path>, C: AsRef<[u8]>>(path: P, contents: C) -> io::Result<()>",
        "std::fs::write(path, contents)",
        "pub async fn append<P: AsRef<Path>, C: AsRef<[u8]>>(path: P, contents: C) -> io::Result<()>",
        "StdOpenOptions::new().create(true).append(true).open(path)?",
        "pub async fn copy<P: AsRef<Path>, Q: AsRef<Path>>(from: P, to: Q) -> io::Result<u64>",
        "std::fs::copy(from, to)",
        "pub async fn metadata<P: AsRef<Path>>(path: P) -> io::Result<std::fs::Metadata>",
        "std::fs::metadata(path)",
        "pub async fn rename<P: AsRef<Path>, Q: AsRef<Path>>(from: P, to: Q) -> io::Result<()>",
        "std::fs::rename(from, to)",
        "pub async fn remove_file<P: AsRef<Path>>(path: P) -> io::Result<()>",
        "std::fs::remove_file(path)",
        "pub async fn create_dir<P: AsRef<Path>>(path: P) -> io::Result<()>",
        "std::fs::create_dir(path)",
        "pub async fn create_dir_all<P: AsRef<Path>>(path: P) -> io::Result<()>",
        "std::fs::create_dir_all(path)",
        "pub async fn remove_dir<P: AsRef<Path>>(path: P) -> io::Result<()>",
        "std::fs::remove_dir(path)",
        "pub async fn remove_dir_all<P: AsRef<Path>>(path: P) -> io::Result<()>",
        "std::fs::remove_dir_all(path)",
        "async_file_write_preserves_source_bytes",
        "async_file_append_preserves_prefix_and_appended_bytes",
        "async_file_copy_preserves_source_bytes",
        "async_file_metadata_preserves_file_type_and_length",
        "async_file_rename_preserves_source_bytes_at_destination",
        "async_file_remove_file_deletes_expected_path",
        "async_dir_create_and_remove_preserves_directory_state",
        "async_dir_all_create_and_remove_deletes_nested_tree",
    ] {
        assert!(
            pal_fs_source.contains(required),
            "PAL async file facade source must retain marker {required}"
        );
    }

    for prohibited in [
        "TODO: Add proper async tests once Moirai's async runtime is integrated",
        "Note: These tests are simplified for the tokio removal",
        "AsyncFileOp",
        "For now, return immediately",
        "In a full implementation",
        "vec![0u8; 64 * 1024]",
        "dest.write_all(&buffer",
        "file.sync_all().await",
    ] {
        assert!(
            !fs_source.contains(prohibited),
            "async file facade must not reintroduce missing-test marker {prohibited}"
        );
    }

    for required in [
        "name = \"async_fs_comparison\"",
        "moirai_read",
        "runtime: &moirai::Moirai",
        ".block_on(moirai_async::fs::read(path))",
        "moirai::Moirai::new().expect(\"moirai benchmark runtime must build\")",
        "tokio_read",
        "tokio::fs::read",
        "assert_eq!(moirai_expected, expected)",
        "assert_eq!(tokio_expected, expected)",
        "async_fs_read_to_end",
        "moirai_write",
        "tokio_write",
        "tokio::fs::write",
        "async_fs_write_file",
        "moirai_append",
        "tokio_append",
        "tokio::fs::OpenOptions::new()",
        "AsyncWriteExt",
        "async_fs_append_file",
        "assert_appended_bytes",
        "moirai_metadata_len",
        "tokio_metadata_len",
        "tokio::fs::metadata",
        "async_fs_metadata_file",
        "moirai_rename",
        "tokio_rename",
        "tokio::fs::rename",
        "async_fs_rename_file",
        "moirai_remove_file",
        "tokio_remove_file",
        "tokio::fs::remove_file",
        "async_fs_remove_file",
        "name = \"async_fs_dir_comparison\"",
        "moirai_create_remove_dir",
        "tokio_create_remove_dir",
        "tokio::fs::create_dir",
        "tokio::fs::remove_dir",
        "async_fs_create_remove_dir",
        "moirai_create_remove_dir_all",
        "tokio_create_remove_dir_all",
        "tokio::fs::create_dir_all",
        "tokio::fs::remove_dir_all",
        "async_fs_create_remove_dir_all",
        "moirai_copy",
        "tokio_copy",
        "tokio::fs::copy",
        "assert_copied_bytes",
        "assert_eq!(copied, READ_BYTES as u64)",
        "async_fs_copy_file",
    ] {
        assert!(
            fs_benchmark.contains(required)
                || fs_dir_benchmark.contains(required)
                || benchmark_manifest.contains(required),
            "async fs benchmark must retain comparison marker {required}"
        );
    }

    assert!(
        !fs_benchmark.contains("futures::executor::block_on"),
        "async fs benchmark must use the Moirai runtime surface for Moirai rows"
    );
    assert!(
        !fs_dir_benchmark.contains("futures::executor::block_on"),
        "async fs directory benchmark must use the Moirai runtime surface for Moirai rows"
    );

    for required in [
        "Tokio file facade read",
        "Tokio file facade write",
        "Tokio file facade append",
        "Tokio file facade metadata",
        "Tokio file facade rename",
        "Tokio file facade remove",
        "Tokio file facade copy",
        "Tokio directory facade create/remove",
        "Tokio directory facade recursive create/remove",
        "async_fs_comparison",
        "async_fs_dir_comparison",
        "64 KiB read",
        "64 KiB write",
        "64 KiB append",
        "64 KiB metadata",
        "64 KiB rename",
        "64 KiB remove",
        "64 KiB copy",
        "single directory create/remove",
        "recursive directory create/remove",
        "Tokio I/O drop-in compatibility",
    ] {
        assert!(
            audit.contains(required),
            "Rayon/Tokio audit must retain async fs marker {required}"
        );
    }
}

#[test]
fn async_network_facade_has_loopback_value_tests_and_audited_boundary() {
    let net_source = read_benchmark("../moirai-async/src/net.rs");
    let net_tests = read_benchmark("../moirai-async/src/net/tests.rs");
    let udp_benchmark = read_benchmark("benches/async_udp_comparison.rs");
    let tcp_benchmark = read_benchmark("benches/async_tcp_comparison.rs");
    let tcp_backpressure_benchmark = read_benchmark("benches/async_tcp_backpressure_comparison.rs");
    let tcp_readiness_benchmark = read_benchmark("benches/async_tcp_readiness_comparison.rs");
    let tcp_cancel_benchmark = read_benchmark("benches/async_tcp_cancel_safety_comparison.rs");
    let benchmark_manifest = read_benchmark("Cargo.toml");
    let audit = read_benchmark("../docs/rayon_tokio_gap_audit.md");
    let comparison_report = read_benchmark("../docs/moirai_rayon_tokio_comparison.md");

    for required in [
        "pub async fn accept(&self) -> io::Result<(TcpStream, SocketAddr)>",
        "pub async fn connect(addr: &str) -> io::Result<Self>",
        "pub fn from_std(stream: std::net::TcpStream) -> io::Result<Self>",
        "pub async fn read(&mut self, buf: &mut [u8]) -> io::Result<usize>",
        "pub async fn write(&mut self, buf: &[u8]) -> io::Result<usize>",
        "pub async fn shutdown(&mut self) -> io::Result<()>",
        "self.inner.shutdown_write()",
        "pub fn set_nodelay(&self, on: bool) -> io::Result<()>",
        "pub async fn send_to(&self, buf: &[u8], target: SocketAddr) -> io::Result<usize>",
        "pub async fn recv_from(&self, buf: &mut [u8]) -> io::Result<(usize, SocketAddr)>",
        "pub fn local_addr(&self) -> io::Result<SocketAddr>",
        "test_tcp_loopback_read_write_and_stats_values",
        "test_tcp_shutdown_write_sends_eof_and_stats_values",
        "test_tcp_poll_write_reports_pending_under_backpressure",
        "test_tcp_poll_read_reports_pending_before_peer_data",
        "test_tcp_pending_read_future_drop_preserves_stream_payload",
        "test_udp_loopback_send_recv_and_stats_values",
        "assert_eq!(&inbound, b\"ping\")",
        "assert_eq!(&echo, b\"pong\")",
        "assert_eq!(&received, b\"closed\")",
        "assert!(written <= BACKPRESSURE_MAX_BYTES)",
        "assert!(matches!(pending, Poll::Pending))",
        "assert_eq!(buf, READINESS_PAYLOAD)",
        "assert_eq!(cancelled_buf, [0_u8; READINESS_PAYLOAD.len()])",
        "assert_eq!(&buf[..received], b\"datagram\")",
        "assert_eq!(stats.bytes_received, 4)",
        "assert_eq!(recv_stats.packets_received, 1)",
    ] {
        assert!(
            net_source.contains(required) || net_tests.contains(required),
            "async network facade source must retain marker {required}"
        );
    }

    for prohibited in [
        "TODO: Implement proper async I/O reactor",
        "TODO: Add proper async network tests once Moirai's async runtime is integrated",
        "This is a simplified implementation",
        "For now, use blocking accept",
    ] {
        assert!(
            !net_source.contains(prohibited),
            "async network facade must not reintroduce missing-test marker {prohibited}"
        );
    }

    for required in [
        "Moirai-owned network facade value semantics",
        "TCP loopback read/write",
        "UDP loopback send/receive",
        "Tokio reactor-native I/O drop-in compatibility",
    ] {
        assert!(
            audit.contains(required) && comparison_report.contains(required),
            "I/O audit artifacts must retain network boundary marker {required}"
        );
    }

    for required in [
        "name = \"async_tcp_comparison\"",
        "moirai_tcp_echo_roundtrip",
        "runtime: &moirai::Moirai",
        ".block_on(async {",
        "moirai_async::net::TcpListener::bind",
        "tokio_tcp_echo_roundtrip",
        "tokio::net::TcpListener::bind",
        "assert_eq!(&moirai_expected, CLIENT_PAYLOAD)",
        "assert_eq!(&tokio_expected, CLIENT_PAYLOAD)",
        "async_tcp_loopback_echo",
        "spawn_echo_server",
        "moirai_tcp_stream_echo_once",
        "tokio_tcp_stream_echo_once",
        "moirai_tcp_shutdown_once",
        "tokio_tcp_shutdown_once",
        "async_tcp_stream_echo",
        "async_tcp_write_shutdown",
        "MoiraiAsyncReadExt::read_exact",
        "MoiraiAsyncWriteExt::write_all",
        "MoiraiAsyncWriteExt::shutdown",
        "tokio::io::AsyncWriteExt::shutdown",
        "set_nodelay(true)",
        "assert_eq!(&moirai_stream_expected, SERVER_PAYLOAD)",
        "assert_eq!(&tokio_stream_expected, SERVER_PAYLOAD)",
    ] {
        assert!(
            tcp_benchmark.contains(required) || benchmark_manifest.contains(required),
            "async TCP benchmark must retain comparison marker {required}"
        );
    }

    for required in [
        "name = \"async_tcp_backpressure_comparison\"",
        "async_tcp_write_backpressure",
        "poll_moirai_until_backpressured",
        "poll_tokio_until_backpressured",
        "moirai_async::net::TcpStream::from_std",
        "tokio::net::TcpStream::from_std",
        "set_send_buffer_size",
        "set_recv_buffer_size",
        "Poll::Pending",
        "assert!(written > 0)",
        "MAX_WRITTEN_BYTES",
    ] {
        assert!(
            tcp_backpressure_benchmark.contains(required) || benchmark_manifest.contains(required),
            "async TCP backpressure benchmark must retain marker {required}"
        );
    }

    for required in [
        "name = \"async_tcp_readiness_comparison\"",
        "async_tcp_read_readiness",
        "poll_moirai_before_peer_data",
        "poll_tokio_before_peer_data",
        "moirai_async::net::TcpStream::from_std",
        "tokio::net::TcpStream::from_std",
        "tokio::io::ReadBuf",
        "Poll::Pending",
        "READINESS_PAYLOAD",
        "assert_eq!(buf, READINESS_PAYLOAD)",
    ] {
        assert!(
            tcp_readiness_benchmark.contains(required) || benchmark_manifest.contains(required),
            "async TCP read-readiness benchmark must retain marker {required}"
        );
    }

    for required in [
        "name = \"async_tcp_cancel_safety_comparison\"",
        "async_tcp_pending_read_cancel_safety",
        "cancel_moirai_pending_read",
        "cancel_tokio_pending_read",
        "MoiraiAsyncReadExt::read_exact",
        "tokio::io::AsyncReadExt::read_exact",
        "std::pin::pin!",
        "Poll::Pending",
        "assert_eq!(cancelled_buf, [0_u8; CANCEL_PAYLOAD_LEN])",
        "assert_eq!(buf, CANCEL_PAYLOAD)",
    ] {
        assert!(
            tcp_cancel_benchmark.contains(required) || benchmark_manifest.contains(required),
            "async TCP cancel-safety benchmark must retain marker {required}"
        );
    }

    assert!(
        !tcp_benchmark.contains("futures::executor::block_on"),
        "async TCP benchmark must use the Moirai runtime surface for Moirai rows"
    );

    for required in [
        "Tokio TCP loopback accept/echo",
        "Tokio TCP persistent stream echo",
        "async_tcp_comparison",
        "async_tcp_loopback_echo",
        "async_tcp_stream_echo",
        "async_tcp_write_shutdown",
        "async_tcp_backpressure_comparison",
        "async_tcp_write_backpressure",
        "async_tcp_readiness_comparison",
        "async_tcp_read_readiness",
        "async_tcp_cancel_safety_comparison",
        "async_tcp_pending_read_cancel_safety",
    ] {
        assert!(
            audit.contains(required) && comparison_report.contains(required),
            "I/O audit artifacts must retain TCP benchmark marker {required}"
        );
    }

    for required in [
        "name = \"async_udp_comparison\"",
        "recv_moirai_payload",
        "runtime: &moirai::Moirai",
        "runtime.block_on(receiver.recv_from(buf))",
        "moirai::Moirai::new().expect(\"moirai benchmark runtime must build\")",
        "recv_tokio_payload",
        "tokio::net::UdpSocket::bind",
        "assert_eq!(moirai_expected, PAYLOAD)",
        "assert_eq!(tokio_expected, PAYLOAD)",
        "async_udp_loopback_recv_from",
    ] {
        assert!(
            udp_benchmark.contains(required) || benchmark_manifest.contains(required),
            "async UDP benchmark must retain comparison marker {required}"
        );
    }

    assert!(
        !udp_benchmark.contains("futures::executor::block_on"),
        "async UDP benchmark must use the Moirai runtime surface for Moirai rows"
    );

    for required in [
        "Tokio UDP loopback receive",
        "async_udp_comparison",
        "async_udp_loopback_recv_from",
    ] {
        assert!(
            audit.contains(required) && comparison_report.contains(required),
            "I/O audit artifacts must retain UDP benchmark marker {required}"
        );
    }
}

#[test]
fn pal_async_io_facades_have_value_tests_and_self_wake_contract() {
    let pal_lib = read_benchmark("../moirai-pal/src/lib.rs");
    let pal_fs = read_benchmark("../moirai-pal/src/fs.rs");
    let pal_net = read_benchmark("../moirai-pal/src/net.rs");
    let pal_reactor = [
        "../moirai-pal/src/reactor/core.rs",
        "../moirai-pal/src/reactor/future.rs",
        "../moirai-pal/src/reactor/task.rs",
        "../moirai-pal/src/reactor/tests.rs",
        "../moirai-pal/src/reactor/tls.rs",
    ]
    .into_iter()
    .map(read_benchmark)
    .collect::<String>();
    let epoll_reactor = read_benchmark("../moirai-pal/src/unix/epoll.rs");
    let audit = read_benchmark("../docs/rayon_tokio_gap_audit.md");

    for required in [
        "pub type PlatformReactor",
        "pub fn create_reactor() -> io::Result<PlatformReactor>",
    ] {
        assert!(
            pal_lib.contains(required),
            "PAL lib source must retain static platform reactor marker {required}"
        );
    }

    assert!(
        !pal_lib.contains("io::Result<Box<dyn Reactor>>"),
        "PAL reactor factory must not return a boxed dynamic reactor"
    );

    for required in [
        "pub struct YieldFuture",
        "cx.waker().wake_by_ref();",
        "pub struct AsyncFile",
        "pub async fn open_with_options<P: AsRef<Path>>",
        "async_file_roundtrip_seek_and_metadata_are_value_semantic",
        "async_file_read_to_end_preserves_source_bytes",
        "async_file_write_preserves_source_bytes",
        "async_file_append_preserves_prefix_and_appended_bytes",
        "async_file_metadata_preserves_file_type_and_length",
        "async_file_rename_preserves_source_bytes_at_destination",
        "async_file_remove_file_deletes_expected_path",
        "async_dir_create_and_remove_preserves_directory_state",
        "async_dir_all_create_and_remove_deletes_nested_tree",
        "assert_eq!(&suffix, b\"beta\")",
        "assert_eq!(actual, expected)",
    ] {
        assert!(
            pal_fs.contains(required),
            "PAL async file source must retain marker {required}"
        );
    }

    for required in [
        "fn wake_without_active_reactor(cx: &Context<'_>)",
        "cx.waker().wake_by_ref();",
        "pub fn shutdown_write(&self) -> io::Result<()>",
        "self.inner.shutdown(Shutdown::Write)",
        "IoReactor::get_active()",
        "reactor.register_waker(raw, Interest::READABLE, cx.waker().clone())",
        "reactor.register_waker(raw, Interest::WRITABLE, cx.waker().clone())",
        "wake_without_active_reactor(cx);",
        "tcp_accept_read_write_self_wakes_without_active_reactor",
        "udp_recv_self_wakes_without_active_reactor",
        "pub async fn flush(&mut self) -> io::Result<()>",
        "assert_eq!(&inbound, b\"ping\")",
        "assert_eq!(&echo, b\"pong\")",
        "assert_eq!(&buf[..received], b\"datagram\")",
    ] {
        assert!(
            pal_net.contains(required),
            "PAL async network source must retain marker {required}"
        );
    }

    for prohibited in [
        "Native async file I/O not yet implemented",
        "Native async network I/O not yet implemented",
        "Placeholder implementation",
        "Will contain platform-specific",
    ] {
        assert!(
            !pal_fs.contains(prohibited) && !pal_net.contains(prohibited),
            "PAL async I/O must not reintroduce placeholder marker {prohibited}"
        );
    }

    for required in [
        "const INLINE_REACTOR_TASK_WORDS: usize = 14",
        "task_queue: Arc<Mutex<VecDeque<Arc<ReactorTaskState>>>>",
        "struct ReactorTaskState",
        "future: ErasedReactorTaskFuture",
        "struct ReactorTaskFutureStorage",
        "struct ErasedReactorTaskFuture",
        "storage: UnsafeCell<ReactorTaskFutureStorage>",
        "poll: unsafe fn(*mut ReactorTaskFutureStorage, &mut Context<'_>) -> Poll<()>",
        "drop: unsafe fn(*mut ReactorTaskFutureStorage)",
        "ErasedReactorTaskFuture::new(future)",
        "reactor_future_fits::<F>()",
        "Self::new_boxed(future)",
        "poll_inline_reactor_future::<F>",
        "poll_boxed_reactor_future::<F>",
        "drop_boxed_reactor_future::<F>",
        "struct TaskCompletion",
        "completed: AtomicBool",
        "fn complete(&self)",
        "fn poll(&self, cx: &Context<'_>) -> Poll<()>",
        "task.complete();",
        "spawned_ready_task_handle_completes_after_iteration",
        "reactor_future_storage_budget_is_static_and_bounded",
        "spawned_inline_and_oversized_reactor_futures_complete",
        "assert!(reactor_future_fits::<MaxInlineReadyFuture>())",
        "assert!(!reactor_future_fits::<OversizedShapeFuture>())",
        "assert_eq!(metrics.tasks_executed.load(Ordering::Relaxed), 1)",
    ] {
        assert!(
            pal_reactor.contains(required),
            "PAL reactor source must retain task-handle completion marker {required}"
        );
    }

    for prohibited in [
        "For now, always return pending",
        "this would check if the task completed",
        "Box<dyn Reactor>",
        "Pin<Box<dyn Future<Output = ()>",
        "future: Box::pin(future)",
        "Box::into_raw(Box::new(future))",
    ] {
        assert!(
            !pal_reactor.contains(prohibited),
            "PAL reactor must not reintroduce dynamic dispatch or placeholder marker {prohibited}"
        );
    }

    for required in [
        "wake_fd: RawFd",
        "libc::eventfd(0, libc::EFD_NONBLOCK | libc::EFD_CLOEXEC)",
        "drain_eventfd(self.wake_fd)?;",
        "fn drain_eventfd(wake_fd: RawFd) -> io::Result<()>",
        "test_epoll_wake_returns_no_user_events",
    ] {
        assert!(
            epoll_reactor.contains(required),
            "PAL epoll reactor must retain real wake marker {required}"
        );
    }

    for prohibited in [
        "This is a simplified implementation",
        "A complete implementation would use eventfd",
    ] {
        assert!(
            !epoll_reactor.contains(prohibited),
            "PAL epoll wake must not reintroduce placeholder marker {prohibited}"
        );
    }

    for required in [
        "PAL async file facade",
        "PAL async socket self-wake fallback",
        "PAL reactor task-handle completion",
        "PAL reactor task queue avoids dynamic future dispatch",
        "PAL platform reactor dispatch is static",
        "PAL epoll reactor wake",
        "tcp_accept_read_write_self_wakes_without_active_reactor",
        "udp_recv_self_wakes_without_active_reactor",
    ] {
        assert!(
            audit.contains(required),
            "Rayon/Tokio audit must retain PAL async I/O marker {required}"
        );
    }
}

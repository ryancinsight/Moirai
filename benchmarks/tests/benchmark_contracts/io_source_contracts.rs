#[test]
fn async_io_extension_futures_are_zero_copy_and_value_semantic() {
    let io_source = read_benchmark("../moirai-async/src/io.rs");
    let io_tests = read_benchmark("../moirai-async/src/io/tests.rs");
    let compat_tests = read_benchmark("../moirai-async/src/io/compat/tests.rs");
    let buffered_tests = read_benchmark("../moirai-async/src/io/compat/tests/buffered.rs");
    let vectored_tests = read_benchmark("../moirai-async/src/io/compat/tests/vectored.rs");
    let tcp_benchmark = read_benchmark("benches/async_tcp_comparison.rs");
    let compat_benchmark = read_benchmark("benches/async_io_compat_comparison.rs");
    let benchmark_manifest = read_benchmark("Cargo.toml");

    for required in [
        "pub trait AsyncReadExt: AsyncRead",
        "fn read_exact<'a>(&'a mut self, buf: &'a mut [u8]) -> ReadExact<'a, Self>",
        "pub struct ReadExact<'a, R: ?Sized>",
        "reader: &'a mut R",
        "buf: &'a mut [u8]",
        "filled: usize",
        "io::ErrorKind::UnexpectedEof",
        "pub trait AsyncWriteExt: AsyncWrite",
        "fn shutdown(&mut self) -> Shutdown<'_, Self>",
        "pub struct Shutdown<'a, W: ?Sized>",
        "writer: &'a mut W",
        "#[repr(transparent)]",
        "const TRANSPARENT: () = assert!(",
        "pub struct TokioCompat<T>",
        "pub struct MoiraiCompat<T>",
        "impl<T> From<T> for TokioCompat<T>",
        "impl<T> From<T> for MoiraiCompat<T>",
        "#[cfg(feature = \"tokio-compat\")]",
        "impl<T: AsyncRead + Unpin> tokio::io::AsyncRead for TokioCompat<T>",
        "impl<T: AsyncWrite + Unpin> tokio::io::AsyncWrite for TokioCompat<T>",
        "impl<T: tokio::io::AsyncRead + Unpin> AsyncRead for MoiraiCompat<T>",
        "impl<T: tokio::io::AsyncWrite + Unpin> AsyncWrite for MoiraiCompat<T>",
        "impl<T: AsyncBufRead + Unpin> tokio::io::AsyncBufRead for TokioCompat<T>",
        "impl<T: tokio::io::AsyncBufRead + Unpin> AsyncBufRead for MoiraiCompat<T>",
        "fn poll_write_vectored(",
        "fn is_write_vectored(&self) -> bool",
    ] {
        assert!(
            io_source.contains(required),
            "async I/O extension source must retain zero-copy marker {required}"
        );
    }

    for required in [
        "read_exact_fills_buffer_across_partial_reads",
        "read_exact_reports_unexpected_eof_with_prefix_preserved",
        "read_exact_cancellation_preserves_borrowed_buffer_progress",
        "write_all_flush_and_shutdown_use_borrowed_writer_without_boxing",
        "assert_eq!(&output, b\"abcdef\")",
        "assert_eq!(error.kind(), io::ErrorKind::UnexpectedEof)",
        "assert_eq!(&output[..2], b\"ab\")",
        "assert_eq!(writer.shutdowns, 1)",
        "tokio_compat_preserves_native_reader_writer_values",
        "moirai_compat_preserves_tokio_duplex_values",
        "TokioCompat::from(reader)",
        "MoiraiCompat::from(moirai_side)",
        "tokio_dep::io::AsyncReadExt::read_exact",
        "tokio_dep::io::AsyncWriteExt::shutdown",
        "assert_eq!(&reply, b\"reply\")",
        "assert_eq!(count, 0)",
    ] {
        assert!(
            io_tests.contains(required),
            "async I/O extension tests must retain value marker {required}"
        );
    }

    for required in [
        "chunked_transfer_is_byte_identical_on_native_and_wrapped_paths",
        "tokio_read_registers_the_polling_context_and_repoll_replaces_it",
        "moirai_read_registers_the_polling_context_and_repoll_replaces_it",
        "tokio_write_backpressure_wakes_the_latest_polling_context",
        "moirai_write_backpressure_wakes_the_latest_polling_context",
        "tokio_shutdown_reaches_the_reader_as_eof_and_wakes_it",
        "moirai_observes_tokio_peer_eof_and_broken_pipe",
        "zero_length_operations_transfer_nothing_and_keep_pending_data",
    ] {
        assert!(
            compat_tests.contains(required),
            "async I/O compatibility tests must retain scenario {required}"
        );
    }

    for required in [
        "buffered_windows_match_across_native_and_wrapped_paths",
        "tokio_line_reads_over_a_moirai_buffered_reader_keep_line_boundaries",
        "tokio_fill_buf_registers_the_polling_context_and_consume_advances",
        "moirai_fill_buf_registers_the_polling_context_and_consume_advances",
    ] {
        assert!(
            buffered_tests.contains(required),
            "async I/O buffered-read tests must retain scenario {required}"
        );
    }

    for required in [
        "tokio_vectored_write_reaches_a_vectored_moirai_writer",
        "writers_without_vectored_support_take_the_first_non_empty_slice",
        "moirai_vectored_write_matches_the_native_gather",
        "moirai_vectored_capability_follows_the_tokio_writer",
    ] {
        assert!(
            vectored_tests.contains(required),
            "async I/O vectored-write tests must retain scenario {required}"
        );
    }

    for prohibited in ["Pin<Box", "Box<dyn", "Vec<"] {
        assert!(
            !io_source.contains(prohibited),
            "async I/O extension futures must not allocate or type-erase with {prohibited}"
        );
    }

    for required in [
        "MoiraiAsyncReadExt::read_exact",
        "MoiraiAsyncWriteExt::write_all",
    ] {
        assert!(
            tcp_benchmark.contains(required),
            "async TCP benchmark must use production I/O extension future {required}"
        );
    }

    for required in [
        "name = \"async_io_compat_comparison\"",
        "async_io_compat_read_exact",
        "async_io_compat_write_shutdown",
        "moirai_native",
        "tokio_compat",
        "TokioCompat::from(reader)",
        "TokioCompat::from(writer)",
        "MoiraiAsyncReadExt::read_exact",
        "MoiraiAsyncWriteExt::write_all",
        "tokio::io::AsyncReadExt::read_exact",
        "tokio::io::AsyncWriteExt::write_all",
        "tokio::io::AsyncWriteExt::shutdown",
        "assert_eq!(output, PAYLOAD)",
        "assert_eq!(writer.shutdowns, 1)",
        "time_model::criterion()",
    ] {
        assert!(
            compat_benchmark.contains(required) || benchmark_manifest.contains(required),
            "async I/O compatibility benchmark must retain marker {required}"
        );
    }
}

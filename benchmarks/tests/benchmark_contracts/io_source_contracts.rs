#[test]
fn async_io_extension_futures_are_zero_copy_and_value_semantic() {
    let io_source = read_benchmark("../moirai-async/src/io.rs");
    let io_tests = read_benchmark("../moirai-async/src/io/tests.rs");
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
        "pub struct TokioCompat<T>",
        "pub struct MoiraiCompat<T>",
        "impl<T> From<T> for TokioCompat<T>",
        "impl<T> From<T> for MoiraiCompat<T>",
        "#[cfg(feature = \"tokio-compat\")]",
        "impl<T: AsyncRead + Unpin> tokio::io::AsyncRead for TokioCompat<T>",
        "impl<T: AsyncWrite + Unpin> tokio::io::AsyncWrite for TokioCompat<T>",
        "impl<T: tokio::io::AsyncRead + Unpin> AsyncRead for MoiraiCompat<T>",
        "impl<T: tokio::io::AsyncWrite + Unpin> AsyncWrite for MoiraiCompat<T>",
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
        "sample_size",
        "measurement_time",
        "warm_up_time",
    ] {
        assert!(
            compat_benchmark.contains(required) || benchmark_manifest.contains(required),
            "async I/O compatibility benchmark must retain marker {required}"
        );
    }
}

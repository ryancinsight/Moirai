//! Integration tests for Async I/O traits and Tokio compatibility shims.

use moirai::{
    AsyncReadExt, AsyncWriteExt, File, FileOpenOptions, MoiraiCompat, TcpListener, TokioCompat,
};
use std::io::SeekFrom;
use tempfile::NamedTempFile;

#[test]
fn test_native_file_io() {
    let temp_file = NamedTempFile::new().expect("failed to create temp file");
    let path = temp_file.path().to_path_buf();

    futures::executor::block_on(async {
        let mut file = File::open_with_options(&path, FileOpenOptions::read_write())
            .await
            .expect("failed to open file");

        let data = b"hello, native async I/O!";
        file.write_all(data).await.expect("failed to write data");
        file.flush().await.expect("failed to flush");

        file.seek(SeekFrom::Start(0)).await.expect("failed to seek");

        let mut buf = vec![0; data.len()];
        let mut read = 0;
        while read < buf.len() {
            let n = file.read(&mut buf[read..]).await.expect("failed to read");
            if n == 0 {
                break;
            }
            read += n;
        }

        assert_eq!(&buf[..read], data);
    });
}

#[test]
fn test_native_tcp_io() {
    futures::executor::block_on(async {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("failed to bind listener");
        let addr = listener.local_addr().expect("failed to get local addr");

        let client_task = async move {
            let mut client = moirai::TcpStream::connect(&addr.to_string())
                .await
                .expect("failed to connect");
            let data = b"client message";
            client.write_all(data).await.expect("failed to write");
            client.flush().await.expect("failed to flush");
        };

        let server_task = async move {
            let (mut server, _) = listener.accept().await.expect("failed to accept");
            let mut buf = vec![0; 64];
            let n = server.read(&mut buf).await.expect("failed to read");
            assert_eq!(&buf[..n], b"client message");
        };

        futures::join!(client_task, server_task);
    });
}

#[test]
fn test_tokio_compat_file_io() {
    let temp_file = NamedTempFile::new().expect("failed to create temp file");
    let path = temp_file.path().to_path_buf();

    // Run using standard Tokio runtime to ensure full library compatibility
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async {
        let file = File::open_with_options(&path, FileOpenOptions::read_write())
            .await
            .expect("failed to open file");

        let mut compat_write = TokioCompat::new(file);
        use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};

        let data = b"hello, tokio compatibility!";
        compat_write
            .write_all(data)
            .await
            .expect("failed to write via tokio compat");
        compat_write.flush().await.expect("failed to flush");

        let mut inner_file = compat_write.into_inner();
        inner_file
            .seek(SeekFrom::Start(0))
            .await
            .expect("failed to seek");

        let mut compat_read = TokioCompat::new(inner_file);
        let mut buf = Vec::new();
        compat_read
            .read_to_end(&mut buf)
            .await
            .expect("failed to read via tokio compat");

        assert_eq!(buf, data);
    });
}

#[test]
fn test_tokio_compat_tcp_io() {
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap();

    rt.block_on(async {
        let listener = TcpListener::bind("127.0.0.1:0")
            .await
            .expect("failed to bind");
        let addr = listener.local_addr().expect("failed to get local addr");

        let client_task = async move {
            let client = moirai::TcpStream::connect(&addr.to_string())
                .await
                .expect("failed to connect");
            let mut compat_client = TokioCompat::new(client);
            use tokio::io::AsyncWriteExt as _;
            compat_client
                .write_all(b"hello from tokio wrapper")
                .await
                .expect("failed to write");
            compat_client.flush().await.expect("failed to flush");
        };

        let server_task = async move {
            let (server, _) = listener.accept().await.expect("failed to accept");
            let mut compat_server = TokioCompat::new(server);
            use tokio::io::AsyncReadExt as _;
            let mut buf = vec![0; 64];
            let n = compat_server.read(&mut buf).await.expect("failed to read");
            assert_eq!(&buf[..n], b"hello from tokio wrapper");
        };

        tokio::join!(client_task, server_task);
    });
}

#[test]
fn test_moirai_compat_tokio_io() {
    futures::executor::block_on(async {
        let cursor = std::io::Cursor::new(vec![1, 2, 3, 4, 5]);
        let mut compat = MoiraiCompat::new(cursor);

        let mut buf = vec![0; 5];
        let n = compat.read(&mut buf).await.expect("failed to read");
        assert_eq!(n, 5);
        assert_eq!(buf, vec![1, 2, 3, 4, 5]);
    });
}

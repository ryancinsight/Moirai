use super::*;
use crate::io::{AsyncLength, AsyncReadAt};
use std::io::SeekFrom;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn test_path(name: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock must be after unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "moirai_async_fs_{name}_{}_{}",
        std::process::id(),
        nonce
    ))
}

#[test]
fn test_file_options() {
    let options = FileOpenOptions::read_only();
    assert!(options.read);
    assert!(!options.write);

    let options = FileOpenOptions::write_only();
    assert!(!options.read);
    assert!(options.write);

    let options = FileOpenOptions::append_only();
    assert!(!options.read);
    assert!(options.write);
    assert!(options.append);
}

#[test]
fn test_file_stats() {
    let stats = FileStats::default();
    assert_eq!(stats.bytes_read, 0);
    assert_eq!(stats.bytes_written, 0);
    assert_eq!(stats.read_operations, 0);
    assert_eq!(stats.write_operations, 0);
    assert_eq!(stats.seek_operations, 0);
}

#[test]
fn test_file_write_read_append_and_stats_values() {
    let path = test_path("roundtrip.txt");
    futures::executor::block_on(async {
        write_str(&path, "alpha")
            .await
            .expect("write_str must succeed");
        append_str(&path, "-beta")
            .await
            .expect("append_str must succeed");

        let contents = read_to_string(&path)
            .await
            .expect("read_to_string must succeed");
        assert_eq!(contents, "alpha-beta");

        let mut file = File::open(&path).await.expect("open must succeed");
        let mut prefix = [0_u8; 5];
        let bytes_read = file.read(&mut prefix).await.expect("read must succeed");
        assert_eq!(bytes_read, 5);
        assert_eq!(&prefix, b"alpha");
        assert_eq!(file.stats().bytes_read, 5);
        assert_eq!(file.stats().read_operations, 1);

        let position = file
            .stream_position()
            .await
            .expect("stream_position must succeed");
        assert_eq!(position, 5);
        let new_position = file
            .seek(SeekFrom::Start(6))
            .await
            .expect("seek must succeed");
        assert_eq!(new_position, 6);
        assert_eq!(file.stats().seek_operations, 1);
    });
    std::fs::remove_file(&path).expect("test file cleanup must succeed");
}

#[test]
fn test_file_positioned_read_preserves_cursor_length_and_stream_stats() {
    let path = test_path("positioned.bin");
    let expected = b"0123456789";
    std::fs::write(&path, expected).expect("positioned source write must succeed");

    futures::executor::block_on(async {
        let mut file = File::open(&path).await.expect("open must succeed");
        let mut prefix = [0_u8; 2];
        let prefix_read = file
            .read(&mut prefix)
            .await
            .expect("stream read must succeed");
        assert_eq!(prefix_read, prefix.len());
        assert_eq!(&prefix, b"01");

        let mut positioned = [0_u8; 4];
        file.read_at(6, &mut positioned)
            .await
            .expect("positioned read must succeed");
        assert_eq!(&positioned, b"6789");
        assert_eq!(
            file.stream_position().await.expect("position must succeed"),
            2
        );
        assert_eq!(
            file.len().await.expect("length must succeed"),
            u64::try_from(expected.len()).expect("fixture length fits u64")
        );
        assert!(!file.is_empty().await.expect("empty query must succeed"));
        assert_eq!(file.stats().bytes_read, 2);
        assert_eq!(file.stats().read_operations, 1);
    });

    std::fs::remove_file(&path).expect("positioned file cleanup must succeed");
}

#[test]
fn test_file_positioned_read_reports_short_source() {
    let path = test_path("positioned-short.bin");
    std::fs::write(&path, b"0123456789").expect("short positioned source write must succeed");

    futures::executor::block_on(async {
        let file = File::open(&path).await.expect("open must succeed");
        let mut output = [0_u8; 4];
        let error = file
            .read_at(8, &mut output)
            .await
            .expect_err("positioned read must reject a short source");
        assert_eq!(error.kind(), std::io::ErrorKind::UnexpectedEof);
        assert_eq!(&output[..2], b"89");
    });

    std::fs::remove_file(&path).expect("short positioned file cleanup must succeed");
}

#[test]
fn test_file_positioned_read_validates_empty_and_overflow_ranges() {
    let path = test_path("positioned-range.bin");
    std::fs::write(&path, b"").expect("positioned source write must succeed");

    futures::executor::block_on(async {
        let file = File::open(&path).await.expect("open must succeed");
        assert_eq!(file.len().await.expect("length must succeed"), 0);
        assert!(file.is_empty().await.expect("empty query must succeed"));

        let mut empty = [];
        file.read_at(u64::MAX, &mut empty)
            .await
            .expect("empty positioned read must not inspect its offset");

        let mut output = [0xA5_u8; 1];
        let error = file
            .read_at(u64::MAX, &mut output)
            .await
            .expect_err("overflowing range must fail before I/O");
        assert_eq!(error.kind(), std::io::ErrorKind::InvalidInput);
        assert_eq!(output, [0xA5]);
    });

    std::fs::remove_file(&path).expect("positioned range file cleanup must succeed");
}

#[test]
fn test_file_copy_and_directory_values() {
    let dir = test_path("dir");
    let source = dir.join("source.bin");
    let dest = dir.join("dest.bin");
    let renamed = dir.join("renamed.bin");
    futures::executor::block_on(async {
        create_dir(&dir).await.expect("create_dir must succeed");
        assert!(dir.is_dir());
        write(&source, b"0123456789")
            .await
            .expect("source write must succeed");
        let copied = copy(&source, &dest).await.expect("copy must succeed");
        assert_eq!(copied, 10);
        let dest_bytes = read(&dest).await.expect("read copied file must succeed");
        assert_eq!(dest_bytes, b"0123456789");
        let dest_metadata = metadata(&dest).await.expect("metadata must succeed");
        assert!(dest_metadata.is_file());
        assert_eq!(dest_metadata.len(), 10);
        rename(&dest, &renamed).await.expect("rename must succeed");
        assert!(!dest.exists());
        let renamed_bytes = read(&renamed)
            .await
            .expect("read renamed file must succeed");
        assert_eq!(renamed_bytes, b"0123456789");
        remove_file(&source)
            .await
            .expect("remove source must succeed");
        assert!(!source.exists());
        remove_file(&renamed)
            .await
            .expect("remove renamed dest must succeed");
        assert!(!renamed.exists());
        remove_dir(&dir).await.expect("remove dir must succeed");
        assert!(!dir.exists());
    });
}

#[test]
fn test_recursive_directory_values() {
    let root = test_path("recursive-dir");
    let leaf = root.join("a").join("b").join("c");
    let marker = leaf.join("marker.bin");
    let expected: Vec<u8> = (0_u8..=63).map(|value| value.wrapping_mul(29)).collect();

    futures::executor::block_on(async {
        create_dir_all(&leaf)
            .await
            .expect("create_dir_all must succeed");
        assert!(leaf.is_dir());
        write(&marker, &expected)
            .await
            .expect("nested marker write must succeed");
        let actual = read(&marker)
            .await
            .expect("nested marker read must succeed");
        assert_eq!(actual, expected);
        remove_dir_all(&root)
            .await
            .expect("remove_dir_all must succeed");
        assert!(!root.exists());
    });
}

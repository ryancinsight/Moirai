use super::*;
use futures::executor::block_on;
use std::io::SeekFrom;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn test_path(name: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock must be after unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "moirai_pal_async_file_{name}_{}_{}",
        std::process::id(),
        nonce
    ))
}

#[test]
fn async_file_roundtrip_seek_and_metadata_are_value_semantic() {
    let path = test_path("roundtrip.bin");
    block_on(async {
        let mut file = AsyncFile::open_with(&path, FileOpenOptions::read_write_truncate())
            .await
            .expect("file create must succeed");
        let written = file.write(b"alpha-beta").await.expect("write must succeed");
        assert_eq!(written, 10);
        file.flush().await.expect("flush must succeed");

        let position = file
            .seek(SeekFrom::Start(6))
            .await
            .expect("seek must succeed");
        assert_eq!(position, 6);

        let mut suffix = [0_u8; 4];
        let read = file.read(&mut suffix).await.expect("read must succeed");
        assert_eq!(read, 4);
        assert_eq!(&suffix, b"beta");

        let metadata = file.metadata().await.expect("metadata must succeed");
        assert_eq!(metadata.len(), 10);
    });
    std::fs::remove_file(&path).expect("test file cleanup must succeed");
}

#[test]
fn async_file_read_to_end_preserves_source_bytes() {
    let path = test_path("source.bin");
    let expected: Vec<u8> = (0_u8..=31).map(|value| value.wrapping_mul(3)).collect();
    std::fs::write(&path, &expected).expect("source write must succeed");

    block_on(async {
        let mut file = AsyncFile::open(&path).await.expect("open must succeed");
        let mut actual = Vec::new();
        let read = file
            .read_to_end(&mut actual)
            .await
            .expect("read_to_end must succeed");
        assert_eq!(read, expected.len());
        assert_eq!(actual, expected);
    });

    std::fs::remove_file(&path).expect("test file cleanup must succeed");
}

#[test]
fn async_file_positioned_read_preserves_stream_cursor() {
    let path = test_path("positioned.bin");
    std::fs::write(&path, b"0123456789").expect("positioned source write must succeed");

    block_on(async {
        let mut file = AsyncFile::open(&path).await.expect("open must succeed");
        let position = file
            .seek(SeekFrom::Start(2))
            .await
            .expect("initial seek must succeed");
        assert_eq!(position, 2);

        let mut positioned = [0_u8; 4];
        let read = file
            .read_at(&mut positioned, 6)
            .expect("positioned read must succeed");
        assert_eq!(read, positioned.len());
        assert_eq!(&positioned, b"6789");

        let cursor = file
            .seek(SeekFrom::Current(0))
            .await
            .expect("cursor query must succeed");
        assert_eq!(cursor, 2);

        let mut stream = [0_u8; 2];
        let stream_read = file
            .read(&mut stream)
            .await
            .expect("stream read must succeed");
        assert_eq!(stream_read, stream.len());
        assert_eq!(&stream, b"23");
    });

    std::fs::remove_file(&path).expect("positioned file cleanup must succeed");
}

#[test]
fn async_file_copy_preserves_source_bytes() {
    let source = test_path("copy-source.bin");
    let dest = test_path("copy-dest.bin");
    let expected: Vec<u8> = (0_u8..=63).map(|value| value.wrapping_mul(5)).collect();
    std::fs::write(&source, &expected).expect("source write must succeed");

    block_on(async {
        let copied = copy(&source, &dest).await.expect("copy must succeed");
        assert_eq!(copied, expected.len() as u64);
        let actual = std::fs::read(&dest).expect("dest read must succeed");
        assert_eq!(actual, expected);
    });

    std::fs::remove_file(&source).expect("source cleanup must succeed");
    std::fs::remove_file(&dest).expect("dest cleanup must succeed");
}

#[test]
fn async_file_write_preserves_source_bytes() {
    let path = test_path("write.bin");
    let expected: Vec<u8> = (0_u8..=127).map(|value| value.wrapping_mul(7)).collect();

    block_on(async {
        write(&path, &expected).await.expect("write must succeed");
        let actual = std::fs::read(&path).expect("written file must be readable");
        assert_eq!(actual, expected);
    });

    std::fs::remove_file(&path).expect("written file cleanup must succeed");
}

#[test]
fn async_file_append_preserves_prefix_and_appended_bytes() {
    let path = test_path("append.bin");
    let prefix: Vec<u8> = (0_u8..=31).map(|value| value.wrapping_mul(3)).collect();
    let suffix: Vec<u8> = (0_u8..=31).map(|value| value.wrapping_mul(11)).collect();
    std::fs::write(&path, &prefix).expect("prefix write must succeed");

    block_on(async {
        append(&path, &suffix).await.expect("append must succeed");
        let actual = std::fs::read(&path).expect("appended file must be readable");
        assert_eq!(&actual[..prefix.len()], prefix.as_slice());
        assert_eq!(&actual[prefix.len()..], suffix.as_slice());
    });

    std::fs::remove_file(&path).expect("appended file cleanup must succeed");
}

#[test]
fn async_file_metadata_preserves_file_type_and_length() {
    let path = test_path("metadata.bin");
    let expected: Vec<u8> = (0_u8..=95).map(|value| value.wrapping_mul(13)).collect();
    std::fs::write(&path, &expected).expect("metadata source write must succeed");

    block_on(async {
        let actual = metadata(&path).await.expect("metadata must succeed");
        assert!(actual.is_file());
        assert_eq!(actual.len(), expected.len() as u64);
    });

    std::fs::remove_file(&path).expect("metadata file cleanup must succeed");
}

#[test]
fn async_file_rename_preserves_source_bytes_at_destination() {
    let source = test_path("rename-source.bin");
    let dest = test_path("rename-dest.bin");
    let expected: Vec<u8> = (0_u8..=79).map(|value| value.wrapping_mul(17)).collect();
    std::fs::write(&source, &expected).expect("rename source write must succeed");

    block_on(async {
        rename(&source, &dest).await.expect("rename must succeed");
        assert!(!source.exists());
        let actual = std::fs::read(&dest).expect("renamed dest read must succeed");
        assert_eq!(actual, expected);
    });

    std::fs::remove_file(&dest).expect("renamed file cleanup must succeed");
}

#[test]
fn async_file_remove_file_deletes_expected_path() {
    let path = test_path("remove.bin");
    let expected: Vec<u8> = (0_u8..=47).map(|value| value.wrapping_mul(19)).collect();
    std::fs::write(&path, &expected).expect("remove source write must succeed");

    block_on(async {
        let actual = std::fs::read(&path).expect("remove source read must succeed");
        assert_eq!(actual, expected);
        remove_file(&path).await.expect("remove_file must succeed");
        assert!(!path.exists());
    });
}

#[test]
fn async_dir_create_and_remove_preserves_directory_state() {
    let dir = test_path("dir");

    block_on(async {
        create_dir(&dir).await.expect("create_dir must succeed");
        let metadata = std::fs::metadata(&dir).expect("created dir metadata must exist");
        assert!(metadata.is_dir());
        remove_dir(&dir).await.expect("remove_dir must succeed");
        assert!(!dir.exists());
    });
}

#[test]
fn async_dir_all_create_and_remove_deletes_nested_tree() {
    let root = test_path("dir-all");
    let leaf = root.join("alpha").join("beta");
    let marker = leaf.join("marker.bin");
    let expected: Vec<u8> = (0_u8..=31).map(|value| value.wrapping_mul(23)).collect();

    block_on(async {
        create_dir_all(&leaf)
            .await
            .expect("create_dir_all must succeed");
        assert!(leaf.is_dir());
        std::fs::write(&marker, &expected).expect("nested marker write must succeed");
        let actual = std::fs::read(&marker).expect("nested marker read must succeed");
        assert_eq!(actual, expected);
        remove_dir_all(&root)
            .await
            .expect("remove_dir_all must succeed");
        assert!(!root.exists());
    });
}

use super::*;
use std::io::SeekFrom;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn test_path(name: &str) -> PathBuf {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock must be after unix epoch")
        .as_nanos();
    std::env::temp_dir().join(format!(
        "moirai_pal_file_{name}_{}_{}",
        std::process::id(),
        nonce
    ))
}

#[test]
fn file_roundtrip_seek_and_metadata_are_value_semantic() {
    let path = test_path("roundtrip.bin");
    let file = File::open_with(&path, FileOpenOptions::read_write_truncate())
        .expect("file create must succeed");
    let written = file.write(b"alpha-beta").expect("write must succeed");
    assert_eq!(written, 10);

    let position = file.seek(SeekFrom::Start(6)).expect("seek must succeed");
    assert_eq!(position, 6);

    let mut suffix = [0_u8; 4];
    let read = file.read(&mut suffix).expect("read must succeed");
    assert_eq!(read, 4);
    assert_eq!(&suffix, b"beta");

    let metadata = file.metadata().expect("metadata must succeed");
    assert_eq!(metadata.len(), 10);
    drop(file);
    std::fs::remove_file(&path).expect("test file cleanup must succeed");
}

#[test]
fn file_read_to_end_preserves_source_bytes() {
    let path = test_path("source.bin");
    let expected: Vec<u8> = (0_u8..=31).map(|value| value.wrapping_mul(3)).collect();
    std::fs::write(&path, &expected).expect("source write must succeed");

    let file = File::open_with(&path, FileOpenOptions::read_only()).expect("open must succeed");
    let mut actual = Vec::new();
    let read = file
        .read_to_end(&mut actual)
        .expect("read_to_end must succeed");
    assert_eq!(read, expected.len());
    assert_eq!(actual, expected);

    drop(file);
    std::fs::remove_file(&path).expect("test file cleanup must succeed");
}

#[test]
fn file_positioned_read_preserves_stream_cursor() {
    let path = test_path("positioned.bin");
    std::fs::write(&path, b"0123456789").expect("positioned source write must succeed");

    let file = File::open_with(&path, FileOpenOptions::read_only()).expect("open must succeed");
    let position = file
        .seek(SeekFrom::Start(2))
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
        .expect("cursor query must succeed");
    assert_eq!(cursor, 2);

    let mut stream = [0_u8; 2];
    let stream_read = file.read(&mut stream).expect("stream read must succeed");
    assert_eq!(stream_read, stream.len());
    assert_eq!(&stream, b"23");

    drop(file);
    std::fs::remove_file(&path).expect("positioned file cleanup must succeed");
}

#[test]
fn concurrent_positioned_reads_never_move_the_stream_cursor() {
    // Positioned reads on one thread and stream reads on another share a
    // handle, as they do across blocking-pool workers. On Windows the cursor
    // lock keeps every stream read at the cursor the previous one left.
    let path = test_path("concurrent.bin");
    let content: Vec<u8> = (0..=255_u8).collect();
    std::fs::write(&path, &content).expect("source write must succeed");
    let file = std::sync::Arc::new(
        File::open_with(&path, FileOpenOptions::read_only()).expect("open must succeed"),
    );

    let positioned = {
        let file = std::sync::Arc::clone(&file);
        std::thread::spawn(move || {
            let mut byte = [0_u8; 1];
            for _ in 0..2_000 {
                file.read_at(&mut byte, 200)
                    .expect("positioned read must succeed");
                assert_eq!(byte[0], 200);
            }
        })
    };
    let mut streamed = Vec::with_capacity(content.len());
    let mut byte = [0_u8; 1];
    while file.read(&mut byte).expect("stream read must succeed") == 1 {
        streamed.push(byte[0]);
    }
    positioned.join().expect("positioned reader must not panic");

    assert_eq!(streamed, content, "a stream read observed a moved cursor");
    drop(file);
    std::fs::remove_file(&path).expect("test file cleanup must succeed");
}

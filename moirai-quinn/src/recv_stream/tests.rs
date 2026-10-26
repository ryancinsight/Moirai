use super::*;

#[test]
fn read_to_end_reordered_limit() {
    let mut buffer = ReadToEndBuffer::new(4);
    buffer
        .push(Chunk {
            offset: 4,
            bytes: Bytes::from_static(b"efgh"),
        })
        .unwrap();
    assert_eq!(
        buffer.push(Chunk {
            offset: 0,
            bytes: Bytes::from_static(b"abcd")
        }),
        Err(ReadToEndError::TooLong)
    );
    assert_eq!(buffer.read.len(), 1);
}

#[test]
fn rejected_chunk_preserves_the_previous_read_span() {
    let mut buffer = ReadToEndBuffer::new(4);
    buffer
        .push(Chunk {
            offset: 104,
            bytes: Bytes::from_static(b"efgh"),
        })
        .unwrap();
    assert_eq!(
        buffer.push(Chunk {
            offset: 100,
            bytes: Bytes::from_static(b"abcde"),
        }),
        Err(ReadToEndError::TooLong)
    );
    assert_eq!(buffer.finish(), Ok(b"efgh".to_vec()));
}

#[test]
fn overflowing_chunk_end_is_rejected_without_mutating_the_buffer() {
    let mut buffer = ReadToEndBuffer::new(usize::MAX);
    assert_eq!(
        buffer.push(Chunk {
            offset: u64::MAX - 1,
            bytes: Bytes::from_static(b"abcd"),
        }),
        Err(ReadToEndError::TooLong)
    );
    assert_eq!(buffer.finish(), Ok(Vec::new()));
}

#[test]
fn read_to_end_reordered_exact_limit_after_prefix() {
    let mut buffer = ReadToEndBuffer::new(8);
    // A previously consumed prefix must not count against the remaining-data limit.
    buffer
        .push(Chunk {
            offset: 104,
            bytes: Bytes::from_static(b"efgh"),
        })
        .unwrap();
    buffer
        .push(Chunk {
            offset: 100,
            bytes: Bytes::from_static(b"abcd"),
        })
        .unwrap();
    assert_eq!(buffer.finish(), Ok(b"abcdefgh".to_vec()));
}

#[test]
fn read_to_end_gaps_and_empty() {
    assert_eq!(ReadToEndBuffer::new(0).finish(), Ok(Vec::new()));
    let mut buffer = ReadToEndBuffer::new(5);
    buffer
        .push(Chunk {
            offset: 14,
            bytes: Bytes::from_static(b"e"),
        })
        .unwrap();
    buffer
        .push(Chunk {
            offset: 10,
            bytes: Bytes::from_static(b"a"),
        })
        .unwrap();
    assert_eq!(buffer.finish(), Ok(b"a\0\0\0e".to_vec()));
    let mut buffer = ReadToEndBuffer::new(4);
    buffer
        .push(Chunk {
            offset: 10,
            bytes: Bytes::from_static(b"a"),
        })
        .unwrap();
    assert_eq!(
        buffer.push(Chunk {
            offset: 14,
            bytes: Bytes::from_static(b"e")
        }),
        Err(ReadToEndError::TooLong)
    );
}

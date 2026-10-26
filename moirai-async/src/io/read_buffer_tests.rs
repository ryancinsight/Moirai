use super::ReadBuffer;
use std::mem::MaybeUninit;

fn trace() -> tracing::subscriber::DefaultGuard {
    tracing::subscriber::set_default(tracing_subscriber::fmt().with_test_writer().finish())
}

#[test]
fn initialized_storage_preserves_prefix_and_partial_progress() {
    let _trace = trace();
    let _arrange = tracing::info_span!("arrange").entered();
    let mut storage = [0xa5; 7];
    let mut buffer = ReadBuffer::new(&mut storage);
    drop(_arrange);
    let _act = tracing::info_span!("act").entered();
    buffer.put_slice(&[3, 5]);
    buffer.put_slice(&[7, 11, 13]);
    drop(_act);
    let _assert = tracing::info_span!("assert").entered();
    assert_eq!(buffer.capacity(), 7);
    assert_eq!(buffer.remaining(), 2);
    assert_eq!(buffer.filled(), &[3, 5, 7, 11, 13]);
    assert_eq!(buffer.initialized(), &[3, 5, 7, 11, 13, 0xa5, 0xa5]);
}

#[test]
fn uninitialized_storage_exposes_only_written_bytes() {
    let _trace = trace();
    let _arrange = tracing::info_span!("arrange").entered();
    let mut storage = [MaybeUninit::uninit(); 7];
    let mut buffer = ReadBuffer::uninit(&mut storage);
    drop(_arrange);
    let _act = tracing::info_span!("act").entered();
    assert_eq!(buffer.initialized(), &[]);
    buffer.put_slice(&[17, 19, 23]);
    drop(_act);
    let _assert = tracing::info_span!("assert").entered();
    assert_eq!(buffer.remaining(), 4);
    assert_eq!(buffer.filled(), &[17, 19, 23]);
    assert_eq!(buffer.initialized(), &[17, 19, 23]);
}

#[test]
fn clearing_retains_initialized_bytes_for_reuse() {
    let _trace = trace();
    let _arrange = tracing::info_span!("arrange").entered();
    let mut storage = [MaybeUninit::uninit(); 5];
    let mut buffer = ReadBuffer::uninit(&mut storage);
    drop(_arrange);
    let _act = tracing::info_span!("act").entered();
    buffer.put_slice(&[29, 31, 37]);
    buffer.clear();
    assert_eq!(buffer.filled(), &[]);
    assert_eq!(buffer.initialized(), &[29, 31, 37]);
    buffer.put_slice(&[41]);
    buffer.advance(2);
    drop(_act);
    let _assert = tracing::info_span!("assert").entered();
    assert_eq!(buffer.filled(), &[41, 31, 37]);
    assert_eq!(buffer.remaining(), 2);
}

#[test]
fn initialization_fills_only_the_unknown_tail() {
    let _trace = trace();
    let _arrange = tracing::info_span!("arrange").entered();
    let mut storage = [MaybeUninit::uninit(); 5];
    let mut buffer = ReadBuffer::uninit(&mut storage);
    drop(_arrange);
    let _act = tracing::info_span!("act").entered();
    buffer.put_slice(&[43, 47]);
    buffer.initialize_unfilled()[..2].copy_from_slice(&[53, 59]);
    buffer.advance(2);
    drop(_act);
    let _assert = tracing::info_span!("assert").entered();
    assert_eq!(buffer.filled(), &[43, 47, 53, 59]);
    assert_eq!(buffer.initialized(), &[43, 47, 53, 59, 0]);
    assert_eq!(buffer.remaining(), 1);
}

#[test]
fn empty_buffers_and_empty_appends_preserve_the_invariant() {
    let _trace = trace();
    let _arrange = tracing::info_span!("arrange").entered();
    let mut initialized = [];
    let mut uninitialized = [];
    let mut buffers = [
        ReadBuffer::new(&mut initialized),
        ReadBuffer::uninit(&mut uninitialized),
    ];
    drop(_arrange);
    let _act = tracing::info_span!("act").entered();
    for buffer in &mut buffers {
        buffer.put_slice(&[]);
        buffer.advance(0);
        buffer.clear();
        drop(tracing::info_span!("assert").entered());
        assert_eq!(buffer.capacity(), 0);
        assert_eq!(buffer.remaining(), 0);
        assert_eq!(buffer.filled(), &[]);
        assert_eq!(buffer.initialize_unfilled(), &[]);
    }
}

#[test]
fn append_overflow_rejects_before_mutating_either_prefix() {
    let _trace = trace();
    let _arrange = tracing::info_span!("arrange").entered();
    let mut storage = [MaybeUninit::uninit(); 3];
    let mut buffer = ReadBuffer::uninit(&mut storage);
    buffer.put_slice(&[61, 67]);
    drop(_arrange);
    let _act = tracing::info_span!("act").entered();
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| buffer.put_slice(&[71, 73])));
    drop(_act);
    let _assert = tracing::info_span!("assert").entered();
    assert!(result.is_err());
    assert_eq!(buffer.filled(), &[61, 67]);
    assert_eq!(buffer.initialized(), &[61, 67]);
    assert_eq!(buffer.remaining(), 1);
}

#[test]
fn advance_cannot_expose_an_uninitialized_byte_or_overflow() {
    let _trace = trace();
    let _arrange = tracing::info_span!("arrange").entered();
    let mut storage = [MaybeUninit::uninit(); 3];
    let mut buffer = ReadBuffer::uninit(&mut storage);
    buffer.put_slice(&[79]);
    drop(_arrange);
    let _act = tracing::info_span!("act").entered();
    for count in [1, usize::MAX] {
        let result =
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| buffer.advance(count)));
        let _assert = tracing::info_span!("assert", count).entered();
        assert!(result.is_err());
        assert_eq!(buffer.filled(), &[79]);
        assert_eq!(buffer.initialized(), &[79]);
        assert_eq!(buffer.remaining(), 2);
    }
}

use super::*;
use crate::{Parallel, Sequential};
use core::sync::atomic::{AtomicUsize, Ordering};

fn assert_six_buffer_chunks<P>()
where
    P: ExecutionPolicy,
{
    const BUFFER_COUNT: usize = 6;
    const LENGTH: usize = 19;
    const CHUNK_SIZE: usize = 4;

    let mut buffers = core::array::from_fn::<_, BUFFER_COUNT, _>(|_| [0_usize; LENGTH]);
    let [a, b, c, d, e, f] = &mut buffers;
    for_each_chunk_buffers_mut_enumerated_with::<P, _, _, BUFFER_COUNT>(
        [a, b, c, d, e, f],
        CHUNK_SIZE,
        |chunk_index, chunks| {
            for (buffer_index, chunk) in chunks.into_iter().enumerate() {
                for (lane, value) in chunk.iter_mut().enumerate() {
                    let absolute = chunk_index * CHUNK_SIZE + lane;
                    *value += buffer_index * 1_000 + absolute + 1;
                }
            }
        },
    )
    .expect("equal test buffers must validate");

    for (buffer_index, buffer) in buffers.iter().enumerate() {
        for (index, &value) in buffer.iter().enumerate() {
            assert_eq!(value, buffer_index * 1_000 + index + 1);
        }
    }
}

#[test]
fn chunk_buffers_cover_ragged_tail_sequentially() {
    assert_six_buffer_chunks::<Sequential>();
}

#[test]
fn chunk_buffers_cover_ragged_tail_in_parallel() {
    assert_six_buffer_chunks::<Parallel>();
}

#[test]
fn chunk_buffers_reject_length_mismatch_before_mutation() {
    let mut left = [3_u8; 5];
    let mut right = [7_u8; 4];
    let error = for_each_chunk_buffers_mut_enumerated_with::<Sequential, _, _, 2>(
        [&mut left, &mut right],
        2,
        |_, chunks| chunks.into_iter().for_each(|chunk| chunk.fill(0)),
    )
    .expect_err("unequal buffer lengths must fail");

    assert_eq!(
        error,
        ChunkBuffersError::LengthMismatch {
            buffer_index: 1,
            expected: 5,
            actual: 4,
        }
    );
    assert_eq!(left, [3; 5]);
    assert_eq!(right, [7; 4]);
}

#[test]
fn chunk_buffers_treat_empty_shapes_as_no_ops() {
    let calls = AtomicUsize::new(0);
    for_each_chunk_buffers_mut_enumerated_with::<Sequential, u8, _, 0>([], 4, |_, _| {
        calls.fetch_add(1, Ordering::Relaxed);
    })
    .expect("zero buffers must be valid");

    let mut empty: [u8; 0] = [];
    for_each_chunk_buffers_mut_enumerated_with::<Sequential, _, _, 1>([&mut empty], 4, |_, _| {
        calls.fetch_add(1, Ordering::Relaxed);
    })
    .expect("empty buffers must be valid");

    let mut values = [1_u8, 2, 3];
    for_each_chunk_buffers_mut_enumerated_with::<Sequential, _, _, 1>([&mut values], 0, |_, _| {
        calls.fetch_add(1, Ordering::Relaxed);
    })
    .expect("zero chunk size must be a no-op");

    assert_eq!(calls.load(Ordering::Relaxed), 0);
    assert_eq!(values, [1, 2, 3]);
}

use super::*;
use crate::{Parallel, Sequential};
use core::sync::atomic::{AtomicUsize, Ordering};

#[derive(Debug)]
struct AssertChunkGeometry<const LEN: usize, const CHUNKS: usize>;

impl<const LEN: usize, const CHUNKS: usize> ExecutionPolicy for AssertChunkGeometry<LEN, CHUNKS> {
    fn parallelize(_len: usize) -> bool {
        panic!("chunk operators must dispatch through parallelize_chunks")
    }

    fn parallelize_chunks(len: usize, chunks: usize) -> bool {
        assert_eq!(len, LEN);
        assert_eq!(chunks, CHUNKS);
        false
    }
}

#[test]
fn chunk_operators_report_element_and_task_geometry() {
    type Ragged = AssertChunkGeometry<7, 3>;

    let mut values = [0_u8; 7];
    for_each_chunk_mut_with::<Ragged, _, _>(&mut values, 3, |chunk| chunk.fill(1));
    assert_eq!(values, [1; 7]);

    let mut stateful = [0_u8; 7];
    for_each_chunk_mut_with_state::<Ragged, _, _, _, _>(
        &mut stateful,
        3,
        || 2_u8,
        |state, chunk| chunk.fill(*state),
    );
    assert_eq!(stateful, [2; 7]);

    let mut enumerated = [0_usize; 7];
    for_each_chunk_mut_enumerated_with::<Ragged, _, _>(&mut enumerated, 3, |chunk_index, chunk| {
        chunk.fill(chunk_index)
    });
    assert_eq!(enumerated, [0, 0, 0, 1, 1, 1, 2]);

    let mut pair_left = [0_usize; 7];
    let mut pair_right = [0_usize; 7];
    for_each_chunk_pair_mut_enumerated_with::<Ragged, _, _, _>(
        &mut pair_left,
        &mut pair_right,
        3,
        |chunk_index, left, right| {
            left.fill(chunk_index + 1);
            right.fill(chunk_index + 4);
        },
    );
    assert_eq!(pair_left, [1, 1, 1, 2, 2, 2, 3]);
    assert_eq!(pair_right, [4, 4, 4, 5, 5, 5, 6]);

    let mut triple = [[0_usize; 7]; 3];
    let [first, second, third] = &mut triple;
    for_each_chunk_triple_mut_enumerated_with::<Ragged, _, _, _, _>(
        first,
        second,
        third,
        3,
        |chunk_index, first, second, third| {
            first.fill(chunk_index);
            second.fill(chunk_index + 3);
            third.fill(chunk_index + 6);
        },
    );
    assert_eq!(triple[0], [0, 0, 0, 1, 1, 1, 2]);
    assert_eq!(triple[1], [3, 3, 3, 4, 4, 4, 5]);
    assert_eq!(triple[2], [6, 6, 6, 7, 7, 7, 8]);

    let mut quad = [[0_usize; 7]; 4];
    let [first, second, third, fourth] = &mut quad;
    for_each_chunk_quad_mut_enumerated_with::<Ragged, _, _, _, _, _>(
        first,
        second,
        third,
        fourth,
        3,
        |chunk_index, first, second, third, fourth| {
            first.fill(chunk_index);
            second.fill(chunk_index + 3);
            third.fill(chunk_index + 6);
            fourth.fill(chunk_index + 9);
        },
    );
    assert_eq!(quad[0], [0, 0, 0, 1, 1, 1, 2]);
    assert_eq!(quad[1], [3, 3, 3, 4, 4, 4, 5]);
    assert_eq!(quad[2], [6, 6, 6, 7, 7, 7, 8]);
    assert_eq!(quad[3], [9, 9, 9, 10, 10, 10, 11]);

    let mut buffers = [[0_usize; 7]; 2];
    let [left, right] = &mut buffers;
    for_each_chunk_buffers_mut_enumerated_with::<Ragged, _, _, 2>(
        [left, right],
        3,
        |chunk_index, [left, right]| {
            left.fill(chunk_index);
            right.fill(chunk_index + 3);
        },
    )
    .expect("equal test buffers must validate");
    assert_eq!(buffers[0], [0, 0, 0, 1, 1, 1, 2]);
    assert_eq!(buffers[1], [3, 3, 3, 4, 4, 4, 5]);
}

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

/// Busy-waits `nanos` on the calling thread so a task has a known cost
/// without touching memory the other tasks share.
fn spin_for(nanos: u64) {
    if nanos == 0 {
        return;
    }
    let start = std::time::Instant::now();
    while start.elapsed().as_nanos() < u128::from(nanos) {
        core::hint::spin_loop();
    }
}

/// Measurement instrument for the fork-join itself, not for any kernel: the
/// distribution of one `for_each_chunk_mut_with` call over 64 chunks whose
/// tasks cost nothing, ten microseconds, or fifty, against the sequential
/// policy running the same loop on the calling thread. A consumer pays the
/// median and the tail of this per pass; a probe that reports minima never
/// sees them. Reports; asserts nothing. Run under a release profile with
/// `--run-ignored all --no-capture`.
#[test]
#[ignore = "measurement instrument for the fork-join latency distribution"]
fn fork_join_latency_distribution() {
    const CHUNKS: usize = 64;
    const CALLS: usize = 2_000;
    const WARM: usize = 100;
    if cfg!(debug_assertions) {
        eprintln!("fork_join_latency_distribution: built without optimization; no reading");
        return;
    }
    let mut data = vec![0_u64; CHUNKS];
    for (label, nanos) in [("empty", 0_u64), ("10us", 10_000), ("50us", 50_000)] {
        for (policy, run) in [
            (
                "parallel",
                (|data: &mut [u64], nanos: u64| {
                    for_each_chunk_mut_with::<Parallel, _, _>(data, 1, |chunk| {
                        spin_for(nanos);
                        chunk[0] = chunk[0].wrapping_add(1);
                    });
                }) as fn(&mut [u64], u64),
            ),
            ("sequential", |data: &mut [u64], nanos: u64| {
                for_each_chunk_mut_with::<Sequential, _, _>(data, 1, |chunk| {
                    spin_for(nanos);
                    chunk[0] = chunk[0].wrapping_add(1);
                });
            }),
        ] {
            for _ in 0..WARM {
                run(&mut data, nanos);
            }
            let mut samples = Vec::with_capacity(CALLS);
            for _ in 0..CALLS {
                let start = std::time::Instant::now();
                run(core::hint::black_box(&mut data), nanos);
                samples.push(start.elapsed());
            }
            samples.sort();
            let at = |q: f64| samples[((samples.len() - 1) as f64 * q) as usize];
            println!(
                "FORK-JOIN {CHUNKS} tasks of {label} {policy}: min {:.1} us  median {:.1} us  p90 {:.1} us  p99 {:.1} us  max {:.1} us",
                samples[0].as_secs_f64() * 1e6,
                at(0.5).as_secs_f64() * 1e6,
                at(0.9).as_secs_f64() * 1e6,
                at(0.99).as_secs_f64() * 1e6,
                samples[samples.len() - 1].as_secs_f64() * 1e6,
            );
        }
    }
}

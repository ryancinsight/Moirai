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

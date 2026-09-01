use super::{
    parallel::{
        reduce_owned_pairs, should_execute_scoped_cache, zero_copy_chunk_size_for_lanes,
        DEFAULT_RING_BUFFER_CAPACITY,
    },
    CacheIterExt, WindowIterator, ZeroCopyParallelIter, CACHE_CHUNK_SIZE, CACHE_LINE_SIZE,
};
use std::{
    mem,
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
};

struct NonClone(u64);

#[repr(C)]
struct WideInput {
    index: usize,
    padding: [u8; 248],
}

impl WideInput {
    fn new(index: usize) -> Self {
        Self {
            index,
            padding: [0; 248],
        }
    }
}

const WIDE_CHUNK: usize = 16_385;
const WIDE_FULL_LEN: usize = WIDE_CHUNK * 4;

fn wide_data(len: usize) -> Vec<WideInput> {
    (0..len).map(WideInput::new).collect()
}

#[test]
fn cache_chunks_advance_for_wide_elements() {
    #[repr(align(8))]
    struct Wide([u64; 24]);

    let data: Vec<Wide> = (0..8).map(|index| Wide([index; 24])).collect();
    let chunks: Vec<&[Wide]> = data.cache_chunks().take(64).collect();

    assert!(chunks.iter().all(|chunk| !chunk.is_empty()));
    assert_eq!(
        chunks
            .iter()
            .flat_map(|chunk| chunk.iter().map(|wide| wide.0[0]))
            .collect::<Vec<_>>(),
        (0..8).collect::<Vec<u64>>()
    );
}

#[test]
fn window_iterator_preserves_overlap_and_ragged_tail() {
    let data = [1, 2, 3, 4, 5, 6, 7, 8];
    let windows: Vec<_> = WindowIterator::new(&data, 3, 2).collect();
    assert_eq!(windows, [&[1, 2, 3][..], &[3, 4, 5], &[5, 6, 7], &[7, 8]]);
}

#[test]
fn cache_aligned_chunks_cover_every_element() {
    let data: Vec<i32> = (0..1_000).collect();
    let chunks: Vec<_> = data.cache_chunks().collect();
    assert!(!chunks.is_empty());
    assert_eq!(chunks.iter().map(|chunk| chunk.len()).sum::<usize>(), 1_000);
}

#[test]
fn zero_copy_for_each_visits_every_value() {
    let data: Vec<i32> = (0..10_000).collect();
    let sum = std::sync::atomic::AtomicI64::new(0);
    data.zero_copy_par_iter().for_each(|&value| {
        sum.fetch_add(value.into(), std::sync::atomic::Ordering::Relaxed);
    });
    assert_eq!(
        sum.load(std::sync::atomic::Ordering::Relaxed),
        (0_i64..10_000).sum()
    );
}

#[test]
fn zero_copy_map_borrows_data_and_closure() {
    let data: Vec<i32> = (0..1_024).collect();
    let factor = 3;
    assert_eq!(
        data.zero_copy_par_iter().map(|value| value * factor),
        data.iter().map(|value| value * factor).collect::<Vec<_>>()
    );
}

#[test]
fn zero_copy_map_matches_sequential_values() {
    let data: Vec<u64> = (0..10_000).collect();
    assert_eq!(
        data.zero_copy_par_iter()
            .map(|value| value.wrapping_mul(5).wrapping_add(7)),
        data.iter()
            .map(|value| value.wrapping_mul(5).wrapping_add(7))
            .collect::<Vec<_>>()
    );
}

#[test]
fn zero_copy_map_parallel_path_preserves_full_and_ragged_ranges() {
    struct NonCloneOutput(usize);

    for len in [WIDE_FULL_LEN, WIDE_FULL_LEN + 7] {
        let data = wide_data(len);
        let iter = ZeroCopyParallelIter {
            data: &data,
            chunk_size: WIDE_CHUNK,
        };
        assert!(
            should_execute_scoped_cache::<WideInput>(len, WIDE_CHUNK),
            "fixture must reach joined fan-out"
        );

        let mapped = iter.map(|input| NonCloneOutput(input.index * 3 + 1));

        assert_eq!(mapped.len(), len);
        assert!(mapped
            .iter()
            .enumerate()
            .all(|(index, output)| output.0 == index * 3 + 1));
    }
}

#[test]
fn zero_copy_map_parallel_path_preserves_zero_sized_outputs() {
    let data = wide_data(WIDE_FULL_LEN + 7);
    let mapped = ZeroCopyParallelIter {
        data: &data,
        chunk_size: WIDE_CHUNK,
    }
    .map(|_| ());

    assert_eq!(mapped, vec![(); data.len()]);
}

#[test]
fn zero_copy_map_drops_every_initialized_output_on_mapper_panic() {
    struct TrackedOutput(Arc<AtomicUsize>);

    impl Drop for TrackedOutput {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::Relaxed);
        }
    }

    let data = wide_data(WIDE_FULL_LEN + 7);
    let created = Arc::new(AtomicUsize::new(0));
    let dropped = Arc::new(AtomicUsize::new(0));
    let output_created = Arc::clone(&created);
    let output_dropped = Arc::clone(&dropped);

    let result = catch_unwind(AssertUnwindSafe(|| {
        ZeroCopyParallelIter {
            data: &data,
            chunk_size: WIDE_CHUNK,
        }
        .map(move |input| {
            assert_ne!(input.index, WIDE_CHUNK + 3, "mapper panic sentinel");
            output_created.fetch_add(1, Ordering::Relaxed);
            TrackedOutput(Arc::clone(&output_dropped))
        })
    }));

    let payload = match result {
        Err(payload) => payload,
        Ok(mapped) => panic!(
            "invariant: mapper sentinel must panic, but returned {} outputs",
            mapped.len()
        ),
    };
    let message = payload
        .downcast_ref::<String>()
        .map(String::as_str)
        .or_else(|| payload.downcast_ref::<&str>().copied())
        .expect("invariant: scheduler propagates a string panic payload");
    assert!(
        message.contains("indexed fan-out failed after partial execution"),
        "unexpected propagated panic: {message}"
    );
    assert_eq!(
        dropped.load(Ordering::Relaxed),
        created.load(Ordering::Relaxed)
    );
}

#[test]
fn reduce_owned_pairs_moves_non_clone_odd_value() {
    let reduced = reduce_owned_pairs(
        vec![NonClone(1), NonClone(2), NonClone(3)],
        &|left, right| NonClone(left.0 + right.0),
    )
    .into_iter()
    .map(|item| item.0)
    .collect::<Vec<_>>();
    assert_eq!(reduced, [3, 3]);
}

#[test]
fn scoped_execution_gate_uses_batch_capacity_floor() {
    let cache_chunk_items = (CACHE_CHUNK_SIZE / mem::size_of::<u64>()).max(1);
    let floor = cache_chunk_items * DEFAULT_RING_BUFFER_CAPACITY;
    assert!(!should_execute_scoped_cache::<u64>(
        floor,
        cache_chunk_items
    ));
    assert!(should_execute_scoped_cache::<u64>(
        floor + 1,
        cache_chunk_items
    ));
}

#[test]
fn chunk_planning_preserves_cache_floor_and_progress() {
    assert_eq!(zero_copy_chunk_size_for_lanes(0, 8, 8), 2_048);
    assert_eq!(zero_copy_chunk_size_for_lanes(1_024, 8, 8), 2_048);
    assert_eq!(zero_copy_chunk_size_for_lanes(32_768, 8, 8), 4_096);
    assert_eq!(zero_copy_chunk_size_for_lanes(32_768, 8, 0), 32_768);
    assert_eq!(
        zero_copy_chunk_size_for_lanes(1, CACHE_CHUNK_SIZE * 2, 64),
        1
    );
}

#[test]
fn zero_copy_for_each_accepts_wide_elements() {
    #[repr(align(8))]
    struct Wide([u64; 24]);

    assert!(mem::size_of::<Wide>() > CACHE_LINE_SIZE);
    let floor = (CACHE_CHUNK_SIZE / mem::size_of::<Wide>()).max(1) * DEFAULT_RING_BUFFER_CAPACITY;
    let data = (0..=floor)
        .map(|index| Wide([index as u64; 24]))
        .collect::<Vec<_>>();
    let visits = (0..data.len())
        .map(|_| std::sync::atomic::AtomicUsize::new(0))
        .collect::<Vec<_>>();
    let iter = ZeroCopyParallelIter {
        data: &data,
        chunk_size: floor,
    };

    iter.for_each(|value| {
        visits[value.0[0] as usize].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    });

    assert!(visits
        .iter()
        .all(|count| count.load(std::sync::atomic::Ordering::Relaxed) == 1));
}

#[test]
fn zero_copy_reduce_accepts_non_clone_reducer() {
    let data = [1_u64, 2, 3, 4];
    let token = NonClone(1);
    assert_eq!(
        data.zero_copy_par_iter()
            .reduce(move |left, right| left + right + token.0),
        Some(13)
    );
}

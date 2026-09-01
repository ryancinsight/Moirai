use super::support::{
    allocation_snapshot, map_source, map_values, zero_copy_parallel_map_values, MAP_LEN,
    MAP_OUTPUT_BYTES, ZERO_COPY_MAP_LEN, ZERO_COPY_MAP_OUTPUT_BYTES, ZERO_COPY_PARALLEL_MAP_LEN,
    ZERO_COPY_PARALLEL_MAP_OUTPUT_BYTES,
};
use moirai_iter::cache::{CacheIterExt, CACHE_CHUNK_SIZE};
use std::{hint::black_box, mem::size_of};

fn process_lanes() -> usize {
    std::thread::available_parallelism().map_or(1, usize::from)
}

fn parallel_iter_completion_bytes() -> usize {
    let chunk_size = MAP_LEN.div_ceil(process_lanes()).max(1);
    MAP_LEN.div_ceil(chunk_size) * size_of::<usize>()
}

fn cache_map_completion_bytes() -> usize {
    let cache_floor = (CACHE_CHUNK_SIZE / size_of::<u64>()).max(1);
    let chunk_size = (ZERO_COPY_PARALLEL_MAP_LEN / process_lanes()).max(cache_floor);
    ZERO_COPY_PARALLEL_MAP_LEN.div_ceil(chunk_size) * size_of::<usize>()
}

#[test]
fn parallel_iter_map_records_output_allocation_ledger() {
    let warm = map_values(map_source());
    assert_eq!(warm.len(), MAP_LEN);

    let input = map_source();
    let (before_allocations, before_bytes) = allocation_snapshot();
    let mapped = map_values(input);
    let (after_allocations, after_bytes) = allocation_snapshot();
    let allocations = after_allocations.saturating_sub(before_allocations);
    let allocated_bytes = after_bytes.saturating_sub(before_bytes);

    assert!(mapped.iter().enumerate().all(|(index, value)| {
        let input = (index as u64).wrapping_mul(31).wrapping_add(7);
        *value == input.wrapping_mul(3).wrapping_add(1)
    }));
    assert_eq!(allocations, 2);
    assert_eq!(
        allocated_bytes,
        MAP_OUTPUT_BYTES + parallel_iter_completion_bytes(),
        "the warmed map allocates only output and completion metadata"
    );
}

#[test]
fn zero_copy_map_separates_constructor_and_output_allocations() {
    let data: Vec<u64> = (0..ZERO_COPY_MAP_LEN as u64)
        .map(|value| value.wrapping_mul(17).wrapping_add(11))
        .collect();
    black_box(data.zero_copy_par_iter());

    let before_constructor = allocation_snapshot();
    let iter = data.zero_copy_par_iter();
    let after_constructor = allocation_snapshot();
    let before_map = after_constructor;
    let mapped = iter.map(|value| value.wrapping_mul(3).wrapping_add(1));
    let after_map = allocation_snapshot();

    assert!(mapped.iter().enumerate().all(|(index, value)| {
        let input = (index as u64).wrapping_mul(17).wrapping_add(11);
        *value == input.wrapping_mul(3).wrapping_add(1)
    }));
    assert_eq!(
        (
            after_constructor.0.saturating_sub(before_constructor.0),
            after_constructor.1.saturating_sub(before_constructor.1),
            after_map.0.saturating_sub(before_map.0),
            after_map.1.saturating_sub(before_map.1),
        ),
        (0, 0, 1, ZERO_COPY_MAP_OUTPUT_BYTES)
    );
}

#[test]
#[ignore = "allocation attribution instrument; run explicitly with --nocapture"]
fn zero_copy_parallel_map_allocation_attribution() {
    assert!(process_lanes() > 1);
    let data = (0..ZERO_COPY_PARALLEL_MAP_LEN as u64)
        .map(|value| value.wrapping_mul(17).wrapping_add(11))
        .collect::<Vec<_>>();
    assert_eq!(
        zero_copy_parallel_map_values(&data).len(),
        ZERO_COPY_PARALLEL_MAP_LEN
    );

    let (before_allocations, before_bytes) = allocation_snapshot();
    let mapped = zero_copy_parallel_map_values(&data);
    let (after_allocations, after_bytes) = allocation_snapshot();
    let allocations = after_allocations.saturating_sub(before_allocations);
    let allocated_bytes = after_bytes.saturating_sub(before_bytes);

    assert!(mapped.iter().enumerate().all(|(index, value)| {
        let input = (index as u64).wrapping_mul(17).wrapping_add(11);
        *value == input.wrapping_mul(3).wrapping_add(1)
    }));
    assert_eq!(allocations, 2);
    assert_eq!(
        allocated_bytes,
        ZERO_COPY_PARALLEL_MAP_OUTPUT_BYTES + cache_map_completion_bytes()
    );
    eprintln!(
        "zero-copy parallel map: {allocations} allocations, {allocated_bytes} gross bytes, +         {ZERO_COPY_PARALLEL_MAP_OUTPUT_BYTES} output bytes"
    );
}

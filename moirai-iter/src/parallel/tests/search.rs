//! Searching and per-item terminals, including short-circuit and error paths.

use super::*;

#[test]
fn test_parallel_count() {
    let data = vec![1, 2, 3, 4, 5];
    let count = data.into_par_iter().count();
    assert_eq!(count, 5);
}

#[test]
fn test_parallel_any() {
    let data = vec![1, 2, 3, 4, 5];
    assert!(data.clone().into_par_iter().any(|x| *x == 3));
    assert!(!data.into_par_iter().any(|x| *x == 10));
}

#[test]
fn test_parallel_try_for_each_returns_ok_after_processing_all_items() {
    let data = vec![1_u64, 2, 3, 4];
    let total = std::sync::atomic::AtomicU64::new(0);
    let result = data.into_par_iter().try_for_each(|value| {
        total.fetch_add(value, std::sync::atomic::Ordering::Relaxed);
        Ok::<(), u64>(())
    });

    assert_eq!(result, Ok(()));
    assert_eq!(total.load(std::sync::atomic::Ordering::Relaxed), 10);
}

#[test]
fn test_parallel_try_for_each_returns_first_error() {
    let data = vec![1_u64, 2, 3, 4];
    let result = data
        .into_par_iter()
        .try_for_each(|value| if value == 3 { Err(value) } else { Ok(()) });

    assert_eq!(result, Err(3));
}

#[test]
fn test_parallel_find_last_returns_last_matching_value() {
    let data = vec![1_u64, 4, 7, 10, 13, 16];
    let result = data
        .clone()
        .into_par_iter()
        .find_last(|value| value % 3 == 1);
    assert_eq!(result, Some(16));

    let missing = data.into_par_iter().find_last(|value| *value > 100);
    assert_eq!(missing, None);
}

#[test]
fn test_parallel_find_map_first_maps_first_present_value() {
    let data = vec![1_u64, 4, 7, 10, 13];
    let result = data
        .clone()
        .into_par_iter()
        .find_map_first(|value| (value % 5 == 0).then_some(value.wrapping_mul(11)));
    assert_eq!(result, Some(110));

    let missing = data
        .into_par_iter()
        .find_map_first(|value| (value > 100).then_some(value));
    assert_eq!(missing, None);
}

#[test]
fn test_parallel_find_map_any_maps_present_value() {
    let data = vec![1_u64, 4, 7, 10, 13];
    let result = data
        .into_par_iter()
        .find_map_any(|value| (value == 7).then_some(value.wrapping_mul(13)));
    assert_eq!(result, Some(91));
}

#[test]
fn test_parallel_find_map_last_maps_last_present_value() {
    let data = vec![1_u64, 4, 7, 10, 13, 16];
    let result = data
        .clone()
        .into_par_iter()
        .find_map_last(|value| (value % 3 == 1).then_some(value.wrapping_mul(17)));
    assert_eq!(result, Some(272));

    let missing = data
        .into_par_iter()
        .find_map_last(|value| (value > 100).then_some(value));
    assert_eq!(missing, None);
}

#[test]
fn test_parallel_for_each_with_uses_cloned_state() {
    let data = vec![1_u64, 2, 3, 4];
    let checksum = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));

    data.into_par_iter()
        .map(|value| value.wrapping_mul(3))
        .for_each_with(std::sync::Arc::clone(&checksum), |state, value| {
            state.fetch_add(value, std::sync::atomic::Ordering::Relaxed);
        });

    assert_eq!(
        checksum.load(std::sync::atomic::Ordering::Relaxed),
        (1_u64 + 2 + 3 + 4) * 3
    );
}

#[test]
fn test_parallel_for_each_init_uses_initialized_state() {
    let data = vec![2_u64, 4, 6, 8];
    let checksum = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let sink = std::sync::Arc::clone(&checksum);

    data.into_par_iter()
        .map(|value| value.wrapping_add(1))
        .for_each_init(
            || std::sync::Arc::clone(&sink),
            |state, value| {
                state.fetch_add(value, std::sync::atomic::Ordering::Relaxed);
            },
        );

    assert_eq!(
        checksum.load(std::sync::atomic::Ordering::Relaxed),
        (2_u64 + 4 + 6 + 8) + 4
    );
}

#[test]
fn test_parallel_try_for_each_with_uses_cloned_state_and_propagates_error() {
    let data = vec![1_u64, 2, 3, 4];
    let checksum = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));

    let result = data.clone().into_par_iter().try_for_each_with(
        std::sync::Arc::clone(&checksum),
        |state, value| {
            state.fetch_add(value.wrapping_mul(5), std::sync::atomic::Ordering::Relaxed);
            Ok::<(), u64>(())
        },
    );
    assert_eq!(result, Ok(()));
    assert_eq!(
        checksum.load(std::sync::atomic::Ordering::Relaxed),
        (1_u64 + 2 + 3 + 4) * 5
    );

    let error =
        data.into_par_iter().try_for_each_with(
            (),
            |_state, value| {
                if value == 3 { Err(value) } else { Ok(()) }
            },
        );
    assert_eq!(error, Err(3));
}

#[test]
fn test_parallel_try_for_each_init_uses_initialized_state_and_propagates_error() {
    let data = vec![2_u64, 4, 6, 8];
    let checksum = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
    let sink = std::sync::Arc::clone(&checksum);

    let result = data.clone().into_par_iter().try_for_each_init(
        || std::sync::Arc::clone(&sink),
        |state, value| {
            state.fetch_add(value.wrapping_add(7), std::sync::atomic::Ordering::Relaxed);
            Ok::<(), u64>(())
        },
    );
    assert_eq!(result, Ok(()));
    assert_eq!(
        checksum.load(std::sync::atomic::Ordering::Relaxed),
        (2_u64 + 4 + 6 + 8) + (4 * 7)
    );

    let error = data.into_par_iter().try_for_each_init(
        || (),
        |_state, value| {
            if value == 6 { Err(value) } else { Ok(()) }
        },
    );
    assert_eq!(error, Err(6));
}

#[test]
fn test_parallel_all() {
    let data = vec![2, 4, 6, 8];
    assert!(data.clone().into_par_iter().all(|x| *x % 2 == 0));
    assert!(!data.into_par_iter().all(|x| *x > 5));
}

// ── Property-based parallel-search parity ──
//
// The example tests above pin fixed predicates; this generalizes the invariant
// that `positions` collects *every* matching logical index in *ascending order*
// — the sequential enumerate-filter-positions oracle — across the parallel
// shard boundaries, for arbitrary data and predicate. Order preservation is the
// error-prone part: shards run concurrently but their matches must merge back in
// index order with none dropped or duplicated.

#[test]
fn test_parallel_find_any_short_circuits_without_losing_the_only_match() {
    // One planted match one eighth in: the abort flag must not let a shard that
    // has not started discard the single answer.
    let mut data = vec![0_u64; ABOVE_DRIVE_THRESHOLD];
    data[ABOVE_DRIVE_THRESHOLD / 8] = 7;

    for _ in 0..64 {
        assert_eq!(
            data.clone().into_par_iter().find_any(|value| *value == 7),
            Some(7)
        );
        assert!(data.clone().into_par_iter().any(|value| *value == 7));
        assert!(!data.clone().into_par_iter().all(|value| *value == 0));
    }
}

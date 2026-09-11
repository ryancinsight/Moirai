//! Behaviour across the drive threshold: parallel results match the sequential oracle.

use super::*;

#[test]
fn parallel_drive_uses_multiple_lanes_and_preserves_order() {
    let data = (0..16_384usize).collect::<Vec<_>>();
    let worker_count = moirai_executor::global().total_workers(); // A single-worker configuration cannot prove cross-lane overlap; keep this
    // value-semantic suite portable while exercising the assertion on the normal
    // multi-worker executor.
    if worker_count < 2 {
        return;
    }

    let lanes = Arc::new(std::sync::Mutex::new(std::collections::HashSet::new()));
    let rendezvous = Arc::new((std::sync::Mutex::new(0usize), std::sync::Condvar::new()));
    let lane_sink = Arc::clone(&lanes);
    let rendezvous_for_map = Arc::clone(&rendezvous);

    let result: Vec<usize> = data
        .into_par_iter()
        .map(move |value| {
            let lane = std::thread::current().id();
            lane_sink
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .insert(lane);
            let (arrivals, signal) = &*rendezvous_for_map;
            let mut arrivals = arrivals
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            *arrivals += 1;
            signal.notify_all();
            if *arrivals < 2 {
                let (updated, timeout) = signal
                    .wait_timeout_while(arrivals, std::time::Duration::from_millis(100), |count| {
                        *count < 2
                    })
                    .unwrap_or_else(std::sync::PoisonError::into_inner);
                assert!(!timeout.timed_out(), "parallel branches did not rendezvous");
                drop(updated);
            }
            value.wrapping_mul(2)
        })
        .collect();

    assert_eq!(
        result,
        (0..16_384usize).map(|value| value * 2).collect::<Vec<_>>()
    );
    assert!(
        lanes
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .len()
            > 1,
        "large non-indexed drive must use more than one scheduler lane"
    );
    let filtered: Vec<usize> = (0..16_384usize)
        .collect::<Vec<_>>()
        .into_par_iter()
        .filter(|value| value % 2 == 0)
        .collect();
    assert_eq!(
        filtered,
        (0..16_384usize)
            .filter(|value| value % 2 == 0)
            .collect::<Vec<_>>()
    );
}

fn threshold_crossing_data() -> Vec<u64> {
    (0..ABOVE_DRIVE_THRESHOLD as u64)
        .map(|value| value.wrapping_mul(2_654_435_761) % 100_003)
        .collect()
}

#[test]
fn test_parallel_value_terminals_match_sequential_above_drive_threshold() {
    let data = threshold_crossing_data();

    assert_eq!(
        data.clone().into_par_iter().sum::<u64>(),
        data.iter().copied().sum::<u64>()
    );
    assert_eq!(
        data.clone().into_par_iter().map(|value| value % 7).count(),
        data.len()
    );
    assert_eq!(
        data.clone().into_par_iter().min(),
        data.iter().copied().min()
    );
    assert_eq!(
        data.clone().into_par_iter().max(),
        data.iter().copied().max()
    );
    assert_eq!(
        data.clone()
            .into_par_iter()
            .filter(|value| value % 3 == 0)
            .sum::<u64>(),
        data.iter()
            .copied()
            .filter(|value| value % 3 == 0)
            .sum::<u64>()
    );

    // Mostly ones so an 8192-term product stays inside u64: overflow-checked
    // debug arithmetic would abort before the terminal contract is observed.
    let product_data: Vec<u64> = data
        .iter()
        .enumerate()
        .map(|(index, _)| if index % 512 == 0 { 2 } else { 1 })
        .collect();
    assert_eq!(
        product_data.clone().into_par_iter().product::<u64>(),
        product_data.iter().copied().product::<u64>()
    );
}

#[test]
fn test_parallel_extremum_tie_breaking_matches_sequential_above_drive_threshold() {
    // Keys repeat many times across shard boundaries, so first-minimum and
    // last-maximum selection is decided by the merge tree rather than by a
    // unique extremum.
    let data: Vec<(u64, u64)> = (0..ABOVE_DRIVE_THRESHOLD as u64)
        .map(|index| (index, index % 16))
        .collect();

    assert_eq!(
        data.clone()
            .into_par_iter()
            .min_by(|left, right| left.1.cmp(&right.1)),
        data.iter()
            .copied()
            .min_by(|left, right| left.1.cmp(&right.1))
    );
    assert_eq!(
        data.clone()
            .into_par_iter()
            .max_by(|left, right| left.1.cmp(&right.1)),
        data.iter()
            .copied()
            .max_by(|left, right| left.1.cmp(&right.1))
    );
    assert_eq!(
        data.clone().into_par_iter().min_by_key(|pair| pair.1),
        data.iter().copied().min_by_key(|pair| pair.1)
    );
    assert_eq!(
        data.clone().into_par_iter().max_by_key(|pair| pair.1),
        data.iter().copied().max_by_key(|pair| pair.1)
    );
}

#[test]
fn test_parallel_search_terminals_match_sequential_above_drive_threshold() {
    let data = threshold_crossing_data();
    // Matches many positions spread across shard boundaries, so first, last and
    // any selection are genuinely distinguishable.
    let matches = |value: &u64| value % 1_000 == 42;

    assert_eq!(
        data.clone().into_par_iter().find_first(matches),
        data.iter().copied().find(matches)
    );
    assert_eq!(
        data.clone().into_par_iter().find_last(matches),
        data.iter().copied().rfind(matches)
    );
    assert_eq!(
        data.clone()
            .into_par_iter()
            .position_first(|value| value % 1_000 == 42),
        data.iter().position(|value| *value % 1_000 == 42)
    );
    assert_eq!(
        data.clone()
            .into_par_iter()
            .position_last(|value| value % 1_000 == 42),
        data.iter().rposition(|value| *value % 1_000 == 42)
    );
    assert_eq!(
        data.clone()
            .into_par_iter()
            .find_map_first(|value| (value % 1_000 == 42).then_some(value * 2)),
        data.iter()
            .copied()
            .find_map(|value| (value % 1_000 == 42).then_some(value * 2))
    );
    assert_eq!(
        data.clone()
            .into_par_iter()
            .find_map_last(|value| (value % 1_000 == 42).then_some(value * 2)),
        data.iter()
            .copied()
            .rev()
            .find_map(|value| (value % 1_000 == 42).then_some(value * 2))
    );

    // `find_any` may return any match, so pin existence rather than identity.
    let any = data.clone().into_par_iter().find_any(matches);
    assert!(any.is_some_and(|value| matches(&value)));
    assert!(data.clone().into_par_iter().any(matches));
    assert!(!data.clone().into_par_iter().any(|value| *value > 100_003));
    assert!(data.clone().into_par_iter().all(|value| *value < 100_003));
    assert!(!data.into_par_iter().all(matches));
}

#[test]
fn test_parallel_try_for_each_returns_first_error_above_drive_threshold() {
    let data: Vec<u64> = (0..ABOVE_DRIVE_THRESHOLD as u64).collect();
    // Two errors in different shards: the earlier one must win regardless of
    // which shard finishes first.
    let result = data.into_par_iter().try_for_each(|value| {
        if value == 3_000 || value == 6_000 {
            Err(value)
        } else {
            Ok(())
        }
    });

    assert_eq!(result, Err(3_000));
}

#[test]
fn test_parallel_partition_and_unzip_preserve_order_above_drive_threshold() {
    let data = threshold_crossing_data();

    let (accepted, rejected): (Vec<u64>, Vec<u64>) = data
        .clone()
        .into_par_iter()
        .partition(|value| value % 2 == 0);
    let (expected_accepted, expected_rejected): (Vec<u64>, Vec<u64>) =
        data.iter().copied().partition(|value| value % 2 == 0);
    assert_eq!(accepted, expected_accepted);
    assert_eq!(rejected, expected_rejected);

    let (left, right): (Vec<u64>, Vec<u64>) = data
        .clone()
        .into_par_iter()
        .map(|value| (value, value.wrapping_mul(3)))
        .unzip();
    let (expected_left, expected_right): (Vec<u64>, Vec<u64>) = data
        .iter()
        .copied()
        .map(|value| (value, value.wrapping_mul(3)))
        .unzip();
    assert_eq!(left, expected_left);
    assert_eq!(right, expected_right);

    let (evens, odds): (Vec<u64>, Vec<u64>) = data.into_par_iter().partition_map(|value| {
        if value % 2 == 0 {
            Either::Left(value)
        } else {
            Either::Right(value)
        }
    });
    assert_eq!(evens, expected_accepted);
    assert_eq!(odds, expected_rejected);
}

#[test]
fn test_reassociated_float_sum_is_reproducible_and_within_the_derived_bound() {
    // Floating-point addition is not associative, so the shard merge tree
    // re-associates the additions. Two properties hold and are checked here:
    // the merge tree depends only on the input length, so the value repeats
    // exactly across runs; and the divergence from a sequential sum stays
    // inside the error bound the two summations share.
    let data: Vec<f64> = (0..ABOVE_DRIVE_THRESHOLD)
        .map(|index| 1.0 / ((index + 1) as f64))
        .collect();

    let parallel = data.clone().into_par_iter().sum_reassociated::<f64>();
    for _ in 0..16 {
        assert_eq!(
            data.clone().into_par_iter().sum_reassociated::<f64>(),
            parallel
        );
    }

    let sequential: f64 = data.iter().copied().sum();
    // For round-to-nearest arithmetic, gamma(k) = k*u/(1-k*u) bounds the
    // accumulated relative error across k additions. The sequential result
    // has n-1 additions. Above the dispatch threshold, VecParIter performs one
    // split; drive_split folds each half sequentially and merges the two
    // outputs once. The difference is bounded by the sum of those two
    // forward-error bounds. Inflate the rounded magnitude to an upper bound on
    // the exact positive sum before applying them.
    let (leaf_width, merge_depth) = if data.len() <= super::sources::PARALLEL_DRIVE_THRESHOLD {
        (data.len(), 0)
    } else {
        (data.len().div_ceil(2), 1)
    };
    let sequential_additions = data.len().saturating_sub(1);
    let leaf_additions = leaf_width.saturating_sub(1);
    let reassociated_additions = leaf_additions + merge_depth;
    let unit_roundoff = f64::EPSILON / 2.0;
    let gamma = |additions: usize| {
        let scaled = (additions as f64) * unit_roundoff;
        scaled / (1.0 - scaled)
    };
    let sequential_gamma = gamma(sequential_additions);
    let reassociated_gamma = gamma(reassociated_additions);
    let rounded_magnitude: f64 = data.iter().map(|value| value.abs()).sum();
    let magnitude_upper = rounded_magnitude / (1.0 - sequential_gamma);
    let bound = (sequential_gamma + reassociated_gamma) * magnitude_upper;
    assert!(
        (parallel - sequential).abs() <= bound,
        "parallel {parallel} and sequential {sequential} differ by more than the derived bound {bound}"
    );
}

proptest::proptest! {
    #[test]
    fn prop_parallel_positions_match_sequential_filter(
        data in proptest::collection::vec(proptest::prelude::any::<u64>(), 0..600),
        divisor in 1u64..16,
        remainder in 0u64..16,
    ) {
        let r = remainder % divisor;
        let par: Vec<usize> = data.par_iter().positions(|value| *value % divisor == r).collect();
        let seq: Vec<usize> = data
            .iter()
            .enumerate()
            .filter(|(_, value)| **value % divisor == r)
            .map(|(index, _)| index)
            .collect();
        proptest::prop_assert_eq!(par, seq);
    }

    /// `find_map_first` returns the mapped value of the *first* element (in
    /// iteration order) for which the closure yields `Some`, matching the
    /// sequential `Iterator::find_map` — even though shards search concurrently,
    /// the lowest-index match must win (or `None` when nothing matches).
    #[test]
    fn prop_find_map_first_matches_sequential(
        data in proptest::collection::vec(proptest::prelude::any::<u64>(), 0..600),
        divisor in 1u64..16,
        remainder in 0u64..16,
    ) {
        let r = remainder % divisor;
        let par = data
            .clone()
            .into_par_iter()
            .find_map_first(|value| (value % divisor == r).then_some(value.wrapping_mul(11)));
        let seq = data
            .iter()
            .find_map(|&value| (value % divisor == r).then_some(value.wrapping_mul(11)));
        proptest::prop_assert_eq!(par, seq);
    }

    /// `reduce` (no identity) folds with an associative+commutative op and equals
    /// the sequential `Iterator::reduce` for any input — `None` on empty, the
    /// combined value otherwise. Distinct code path from the identity-seeded
    /// reduce: the parallel combine of per-shard partials must not diverge.
    #[test]
    fn prop_reduce_no_identity_matches_sequential(
        data in proptest::collection::vec(proptest::prelude::any::<u64>(), 0..600),
    ) {
        let par = data.clone().into_par_iter().reduce(|a, b| a.wrapping_add(b));
        let seq = data.into_iter().reduce(|a, b| a.wrapping_add(b));
        proptest::prop_assert_eq!(par, seq);
    }

    /// Parallel `min`/`max` equal the sequential extrema for any input (`None`
    /// on empty). The per-shard partial extrema must combine to the global one.
    #[test]
    fn prop_min_max_match_sequential(
        data in proptest::collection::vec(proptest::prelude::any::<u64>(), 0..600),
    ) {
        let par_min = data.clone().into_par_iter().min();
        let par_max = data.clone().into_par_iter().max();
        proptest::prop_assert_eq!(par_min, data.iter().copied().min());
        proptest::prop_assert_eq!(par_max, data.iter().copied().max());
    }

    /// Parallel `all`/`any` equal the sequential short-circuiting predicates for
    /// any input and predicate — every shard's verdict must fold to the global
    /// one (all => conjunction, any => disjunction), including the empty case.
    #[test]
    fn prop_all_any_match_sequential(
        data in proptest::collection::vec(proptest::prelude::any::<u64>(), 0..600),
        divisor in 1u64..16,
        remainder in 0u64..16,
    ) {
        let r = remainder % divisor;
        let par_all = data.clone().into_par_iter().all(|value| *value % divisor == r);
        let par_any = data.clone().into_par_iter().any(|value| *value % divisor == r);
        proptest::prop_assert_eq!(par_all, data.iter().all(|value| value % divisor == r));
        proptest::prop_assert_eq!(par_any, data.iter().any(|value| value % divisor == r));
    }

    /// `find_map_last` returns the mapped *last* (highest-index) match in
    /// iteration order — the reverse-search dual of find_map_first — matching
    /// the sequential reverse find_map, with shards searching concurrently.
    #[test]
    fn prop_find_map_last_matches_sequential(
        data in proptest::collection::vec(proptest::prelude::any::<u64>(), 0..600),
        divisor in 1u64..16,
        remainder in 0u64..16,
    ) {
        let r = remainder % divisor;
        let par = data
            .clone()
            .into_par_iter()
            .find_map_last(|value| (value % divisor == r).then_some(value.wrapping_mul(17)));
        let seq = data
            .iter()
            .rev()
            .find_map(|&value| (value % divisor == r).then_some(value.wrapping_mul(17)));
        proptest::prop_assert_eq!(par, seq);
    }

    /// `min_by_key`/`max_by_key` select the element with the extremal key under a
    /// custom key function — the comparator path, distinct from `min`/`max`. The
    /// extremal *key* is unique even when several elements share it, so comparing
    /// result keys is tie-break-agnostic between the parallel and sequential
    /// choice of which equal-keyed element is returned.
    #[test]
    fn prop_min_max_by_key_match_sequential(
        data in proptest::collection::vec(proptest::prelude::any::<u64>(), 0..600),
    ) {
        let key = |value: &u64| value.rotate_left(7) ^ value.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let par_min = data.clone().into_par_iter().min_by_key(key);
        let par_max = data.clone().into_par_iter().max_by_key(key);
        let seq_min = data.iter().copied().min_by_key(key);
        let seq_max = data.iter().copied().max_by_key(key);
        proptest::prop_assert_eq!(par_min.map(|v| key(&v)), seq_min.map(|v| key(&v)));
        proptest::prop_assert_eq!(par_max.map(|v| key(&v)), seq_max.map(|v| key(&v)));
    }

    /// `find_map_any` may return *any* shard's match (order-unspecified), so the
    /// contract is consistency, not identity: it yields `Some` exactly when a
    /// match exists, and any value it yields is a genuine mapped match of some
    /// element actually present in the input.
    #[test]
    fn prop_find_map_any_is_a_valid_match(
        data in proptest::collection::vec(proptest::prelude::any::<u64>(), 0..600),
        divisor in 1u64..16,
        remainder in 0u64..16,
    ) {
        let r = remainder % divisor;
        let par = data
            .clone()
            .into_par_iter()
            .find_map_any(|value| (value % divisor == r).then_some(value.wrapping_mul(11)));
        let any_match = data.iter().any(|value| value % divisor == r);
        proptest::prop_assert_eq!(par.is_some(), any_match);
        if let Some(mapped) = par {
            let valid = data
                .iter()
                .any(|&value| value % divisor == r && value.wrapping_mul(11) == mapped);
            proptest::prop_assert!(valid);
        }
    }

    /// `reduce_with` carries the same associative-combine contract as `reduce`
    /// and must equal the sequential `Iterator::reduce` for any input (`None` on
    /// empty). The two methods now drive one shared `ReduceConsumer`; this pins
    /// the public `reduce_with` surface to an independent sequential oracle so a
    /// future re-divergence of the two terminals cannot silently regress its
    /// value semantics.
    #[test]
    fn prop_reduce_with_matches_sequential(
        data in proptest::collection::vec(proptest::prelude::any::<u64>(), 0..600),
    ) {
        let par = data.clone().into_par_iter().reduce_with(|a, b| a.wrapping_add(b));
        let seq = data.into_iter().reduce(|a, b| a.wrapping_add(b));
        proptest::prop_assert_eq!(par, seq);
    }
}

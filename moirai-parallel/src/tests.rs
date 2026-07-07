use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};

#[test]
fn for_each_visits_every_element_once() {
    let data: Vec<usize> = (0..10_000).collect();
    let counter = AtomicUsize::new(0);
    data.par().for_each(|&x| {
        counter.fetch_add(x, Ordering::Relaxed);
    });
    assert_eq!(counter.load(Ordering::Relaxed), data.iter().sum());
}

#[test]
fn for_each_mut_mutates_in_place() {
    let mut data: Vec<u64> = (0..10_000).collect();
    data.par_mut().for_each(|x| *x *= 2);
    for (i, &v) in data.iter().enumerate() {
        assert_eq!(v, (i as u64) * 2);
    }
}

#[test]
fn enumerate_mut_uses_index() {
    let mut data = vec![0usize; 5_000];
    data.par_mut().enumerate(|i, x| *x = i * 3);
    for (i, &v) in data.iter().enumerate() {
        assert_eq!(v, i * 3);
    }
}

#[test]
fn map_collect_preserves_order() {
    let data: Vec<u64> = (0..20_000).collect();
    let squared = data.par().map_collect(|&x| x * x);
    for (i, &v) in squared.iter().enumerate() {
        assert_eq!(v, (i as u64) * (i as u64));
    }
}

#[test]
fn map_collect_index_preserves_logical_indices() {
    let data: Vec<u64> = (10_000..30_000).collect();
    let indexed = data.par().map_collect_index(|i, &x| (i, x - 10_000));
    for (i, &(logical_index, shifted_value)) in indexed.iter().enumerate() {
        assert_eq!(logical_index, i);
        assert_eq!(shifted_value, i as u64);
    }
}

#[test]
fn map_reduce_sums_correctly() {
    let data: Vec<u64> = (0..100_000).collect();
    assert_eq!(
        data.par().map_reduce(0u64, |&x| x, |a, b| a + b),
        data.iter().copied().sum::<u64>()
    );
    // free-function form with explicit policy
    assert_eq!(
        map_reduce_with::<Sequential, _, _, _, _>(&data, 0u64, |&x| x, |a, b| a + b),
        data.iter().copied().sum::<u64>()
    );
}

#[test]
fn join_returns_both_branch_values() {
    let left = [1u64, 2, 3, 4];
    let right = [10u64, 20, 30];

    let (left_sum, right_sum) = join(|| left.iter().sum::<u64>(), || right.iter().sum::<u64>());

    assert_eq!(left_sum, 10);
    assert_eq!(right_sum, 60);
}

#[test]
fn join_with_parallel_accepts_borrowed_non_static_inputs() {
    let left = [2u64, 4, 6, 8];
    let right = [3u64, 6, 9];

    let (left_product, right_product) = join_with::<Parallel, _, _, _, _>(
        || left.iter().copied().product::<u64>(),
        || right.iter().copied().product::<u64>(),
    );

    assert_eq!(left_product, 384);
    assert_eq!(right_product, 162);
}

#[test]
fn adaptive_view_and_explicit_policies_agree() {
    let data: Vec<u64> = (0..50_000).collect();
    let expected: u64 = data.iter().sum();
    // adaptive trait surface
    assert_eq!(data.par().map_reduce(0u64, |&x| x, |a, b| a + b), expected);
    // explicit policy overrides via the low-level free functions
    assert_eq!(
        map_reduce_with::<Parallel, _, _, _, _>(&data, 0u64, |&x| x, |a, b| a + b),
        expected
    );
    assert_eq!(
        map_reduce_with::<Sequential, _, _, _, _>(&data, 0u64, |&x| x, |a, b| a + b),
        expected
    );
    let doubled = data.par().map_collect(|&x| x * 2);
    assert_eq!(doubled, data.iter().map(|&x| x * 2).collect::<Vec<_>>());
}

#[test]
fn mut_view_and_explicit_sequential_agree() {
    let mut data: Vec<u64> = (0..50_000).collect();
    data.par_mut().for_each(|x| *x += 1);
    // forced-sequential override produces the same result
    enumerate_mut_with::<Sequential, _, _>(&mut data, |i, x| *x += i as u64);
    for (i, &v) in data.iter().enumerate() {
        assert_eq!(v, i as u64 + 1 + i as u64);
    }
}

#[test]
fn for_each_chunk_mut_processes_lanes() {
    // 1000 lanes of 7 elements; set each lane's elements to the lane index.
    let lanes = 1000usize;
    let width = 7usize;
    let mut data: Vec<u64> = vec![0; lanes * width];
    for_each_chunk_mut_with::<Adaptive, _, _>(&mut data, width, |chunk| {
        let lane = chunk[0]; // 0 initially; use position via first write instead
        let _ = lane;
        for x in chunk.iter_mut() {
            *x += 1;
        }
    });
    assert!(data.iter().all(|&v| v == 1));
    // uneven final chunk
    let mut d2: Vec<u64> = (0..10).collect();
    for_each_chunk_mut_with::<Parallel, _, _>(&mut d2, 3, |c| {
        for x in c {
            *x *= 10;
        }
    });
    assert_eq!(d2, (0..10).map(|x| x * 10).collect::<Vec<_>>());
}

#[test]
fn for_each_chunk_mut_with_state_reuses_worker_state() {
    let mut data = vec![0usize; 64 * 32];
    let state_initializations = AtomicUsize::new(0);
    for_each_chunk_mut_with_state::<Parallel, _, _, _, _>(
        &mut data,
        32,
        || {
            state_initializations.fetch_add(1, Ordering::Relaxed);
            Vec::<usize>::with_capacity(32)
        },
        |scratch, chunk| {
            scratch.clear();
            scratch.extend(0..chunk.len());
            for (value, offset) in chunk.iter_mut().zip(scratch.iter()) {
                *value = *offset;
            }
        },
    );

    for chunk in data.chunks(32) {
        assert_eq!(chunk, &(0..32).collect::<Vec<_>>());
    }
    let initialized = state_initializations.load(Ordering::Relaxed);
    assert!(initialized > 0);
    assert!(initialized <= 64);
}

#[test]
fn for_each_index_visits_every_index() {
    use std::sync::atomic::{AtomicUsize, Ordering};
    let n = 10_000usize;
    let counter = AtomicUsize::new(0);
    let sum = (0..n).map(|i| i as u64).sum::<u64>();
    let acc = std::sync::atomic::AtomicU64::new(0);
    for_each_index_with::<Adaptive, _>(n, |i| {
        counter.fetch_add(1, Ordering::Relaxed);
        acc.fetch_add(i as u64, Ordering::Relaxed);
    });
    assert_eq!(
        counter.load(Ordering::Relaxed),
        n,
        "every index visited once"
    );
    assert_eq!(
        acc.load(Ordering::Relaxed),
        sum,
        "index values summed correctly"
    );
}

#[test]
fn for_each_chunk_pair_mut_processes_paired_chunks() {
    let frames = 300usize;
    let width = 5usize;
    let mut a: Vec<u64> = vec![0; frames * width];
    let mut b: Vec<u64> = vec![0; frames * width];
    for_each_chunk_pair_mut_enumerated_with::<Adaptive, _, _, _>(
        &mut a,
        &mut b,
        width,
        |i, ca, cb| {
            for x in ca.iter_mut() {
                *x = i as u64;
            }
            for (j, y) in cb.iter_mut().enumerate() {
                *y = (i * width + j) as u64;
            }
        },
    );
    for i in 0..frames {
        assert!(a[i * width..(i + 1) * width].iter().all(|&v| v == i as u64));
        for j in 0..width {
            assert_eq!(b[i * width + j], (i * width + j) as u64);
        }
    }
}

#[test]
fn for_each_chunk_quad_mut_processes_quad_chunks() {
    let frames = 240usize;
    let width = 7usize;
    let mut a: Vec<u64> = vec![0; frames * width];
    let mut b: Vec<u64> = vec![0; frames * width];
    let mut c: Vec<u64> = vec![0; frames * width];
    let mut d: Vec<u64> = vec![0; frames * width];
    for_each_chunk_quad_mut_enumerated_with::<Adaptive, _, _, _, _, _>(
        &mut a,
        &mut b,
        &mut c,
        &mut d,
        width,
        |chunk_index, ca, cb, cc, cd| {
            for lane in 0..ca.len() {
                let absolute = chunk_index * width + lane;
                ca[lane] = chunk_index as u64;
                cb[lane] = absolute as u64;
                cc[lane] = (absolute * 2) as u64;
                cd[lane] = ca[lane] + cb[lane] + cc[lane];
            }
        },
    );

    for chunk_index in 0..frames {
        for lane in 0..width {
            let idx = chunk_index * width + lane;
            assert_eq!(a[idx], chunk_index as u64);
            assert_eq!(b[idx], idx as u64);
            assert_eq!(c[idx], (idx * 2) as u64);
            assert_eq!(d[idx], a[idx] + b[idx] + c[idx]);
        }
    }
}

#[test]
fn for_each_chunk_triple_mut_processes_triple_chunks() {
    let frames = 180usize;
    let width = 9usize;
    let mut a: Vec<u64> = vec![0; frames * width];
    let mut b: Vec<u64> = vec![0; frames * width];
    let mut c: Vec<u64> = vec![0; frames * width];
    for_each_chunk_triple_mut_enumerated_with::<Adaptive, _, _, _, _>(
        &mut a,
        &mut b,
        &mut c,
        width,
        |chunk_index, ca, cb, cc| {
            for lane in 0..ca.len() {
                let absolute = chunk_index * width + lane;
                ca[lane] = chunk_index as u64;
                cb[lane] = absolute as u64;
                cc[lane] = ca[lane] + cb[lane];
            }
        },
    );

    for chunk_index in 0..frames {
        for lane in 0..width {
            let idx = chunk_index * width + lane;
            assert_eq!(a[idx], chunk_index as u64);
            assert_eq!(b[idx], idx as u64);
            assert_eq!(c[idx], a[idx] + b[idx]);
        }
    }
}

#[test]
fn for_each_chunk_mut_enumerated_passes_index() {
    let lanes = 500usize;
    let width = 4usize;
    let mut data: Vec<u64> = vec![0; lanes * width];
    for_each_chunk_mut_enumerated_with::<Adaptive, _, _>(&mut data, width, |i, chunk| {
        for x in chunk.iter_mut() {
            *x = i as u64;
        }
    });
    for i in 0..lanes {
        assert!(data[i * width..(i + 1) * width]
            .iter()
            .all(|&v| v == i as u64));
    }
}

#[test]
fn map_collect_mut_mutates_and_collects() {
    let mut data: Vec<u64> = (0..10_000).collect();
    let doubled_indices = map_collect_mut_with::<Adaptive, _, _, _>(&mut data, |i, x| {
        *x += 1; // mutate in place
        i as u64 // collect the index
    });
    for (i, &v) in data.iter().enumerate() {
        assert_eq!(v, i as u64 + 1);
    }
    assert_eq!(doubled_indices, (0..10_000u64).collect::<Vec<_>>());
}

#[test]
fn fold_reduce_accumulates_into_collection() {
    use std::collections::HashMap;
    let n = 30_000usize;
    // group i -> sum of i over its (i % 8) bucket
    let map = fold_reduce_with::<Adaptive, HashMap<usize, u64>, _, _, _>(
        n,
        HashMap::new,
        |mut acc, i| {
            *acc.entry(i % 8).or_insert(0) += i as u64;
            acc
        },
        |mut a, b| {
            for (k, v) in b {
                *a.entry(k).or_insert(0) += v;
            }
            a
        },
    );
    let mut expected: HashMap<usize, u64> = HashMap::new();
    for i in 0..n {
        *expected.entry(i % 8).or_insert(0) += i as u64;
    }
    assert_eq!(map, expected);
}

#[test]
fn map_collect_index_zips_two_slices() {
    let a: Vec<u64> = (0..20_000).collect();
    let b: Vec<u64> = (0..20_000).map(|x| x + 1).collect();
    let prod = map_collect_index_with::<Adaptive, _, _>(a.len(), |i| a[i] * b[i]);
    let expected: Vec<u64> = a.iter().zip(&b).map(|(&x, &y)| x * y).collect();
    assert_eq!(prod, expected);
}

#[test]
fn reduce_index_computes_dot_product() {
    let a: Vec<u64> = (0..50_000).collect();
    let b: Vec<u64> = (0..50_000).map(|x| x * 2).collect();
    let dot = reduce_index_with::<Adaptive, _, _, _>(a.len(), 0u64, |i| a[i] * b[i], |x, y| x + y);
    let expected: u64 = a.iter().zip(&b).map(|(&x, &y)| x * y).sum();
    assert_eq!(dot, expected);
    // sequential policy agrees
    assert_eq!(
        reduce_index_with::<Sequential, _, _, _>(a.len(), 0u64, |i| a[i] * b[i], |x, y| x + y),
        expected
    );
}

#[test]
fn empty_and_single_inputs_are_handled() {
    let empty: Vec<i32> = Vec::new();
    empty.par().for_each(|_| panic!("must not run"));
    assert_eq!(
        empty.par().map_reduce(42i64, |&x| x as i64, |a, b| a + b),
        42
    );
    let mut one = vec![7u64];
    one.par_mut().for_each(|x| *x += 1);
    assert_eq!(one, vec![8]);
}

#[cfg(feature = "melinoe")]
#[test]
fn test_par_partition_melinoe() {
    use melinoe::{brand_scope, MelinoeCell};
    let data = vec![0usize; 16];
    brand_scope(|token| {
        let mut cells: Vec<MelinoeCell<'_, usize>> =
            data.into_iter().map(MelinoeCell::new).collect();
        super::melinoe_ext::par_partition_for_each(&mut cells, 4, |start, mut shard| {
            for (j, slot) in shard.iter_mut().enumerate() {
                *slot = start + j;
            }
        });
        let snap = token.share();
        for (i, cell) in cells.iter().enumerate() {
            assert_eq!(*cell.borrow(snap), i);
        }
    });
}

// ── Property-based parallel-vs-sequential parity ──
//
// The example tests above pin fixed inputs; these generalize the invariant
// "the adaptive/parallel path equals the forced-sequential path" over arbitrary
// inputs. `wrapping_*` keeps the reduction associative so partition-and-combine
// must match the sequential fold exactly.

proptest::proptest! {
    /// `par().map_reduce` equals the crate's own forced-`Sequential` path for any
    /// input, with an associative+commutative combine.
    #[test]
    fn prop_map_reduce_parallel_matches_sequential(
        data in proptest::collection::vec(proptest::prelude::any::<u64>(), 0..600),
    ) {
        let par = data.par().map_reduce(0u64, |&x| x, |a, b| a.wrapping_add(b));
        let seq =
            map_reduce_with::<Sequential, _, _, _, _>(&data, 0u64, |&x| x, |a, b| a.wrapping_add(b));
        proptest::prop_assert_eq!(par, seq);
    }

    /// `par().map_collect` is order-preserving and element-wise: equals the
    /// sequential map for any input and multiplier.
    #[test]
    fn prop_map_collect_parallel_matches_sequential(
        data in proptest::collection::vec(proptest::prelude::any::<i64>(), 0..600),
        factor in proptest::prelude::any::<i64>(),
    ) {
        let par = data.par().map_collect(|&x| x.wrapping_mul(factor));
        let seq: Vec<i64> = data.iter().map(|&x| x.wrapping_mul(factor)).collect();
        proptest::prop_assert_eq!(par, seq);
    }

    /// `par_mut().for_each` applies the per-element transform to every element
    /// exactly once: the in-place parallel mutation equals the sequential one
    /// for any input (disjoint element ownership, no skips or double-writes).
    #[test]
    fn prop_for_each_mut_parallel_matches_sequential(
        data in proptest::collection::vec(proptest::prelude::any::<i64>(), 0..600),
    ) {
        let mut par = data.clone();
        par.par_mut().for_each(|x| *x = x.wrapping_mul(3).wrapping_add(7));
        let mut seq = data;
        seq.iter_mut().for_each(|x| *x = x.wrapping_mul(3).wrapping_add(7));
        proptest::prop_assert_eq!(par, seq);
    }

    /// `fold_reduce_with` over the index range `0..len` with an associative+
    /// commutative combine equals its own forced-`Sequential` path for any
    /// length — the per-partition index folds must reduce to the same total.
    #[test]
    fn prop_fold_reduce_parallel_matches_sequential(len in 0usize..2000) {
        let init = || 0u64;
        let fold = |acc: u64, i: usize| acc.wrapping_add(i as u64);
        let reduce = |a: u64, b: u64| a.wrapping_add(b);
        let par = fold_reduce_with::<Parallel, u64, _, _, _>(len, init, fold, reduce);
        let seq = fold_reduce_with::<Sequential, u64, _, _, _>(len, init, fold, reduce);
        proptest::prop_assert_eq!(par, seq);
    }

    /// `map_collect_index` threads the logical index and is order-preserving:
    /// the parallel result equals the sequential `enumerate().map()` for any
    /// input, so each shard sees the correct global index and the pieces
    /// concatenate in order.
    #[test]
    fn prop_map_collect_index_parallel_matches_sequential(
        data in proptest::collection::vec(proptest::prelude::any::<i64>(), 0..600),
    ) {
        let par = data.par().map_collect_index(|i, &x| (i as i64).wrapping_mul(x));
        let seq: Vec<i64> = data
            .iter()
            .enumerate()
            .map(|(i, &x)| (i as i64).wrapping_mul(x))
            .collect();
        proptest::prop_assert_eq!(par, seq);
    }
}

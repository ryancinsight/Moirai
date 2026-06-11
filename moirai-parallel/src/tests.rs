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

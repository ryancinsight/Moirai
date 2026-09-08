//! Element-wise adapters: mapping, filtering, flattening, and panic containment.

use super::*;

#[test]
fn test_parallel_map() {
    let data = vec![1, 2, 3, 4, 5];
    let result: Vec<i32> = data.into_par_iter().map(|x| x * 2).collect();
    assert_eq!(result, vec![2, 4, 6, 8, 10]);
}

#[test]
fn test_parallel_map_with_uses_cloned_state() {
    let data = vec![1_u64, 2, 3, 4];
    let result: Vec<u64> = data
        .into_par_iter()
        .map_with(10_u64, |state, value| {
            *state = state.wrapping_add(1);
            value.wrapping_mul(*state)
        })
        .collect();

    assert_eq!(result, vec![11, 24, 39, 56]);
}

#[test]
fn test_parallel_map_init_uses_initialized_state() {
    let data = vec![2_u64, 4, 6];
    let result: Vec<u64> = data
        .into_par_iter()
        .map_init(
            || 3_u64,
            |state, value| {
                let output = value.wrapping_add(*state);
                *state = state.wrapping_add(2);
                output
            },
        )
        .collect();

    assert_eq!(result, vec![5, 9, 13]);
}

#[test]
fn test_parallel_update_mutates_items_before_yielding() {
    let data = vec![1_u64, 2, 3, 4];
    let result: Vec<u64> = data
        .into_par_iter()
        .update(|value| {
            *value = value.wrapping_mul(3).wrapping_add(1);
        })
        .collect();

    assert_eq!(result, vec![4, 7, 10, 13]);
}

#[test]
fn test_parallel_filter() {
    let data = vec![1, 2, 3, 4, 5, 6];
    let result: Vec<i32> = data.into_par_iter().filter(|&x| x % 2 == 0).collect();
    assert_eq!(result, vec![2, 4, 6]);
}

#[test]
fn test_parallel_collect_vec_list_moves_non_clone_values() {
    struct NonCloneValue {
        value: u64,
    }

    let data = vec![
        NonCloneValue { value: 1 },
        NonCloneValue { value: 2 },
        NonCloneValue { value: 3 },
    ];

    let list = data.into_par_iter().collect_vec_list();
    let flattened: Vec<u64> = list.into_iter().flatten().map(|item| item.value).collect();

    assert_eq!(flattened, vec![1, 2, 3]);

    let empty = Vec::<NonCloneValue>::new()
        .into_par_iter()
        .collect_vec_list();
    assert!(empty.is_empty());
}

#[test]
fn test_parallel_inspect_observes_items_without_changing_output() {
    let data = vec![1, 2, 3, 4];
    let observed = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
    let sink = std::sync::Arc::clone(&observed);

    let result: Vec<i32> = data
        .clone()
        .into_par_iter()
        .inspect(move |value| sink.lock().expect("inspection lock").push(*value))
        .collect();

    assert_eq!(result, data);
    assert_eq!(*observed.lock().expect("inspection lock"), vec![1, 2, 3, 4]);
}

#[test]
fn test_parallel_panic_fuse_preserves_values() {
    let data = vec![1, 2, 3];
    let result: Vec<i32> = data
        .into_par_iter()
        .panic_fuse()
        .map(|value| value * 2)
        .collect();
    assert_eq!(result, vec![2, 4, 6]);
}

#[test]
#[should_panic(expected = "panic-fuse propagation")]
fn test_parallel_panic_fuse_propagates_panic() {
    let data = vec![1, 2, 3];
    let _: Vec<i32> = data
        .into_par_iter()
        .panic_fuse()
        .map(|value| {
            if value == 2 {
                panic!("panic-fuse propagation");
            }
            value
        })
        .collect();
}

#[test]
fn test_parallel_filter_map_retains_present_values() {
    let data = vec![1, 2, 3, 4, 5, 6];
    let result: Vec<i32> = data
        .into_par_iter()
        .filter_map(|value| (value % 2 == 0).then_some(value * 10))
        .collect();
    assert_eq!(result, vec![20, 40, 60]);
}

#[test]
fn test_parallel_while_some_unwraps_present_prefix() {
    let data = vec![Some(1_u64), Some(2), Some(3), None, Some(5)];
    let result: Vec<_> = data.into_par_iter().while_some().collect();
    assert_eq!(result, vec![1, 2, 3]);
}

#[test]
fn test_parallel_while_some_empty_when_first_is_none() {
    let data = vec![None, Some(2_u64), Some(3)];
    let result: Vec<_> = data.into_par_iter().while_some().collect();
    assert!(result.is_empty());
}

#[test]
fn test_parallel_flat_map_preserves_flattened_order() {
    let data = vec![1, 2, 3];
    let result: Vec<i32> = data.into_par_iter().flat_map(|value| 0..value).collect();
    assert_eq!(result, vec![0, 0, 1, 0, 1, 2]);
}

#[test]
fn test_parallel_flat_map_iter_accepts_serial_inner_iterators() {
    let data = vec![1_usize, 2, 3];
    let result: Vec<usize> = data
        .into_par_iter()
        .flat_map_iter(|limit| {
            let inner = RefCell::new(0..limit);
            std::iter::from_fn(move || inner.borrow_mut().next())
        })
        .collect();
    assert_eq!(result, vec![0, 0, 1, 0, 1, 2]);
}

#[test]
fn test_parallel_flatten_preserves_nested_order() {
    let data = vec![vec![1, 2], Vec::new(), vec![3, 4, 5]];
    let result: Vec<i32> = data.into_par_iter().flatten().collect();
    assert_eq!(result, vec![1, 2, 3, 4, 5]);
}

#[test]
fn test_parallel_flatten_iter_preserves_serial_inner_order() {
    let data = vec![0_usize..2, 2..2, 2..5];
    let result: Vec<usize> = data.into_par_iter().flatten_iter().collect();
    assert_eq!(result, vec![0, 1, 2, 3, 4]);
}

#[test]
fn test_parallel_copied_materializes_borrowed_copy_values() {
    let data = vec![1_u64, 2, 3, 4];
    let result: Vec<u64> = data.par_iter().copied().map(|value| value * 3).collect();
    assert_eq!(result, vec![3, 6, 9, 12]);
}

#[test]
fn test_parallel_cloned_materializes_borrowed_clone_values() {
    let data = vec!["alpha".to_owned(), "beta".to_owned(), "gamma".to_owned()];
    let result: Vec<String> = data
        .par_iter()
        .cloned()
        .filter(|value| value.contains('a'))
        .collect();
    assert_eq!(
        result,
        vec!["alpha".to_owned(), "beta".to_owned(), "gamma".to_owned()]
    );
}

#[test]
fn test_non_clone_parallel_ref_iterator_maps_borrowed_values() {
    struct NonCloneBorrowed {
        value: u64,
    }

    let data = vec![
        NonCloneBorrowed { value: 2 },
        NonCloneBorrowed { value: 3 },
        NonCloneBorrowed { value: 5 },
    ];

    let result = data
        .par_iter()
        .map(|item| item.value.wrapping_mul(7))
        .collect::<Vec<_>>();

    assert_eq!(result, vec![14, 21, 35]);
}

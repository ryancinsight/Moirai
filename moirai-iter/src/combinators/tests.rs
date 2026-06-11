//! Unit tests for iterator combinators.

extern crate alloc;
use super::ext::CombinatorExt;
use alloc::vec;
use alloc::vec::Vec;

#[test]
fn test_scan() {
    let numbers = vec![1, 2, 3, 4, 5];
    let sums: Vec<_> = CombinatorExt::scan(numbers.into_iter(), 0, |state, x| {
        *state += x;
        Some(*state)
    })
    .collect();
    assert_eq!(sums, vec![1, 3, 6, 10, 15]);
}

#[test]
fn test_flat_map() {
    let data = vec![vec![1, 2], vec![3, 4], vec![5]];
    let flattened: Vec<_> =
        CombinatorExt::flat_map(data.into_iter(), |v| v.into_iter()).collect();
    assert_eq!(flattened, vec![1, 2, 3, 4, 5]);
}

#[test]
fn test_inspect() {
    let mut inspected = Vec::new();
    let data: Vec<_> = CombinatorExt::inspect(1..=5, |x| inspected.push(*x))
        .map(|x| x * 2)
        .collect();
    assert_eq!(data, vec![2, 4, 6, 8, 10]);
    assert_eq!(inspected, vec![1, 2, 3, 4, 5]);
}

#[test]
fn test_peekable() {
    let mut iter = CombinatorExt::peekable(vec![1, 2, 3].into_iter());
    assert_eq!(iter.peek(), Some(&1));
    assert_eq!(iter.peek(), Some(&1));
    assert_eq!(iter.next(), Some(1));
    assert_eq!(iter.peek(), Some(&2));
}

#[test]
fn test_skip() {
    let data: Vec<_> = CombinatorExt::skip(1..=10, 5).collect();
    assert_eq!(data, vec![6, 7, 8, 9, 10]);
}

#[test]
fn test_skip_while() {
    let data: Vec<_> = CombinatorExt::skip_while(1..=10, |&x| x < 5).collect();
    assert_eq!(data, vec![5, 6, 7, 8, 9, 10]);
}

#[test]
fn test_step_by() {
    let data: Vec<_> = CombinatorExt::step_by(0..10, 2).collect();
    assert_eq!(data, vec![0, 2, 4, 6, 8]);
}

#[test]
fn test_cycle() {
    let data: Vec<_> = CombinatorExt::cycle(vec![1, 2, 3].into_iter())
        .take(10)
        .collect();
    assert_eq!(data, vec![1, 2, 3, 1, 2, 3, 1, 2, 3, 1]);
}

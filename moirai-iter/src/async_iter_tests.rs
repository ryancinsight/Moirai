use super::*;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::Poll;

#[test]
fn async_source_iterators_do_not_store_unused_cursors() {
    assert_eq!(
        std::mem::size_of::<AsyncVecIter<u64>>(),
        std::mem::size_of::<Vec<u64>>()
    );
    assert_eq!(
        std::mem::size_of::<AsyncRangeIter>(),
        std::mem::size_of::<std::ops::Range<usize>>()
    );
}

#[tokio::test]
async fn test_async_vec_iter() {
    let data = vec![1, 2, 3, 4, 5];
    let iter = data.into_async_iter();

    let result: Vec<i32> = iter.collect().await;
    assert_eq!(result, vec![1, 2, 3, 4, 5]);
}

#[tokio::test]
async fn test_async_map() {
    let data = vec![1, 2, 3, 4, 5];
    let iter = data.into_async_iter();

    let doubled = iter.map(|x| async move { x * 2 });
    let result: Vec<i32> = doubled.collect().await;
    assert_eq!(result, vec![2, 4, 6, 8, 10]);
}

#[tokio::test]
async fn test_async_take_skip_window_values() {
    let result: Vec<i32> = vec![1, 2, 3, 4, 5, 6]
        .into_async_iter()
        .take(5)
        .skip(2)
        .collect()
        .await;

    assert_eq!(result, vec![3, 4, 5]);

    let empty: Vec<i32> = vec![1, 2].into_async_iter().skip(3).collect().await;
    assert_eq!(empty, Vec::<i32>::new());
}

#[tokio::test]
async fn test_async_enumerate_zip_values() {
    let right = vec![10, 20, 30, 40].into_async_iter();
    let result: Vec<(usize, (i32, i32))> = vec![1, 2, 3]
        .into_async_iter()
        .zip(right)
        .enumerate()
        .collect()
        .await;

    assert_eq!(result, vec![(0, (1, 10)), (1, (2, 20)), (2, (3, 30))]);
}

#[tokio::test]
async fn test_parallel_async_map() {
    let data = vec![1, 2, 3, 4, 5];
    let iter = data.into_async_iter().into_parallel();

    let doubled = iter.par_map(2, |x| async move { x * 2 });
    let result: Vec<i32> = doubled.collect().await;
    assert_eq!(result, vec![2, 4, 6, 8, 10]);
}

#[tokio::test]
async fn test_parallel_async_map_uses_bounded_in_flight_work() {
    let active = Arc::new(AtomicUsize::new(0));
    let max_active = Arc::new(AtomicUsize::new(0));
    let result: Vec<usize> = (0..8)
        .collect::<Vec<_>>()
        .into_async_iter()
        .into_parallel()
        .par_map(3, {
            let active = Arc::clone(&active);
            let max_active = Arc::clone(&max_active);
            move |item| {
                let active = Arc::clone(&active);
                let max_active = Arc::clone(&max_active);
                let mut yielded = false;
                futures::future::poll_fn(move |cx| {
                    if !yielded {
                        yielded = true;
                        let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                        max_active.fetch_max(now, Ordering::SeqCst);
                        cx.waker().wake_by_ref();
                        return Poll::Pending;
                    }
                    active.fetch_sub(1, Ordering::SeqCst);
                    Poll::Ready(item * 2)
                })
            }
        })
        .collect()
        .await;

    assert_eq!(result, vec![0, 2, 4, 6, 8, 10, 12, 14]);
    assert_eq!(max_active.load(Ordering::SeqCst), 3);
    assert_eq!(active.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn test_parallel_async_filter_uses_bounded_in_flight_work() {
    let active = Arc::new(AtomicUsize::new(0));
    let max_active = Arc::new(AtomicUsize::new(0));
    let result: Vec<usize> = (0..8)
        .collect::<Vec<_>>()
        .into_async_iter()
        .into_parallel()
        .par_filter(4, {
            let active = Arc::clone(&active);
            let max_active = Arc::clone(&max_active);
            move |item| {
                let item = *item;
                let active = Arc::clone(&active);
                let max_active = Arc::clone(&max_active);
                let mut yielded = false;
                futures::future::poll_fn(move |cx| {
                    if !yielded {
                        yielded = true;
                        let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                        max_active.fetch_max(now, Ordering::SeqCst);
                        cx.waker().wake_by_ref();
                        return Poll::Pending;
                    }
                    active.fetch_sub(1, Ordering::SeqCst);
                    Poll::Ready(item % 2 == 0)
                })
            }
        })
        .collect()
        .await;

    assert_eq!(result, vec![0, 2, 4, 6]);
    assert_eq!(max_active.load(Ordering::SeqCst), 4);
    assert_eq!(active.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn test_parallel_async_for_each_uses_bounded_in_flight_work() {
    let active = Arc::new(AtomicUsize::new(0));
    let max_active = Arc::new(AtomicUsize::new(0));
    let sum = Arc::new(AtomicUsize::new(0));
    (0..8)
        .collect::<Vec<_>>()
        .into_async_iter()
        .into_parallel()
        .par_for_each(2, {
            let active = Arc::clone(&active);
            let max_active = Arc::clone(&max_active);
            let sum = Arc::clone(&sum);
            move |item| {
                let active = Arc::clone(&active);
                let max_active = Arc::clone(&max_active);
                let sum = Arc::clone(&sum);
                let mut yielded = false;
                futures::future::poll_fn(move |cx| {
                    if !yielded {
                        yielded = true;
                        let now = active.fetch_add(1, Ordering::SeqCst) + 1;
                        max_active.fetch_max(now, Ordering::SeqCst);
                        cx.waker().wake_by_ref();
                        return Poll::Pending;
                    }
                    sum.fetch_add(item, Ordering::SeqCst);
                    active.fetch_sub(1, Ordering::SeqCst);
                    Poll::Ready(())
                })
            }
        })
        .await;

    assert_eq!(sum.load(Ordering::SeqCst), 28);
    assert_eq!(max_active.load(Ordering::SeqCst), 2);
    assert_eq!(active.load(Ordering::SeqCst), 0);
}

#[tokio::test]
async fn test_async_filter_fold_reduce_values() {
    let filtered: Vec<i32> = vec![1, 2, 3, 4, 5, 6]
        .into_async_iter()
        .filter(|value| {
            let value = *value;
            async move { value % 2 == 0 }
        })
        .collect()
        .await;
    assert_eq!(filtered, vec![2, 4, 6]);

    let folded = vec![1, 2, 3]
        .into_async_iter()
        .fold(10, |acc, item| async move { acc - item })
        .await;
    assert_eq!(folded, 4);

    let reduced = vec![1, 2, 3, 4]
        .into_async_iter()
        .reduce(|left, right| async move { left + right })
        .await;
    assert_eq!(reduced, Some(10));
}

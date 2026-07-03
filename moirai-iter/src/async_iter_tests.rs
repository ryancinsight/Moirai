use super::*;
use std::future::Future;
use std::pin::pin;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

/// A user future that returns `Pending` exactly once (re-waking itself) before
/// yielding its value. Driving a terminal over items backed by this future
/// forces the terminal's `poll` to observe `Pending` — proving it awaits
/// cooperatively instead of blocking the executor synchronously.
struct PendingOnce<T> {
    value: Option<T>,
    yielded: bool,
}

impl<T> PendingOnce<T> {
    fn new(value: T) -> Self {
        Self {
            value: Some(value),
            yielded: false,
        }
    }
}

impl<T: Unpin> Future for PendingOnce<T> {
    type Output = T;

    fn poll(mut self: std::pin::Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<T> {
        if !self.yielded {
            self.yielded = true;
            cx.waker().wake_by_ref();
            return Poll::Pending;
        }
        Poll::Ready(
            self.value
                .take()
                .expect("PendingOnce polled after completion"),
        )
    }
}

/// Waker that records whether it was woken, so a manual poll loop can advance
/// only when the future asked to be re-polled.
struct FlagWaker(AtomicBool);

impl Wake for FlagWaker {
    fn wake(self: Arc<Self>) {
        self.0.store(true, Ordering::SeqCst);
    }
    fn wake_by_ref(self: &Arc<Self>) {
        self.0.store(true, Ordering::SeqCst);
    }
}

/// Manually drive a future to completion, counting how many `Pending` polls it
/// returns. A synchronous `block_on`-in-`poll` terminal would complete in a
/// single poll (count `0`); a cooperative terminal yields `Pending` at least
/// once per pending item future.
fn drive_counting_pending<F: Future>(fut: F) -> (F::Output, usize) {
    let flag = Arc::new(FlagWaker(AtomicBool::new(true)));
    let waker = Waker::from(Arc::clone(&flag));
    let mut cx = Context::from_waker(&waker);
    let mut fut = pin!(fut);
    let mut pending_polls = 0usize;
    loop {
        assert!(
            flag.0.swap(false, Ordering::SeqCst),
            "future returned Pending without registering a wake — would stall"
        );
        match fut.as_mut().poll(&mut cx) {
            Poll::Ready(out) => return (out, pending_polls),
            Poll::Pending => pending_polls += 1,
        }
    }
}

#[test]
fn for_each_awaits_cooperatively_without_blocking() {
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));
    let seen_for_closure = Arc::clone(&seen);
    let fut = vec![1, 2, 3].into_async_iter().for_each(move |item| {
        let seen = Arc::clone(&seen_for_closure);
        async move {
            let doubled = PendingOnce::new(item * 2).await;
            seen.lock().expect("mutex poisoned").push(doubled);
        }
    });
    let ((), pending_polls) = drive_counting_pending(fut);
    assert_eq!(*seen.lock().expect("mutex poisoned"), vec![2, 4, 6]);
    assert_eq!(pending_polls, 3, "one Pending per item future");
}

#[test]
fn fold_threads_accumulator_across_cooperative_polls() {
    let fut = vec![1, 2, 3, 4]
        .into_async_iter()
        .fold(100, |acc, item| async move {
            PendingOnce::new(acc - item).await
        });
    let (folded, pending_polls) = drive_counting_pending(fut);
    assert_eq!(folded, 90);
    assert_eq!(pending_polls, 4);
}

#[test]
fn reduce_accumulates_across_cooperative_polls() {
    let fut = vec![1, 2, 3, 4]
        .into_async_iter()
        .reduce(|left, right| async move { PendingOnce::new(left + right).await });
    let (reduced, pending_polls) = drive_counting_pending(fut);
    assert_eq!(reduced, Some(10));
    // Three reduce steps for four items, each pending once.
    assert_eq!(pending_polls, 3);
}

#[test]
fn reduce_empty_input_yields_none_without_pending() {
    let fut = Vec::<i32>::new()
        .into_async_iter()
        .reduce(|left, right| async move { PendingOnce::new(left + right).await });
    let (reduced, pending_polls) = drive_counting_pending(fut);
    assert_eq!(reduced, None);
    assert_eq!(pending_polls, 0);
}

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
async fn par_for_each_drives_runtime_timers_without_blocking_the_executor() {
    // Each item awaits a real tokio timer. A blocking implementation (driving
    // the work with `block_on` inside `poll`) cannot advance the outer runtime's
    // reactor while it blocks, so the timers never fire and this hangs. A
    // cooperative future yields and the runtime drives the timers to completion.
    let count = Arc::new(AtomicUsize::new(0));
    let count_for_items = Arc::clone(&count);

    (0..4)
        .collect::<Vec<_>>()
        .into_async_iter()
        .into_parallel()
        .par_for_each(2, move |_| {
            let count = Arc::clone(&count_for_items);
            async move {
                tokio::time::sleep(std::time::Duration::from_millis(5)).await;
                count.fetch_add(1, Ordering::SeqCst);
            }
        })
        .await;

    assert_eq!(count.load(Ordering::SeqCst), 4);
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

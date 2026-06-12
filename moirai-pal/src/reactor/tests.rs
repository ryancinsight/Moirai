use super::core::IoReactor;
use super::future::{reactor_future_fits, INLINE_REACTOR_TASK_WORDS, ReactorTaskFutureStorage};
use super::task::TaskId;
use futures::task::noop_waker_ref;
use std::future::Future;
use std::pin::Pin;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

#[test]
fn test_reactor_creation() {
    let reactor = IoReactor::new();
    assert!(reactor.is_ok());
}

#[test]
fn test_task_id_generation() {
    let id1 = TaskId::new();
    let id2 = TaskId::new();
    assert_ne!(id1, id2);
}

#[test]
fn test_reactor_metrics() {
    let reactor = IoReactor::new().unwrap();
    let metrics = reactor.metrics();
    assert_eq!(metrics.events_processed.load(Ordering::Relaxed), 0);
    assert_eq!(metrics.tasks_executed.load(Ordering::Relaxed), 0);
}

#[test]
fn spawned_ready_task_handle_completes_after_iteration() {
    let reactor = IoReactor::new().expect("reactor must be created");
    let handle = reactor.spawn(async {});
    let mut handle = Box::pin(handle);
    let waker = noop_waker_ref();
    let mut context = Context::from_waker(waker);

    assert!(matches!(
        Future::poll(handle.as_mut(), &mut context),
        Poll::Pending
    ));

    reactor
        .run_iteration(Some(Duration::from_millis(0)))
        .expect("reactor iteration must complete");

    assert!(matches!(
        Future::poll(handle.as_mut(), &mut context),
        Poll::Ready(())
    ));

    let metrics = reactor.metrics();
    assert_eq!(metrics.tasks_executed.load(Ordering::Relaxed), 1);
}

struct MaxInlineReadyFuture {
    values: [usize; INLINE_REACTOR_TASK_WORDS],
}

impl Future for MaxInlineReadyFuture {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        assert_eq!(self.values[0], 1);
        Poll::Ready(())
    }
}

struct InlineObservedReadyFuture {
    values: [usize; INLINE_REACTOR_TASK_WORDS - 1],
    observed: Arc<AtomicUsize>,
}

impl Future for InlineObservedReadyFuture {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.observed.store(self.values[0], Ordering::Relaxed);
        Poll::Ready(())
    }
}

struct OversizedShapeFuture {
    values: [usize; INLINE_REACTOR_TASK_WORDS + 1],
}

impl Future for OversizedShapeFuture {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        assert_eq!(self.values[0], 1);
        Poll::Ready(())
    }
}

struct OversizedObservedReadyFuture {
    values: [usize; INLINE_REACTOR_TASK_WORDS + 1],
    observed: Arc<AtomicUsize>,
}

impl Future for OversizedObservedReadyFuture {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<Self::Output> {
        self.observed.store(self.values[0], Ordering::Relaxed);
        Poll::Ready(())
    }
}

#[test]
fn reactor_future_storage_budget_is_static_and_bounded() {
    assert_eq!(
        core::mem::size_of::<ReactorTaskFutureStorage>(),
        INLINE_REACTOR_TASK_WORDS * core::mem::size_of::<usize>()
    );
    assert!(reactor_future_fits::<MaxInlineReadyFuture>());
    assert!(!reactor_future_fits::<OversizedShapeFuture>());
    assert!(reactor_future_fits::<Box<OversizedShapeFuture>>());
}

#[test]
fn spawned_inline_and_oversized_reactor_futures_complete() {
    let reactor = IoReactor::new().expect("reactor must be created");
    let inline_observed = Arc::new(AtomicUsize::new(0));
    let oversized_observed = Arc::new(AtomicUsize::new(0));

    let inline_handle = reactor.spawn(InlineObservedReadyFuture {
        values: [7; INLINE_REACTOR_TASK_WORDS - 1],
        observed: Arc::clone(&inline_observed),
    });
    let oversized_handle = reactor.spawn(OversizedObservedReadyFuture {
        values: [11; INLINE_REACTOR_TASK_WORDS + 1],
        observed: Arc::clone(&oversized_observed),
    });

    reactor
        .run_iteration(Some(Duration::from_millis(0)))
        .expect("reactor iteration must complete");

    assert_eq!(inline_observed.load(Ordering::Relaxed), 7);
    assert_eq!(oversized_observed.load(Ordering::Relaxed), 11);

    let waker = noop_waker_ref();
    let mut context = Context::from_waker(waker);
    let mut inline_handle = Box::pin(inline_handle);
    let mut oversized_handle = Box::pin(oversized_handle);
    assert!(matches!(
        Future::poll(inline_handle.as_mut(), &mut context),
        Poll::Ready(())
    ));
    assert!(matches!(
        Future::poll(oversized_handle.as_mut(), &mut context),
        Poll::Ready(())
    ));

    let metrics = reactor.metrics();
    assert_eq!(metrics.tasks_executed.load(Ordering::Relaxed), 2);
}

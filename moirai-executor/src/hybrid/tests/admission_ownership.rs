use std::{
    future::Future,
    pin::Pin,
    sync::{
        Arc, Barrier,
        atomic::{AtomicUsize, Ordering},
    },
    task::{Context, Poll},
};

use moirai_core::{
    Priority,
    error::{ExecutorError, ExecutorResult},
    executor::{ExecutorConfig, ExecutorControl, TaskManager, TaskSpawner, TaskStatus},
    task::TaskId,
};

use super::{
    super::{HybridExecutor, shutdown::Admission},
    fixtures::gate_single_worker,
};
use crate::SyncTask;

struct AdmissionDropProbe<'a>(&'a Admission, &'a AtomicUsize);

impl Drop for AdmissionDropProbe<'_> {
    fn drop(&mut self) {
        self.1.store(self.0.active_calls(), Ordering::Release);
    }
}

struct DropCounter(Arc<AtomicUsize>);

impl Drop for DropCounter {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::Relaxed);
    }
}

struct PanickingInput<'a> {
    admission: &'a Admission,
    active: &'a AtomicUsize,
}

impl Drop for PanickingInput<'_> {
    fn drop(&mut self) {
        self.active
            .store(self.admission.active_calls(), Ordering::Release);
        panic!("admitted input drop panic");
    }
}

fn assert_shutting_down<T>(result: ExecutorResult<T>) {
    assert!(matches!(result, Err(ExecutorError::ShuttingDown)));
}

struct UnpolledFuture {
    polls: Arc<AtomicUsize>,
    drops: Arc<AtomicUsize>,
}

impl Future for UnpolledFuture {
    type Output = ();

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Self::Output> {
        self.polls.fetch_add(1, Ordering::Relaxed);
        Poll::Pending
    }
}

impl Drop for UnpolledFuture {
    fn drop(&mut self) {
        self.drops.fetch_add(1, Ordering::Relaxed);
    }
}

#[test]
fn rejected_first_async_admission_releases_tracking_and_future() {
    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        max_global_queue_size: 2,
        ..ExecutorConfig::default()
    })
    .unwrap();
    let (release, gate) = gate_single_worker::<SyncTask>(&executor);
    let first = executor
        .spawn_result::<SyncTask, _>(Priority::Normal, None, || ())
        .unwrap();
    let second = executor
        .spawn_result::<SyncTask, _>(Priority::Normal, None, || ())
        .unwrap();
    let rejected_id = TaskId::new(
        second
            .id()
            .0
            .checked_add(1)
            .expect("invariant: test task id has a successor"),
    );
    let polls = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));

    let rejection = match executor.spawn_async(UnpolledFuture {
        polls: Arc::clone(&polls),
        drops: Arc::clone(&drops),
    }) {
        Ok(_) => panic!("the third queued task must exceed the two-slot admission bound"),
        Err(error) => error,
    };

    assert!(matches!(rejection, ExecutorError::ResourceExhausted(_)));
    assert_eq!(polls.load(Ordering::Relaxed), 0);
    assert_eq!(drops.load(Ordering::Relaxed), 1);
    assert_eq!(
        executor.task_status(rejected_id),
        Some(TaskStatus::Cancelled)
    );

    release.send(()).unwrap();
    assert_eq!(gate.join(), Some(Ok(())));
    assert_eq!(first.join(), Some(Ok(())));
    assert_eq!(second.join(), Some(Ok(())));
    ExecutorControl::shutdown(&executor);
}

#[test]
fn closed_async_registration_rejects_and_drops_unpolled_future() {
    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        ..ExecutorConfig::default()
    })
    .unwrap();
    assert!(executor.async_tasks.close_and_snapshot().is_empty());
    let polls = Arc::new(AtomicUsize::new(0));
    let drops = Arc::new(AtomicUsize::new(0));

    let rejection = match executor.spawn_async(UnpolledFuture {
        polls: Arc::clone(&polls),
        drops: Arc::clone(&drops),
    }) {
        Ok(_) => panic!("closed async registration must reject the spawn"),
        Err(error) => error,
    };

    assert!(matches!(rejection, ExecutorError::ShuttingDown));
    assert_eq!(polls.load(Ordering::Relaxed), 0);
    assert_eq!(drops.load(Ordering::Relaxed), 1);
    let successor = executor
        .spawn_result::<SyncTask, _>(Priority::Normal, None, || ())
        .unwrap();
    let rejected_id = TaskId::new(
        successor
            .id()
            .0
            .checked_sub(1)
            .expect("invariant: successor follows the rejected task"),
    );
    assert_eq!(
        executor.task_status(rejected_id),
        Some(TaskStatus::Cancelled)
    );
    assert_eq!(successor.join(), Some(Ok(())));
    ExecutorControl::shutdown(&executor);
}

#[test]
fn closed_facade_rejects_every_submission_before_user_code() {
    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        ..ExecutorConfig::default()
    })
    .unwrap();
    executor.admission.close();
    let bodies = Arc::new(AtomicUsize::new(0));
    let closure_drops = Arc::new(AtomicUsize::new(0));

    let body_count = Arc::clone(&bodies);
    let drop_count = Arc::clone(&closure_drops);
    let capture = DropCounter(drop_count);
    assert_shutting_down(
        executor.spawn_result::<SyncTask, _>(Priority::Normal, None, move || {
            let _capture = capture;
            body_count.fetch_add(1, Ordering::Relaxed);
        }),
    );

    let body_count = Arc::clone(&bodies);
    let capture = DropCounter(Arc::clone(&closure_drops));
    assert_shutting_down(executor.spawn_detached(move || {
        let _capture = capture;
        body_count.fetch_add(1, Ordering::Relaxed);
    }));

    let polls = Arc::new(AtomicUsize::new(0));
    let future_drops = Arc::new(AtomicUsize::new(0));
    assert_shutting_down(executor.spawn_async(UnpolledFuture {
        polls: Arc::clone(&polls),
        drops: Arc::clone(&future_drops),
    }));

    let body_count = Arc::clone(&bodies);
    assert_shutting_down(executor.scope::<SyncTask, _>(move |_| {
        body_count.fetch_add(1, Ordering::Relaxed);
        Ok(())
    }));

    let body_count = Arc::clone(&bodies);
    assert_shutting_down(executor.for_each_indexed::<SyncTask, _>(0, move |_| {
        body_count.fetch_add(1, Ordering::Relaxed);
    }));

    let body_count = Arc::clone(&bodies);
    assert_shutting_down(executor.map_reduce_indexed::<SyncTask, _, _, _>(
        0,
        0usize,
        move |_| {
            body_count.fetch_add(1, Ordering::Relaxed);
            1
        },
        |left, right| left + right,
    ));

    assert_eq!(bodies.load(Ordering::Relaxed), 0);
    assert_eq!(closure_drops.load(Ordering::Relaxed), 2);
    assert_eq!(polls.load(Ordering::Relaxed), 0);
    assert_eq!(future_drops.load(Ordering::Relaxed), 1);
    assert_eq!(executor.admission.active_calls(), 0);
    ExecutorControl::shutdown(&executor);
}

#[test]
fn scoped_calls_stay_counted_through_cleanup_and_unwind() {
    let executor = HybridExecutor::new(ExecutorConfig {
        worker_threads: 1,
        ..ExecutorConfig::default()
    })
    .unwrap();
    let active = &executor.admission;
    let scope_observed = AtomicUsize::new(0);
    let probe = AdmissionDropProbe(active, &scope_observed);
    executor
        .scope::<SyncTask, _>(move |_| {
            let _probe = probe;
            assert_eq!(active.active_calls(), 1);
            Ok(())
        })
        .unwrap();
    assert_eq!(scope_observed.load(Ordering::Acquire), 1);
    assert_eq!(executor.admission.active_calls(), 0);
    let reduce_observed = AtomicUsize::new(0);
    let probe = AdmissionDropProbe(active, &reduce_observed);
    let sum = executor
        .map_reduce_indexed::<SyncTask, _, _, _>(
            2,
            0usize,
            |_| 1,
            move |left, right| {
                let _probe = &probe;
                assert_eq!(active.active_calls(), 1);
                left + right
            },
        )
        .unwrap();
    assert_eq!(sum, 2);
    assert_eq!(reduce_observed.load(Ordering::Acquire), 1);
    assert_eq!(executor.admission.active_calls(), 0);
    let admission = Admission::new();
    let observed = AtomicUsize::new(0);
    let probe = AdmissionDropProbe(&admission, &observed);
    let unwind = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let _: ExecutorResult<()> = admission.run(move || {
            let _probe = probe;
            panic!("accepted call panic");
        });
    }));

    let Err(payload) = unwind else {
        panic!("admission call must propagate the panic");
    };
    assert_eq!(payload.downcast_ref::<&str>(), Some(&"accepted call panic"));
    assert_eq!(observed.load(Ordering::Acquire), 1);
    assert_eq!(admission.active_calls(), 0);
    let abandoned = Admission::new();
    let abandoned_active = AtomicUsize::new(0);
    let call = abandoned
        .admit(PanickingInput {
            admission: &abandoned,
            active: &abandoned_active,
        })
        .unwrap();
    abandoned.close();
    let abandoned_unwind = std::panic::catch_unwind(|| drop(call));
    let Err(payload) = abandoned_unwind else {
        panic!("abandoned admitted input must propagate its drop panic");
    };
    assert_eq!(
        payload.downcast_ref::<&str>(),
        Some(&"admitted input drop panic")
    );
    assert_eq!(abandoned_active.load(Ordering::Acquire), 1);
    assert!(abandoned.is_closed());
    assert_eq!(abandoned.active_calls(), 0);
    ExecutorControl::shutdown(&executor);
}

#[test]
fn admission_capacity_and_close_race_are_typed() {
    let full = Admission::at_capacity();
    assert!(matches!(
        full.run(|| Ok(())),
        Err(ExecutorError::ResourceExhausted(_))
    ));
    for _ in 0..16 {
        let admission = Admission::new();
        let barrier = Barrier::new(2);
        let bodies = AtomicUsize::new(0);
        std::thread::scope(|scope| {
            let contender = scope.spawn(|| {
                barrier.wait();
                admission.run(|| {
                    assert_eq!(admission.active_calls(), 1);
                    bodies.fetch_add(1, Ordering::Relaxed);
                    Ok(())
                })
            });
            barrier.wait();
            admission.close();
            match contender.join().unwrap() {
                Ok(()) => assert_eq!(bodies.load(Ordering::Relaxed), 1),
                Err(ExecutorError::ShuttingDown) => {
                    assert_eq!(bodies.load(Ordering::Relaxed), 0);
                }
                Err(error) => panic!("unexpected admission result: {error}"),
            }
        });

        assert!(admission.is_closed());
        assert_eq!(admission.active_calls(), 0);
    }
}

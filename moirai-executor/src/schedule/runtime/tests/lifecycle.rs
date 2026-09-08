//! Scheduler lifecycle: routing every work class through one facade.

use super::*;

#[test]
fn scheduler_runs_all_work_classes_through_one_facade() {
    let scheduler = ThreadScheduler::new(2, "test-scheduler").unwrap();
    let completed = Arc::new(AtomicUsize::new(0));
    let (sender, receiver) = mpsc::channel();

    {
        let completed = Arc::clone(&completed);
        let sender = sender.clone();
        scheduler
            .schedule::<SyncTask, _>(Priority::Normal, None, move |_| {
                completed.fetch_add(1, Ordering::AcqRel);
                sender.send(()).unwrap();
            })
            .unwrap();
    }

    {
        let completed = Arc::clone(&completed);
        let sender = sender.clone();
        scheduler
            .schedule::<AsyncTask, _>(Priority::Normal, None, move |_| {
                completed.fetch_add(1, Ordering::AcqRel);
                sender.send(()).unwrap();
            })
            .unwrap();
    }

    {
        let completed = Arc::clone(&completed);
        scheduler
            .schedule::<BlockingTask, _>(Priority::Normal, None, move |_| {
                completed.fetch_add(1, Ordering::AcqRel);
                sender.send(()).unwrap();
            })
            .unwrap();
    }

    for _ in 0..3 {
        receiver.recv().unwrap();
    }

    scheduler.shutdown();
    let metrics = scheduler.metrics();

    assert_eq!(completed.load(Ordering::Acquire), 3);
    assert_eq!(metrics.worker_count, 2);
    assert_eq!(metrics.pending_tasks, 0);
    assert_eq!(metrics.completed_tasks, 3);
    assert_eq!(metrics.failed_tasks, 0);
}

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use moirai_core::Priority;
use moirai_executor::schedule::{BlockingTask, ThreadScheduler};

fn blocking_lane_schedule_join(c: &mut Criterion) {
    let scheduler = ThreadScheduler::new(1, "criterion-blocking-lane").unwrap();
    let completed = Arc::new(AtomicUsize::new(0));
    let mut expected = 0usize;

    c.bench_function("blocking_lane_schedule_join", |bench| {
        bench.iter(|| {
            let completed_in_job = Arc::clone(&completed);
            scheduler
                .schedule::<BlockingTask, _>(Priority::Normal, None, move |_| {
                    completed_in_job.fetch_add(1, Ordering::Relaxed);
                    black_box(13usize);
                })
                .expect("blocking lane admission must remain available");
            scheduler.join().expect("blocking lane job must complete");
            expected += 1;
            assert_eq!(completed.load(Ordering::Relaxed), expected);
        });
    });

    scheduler.shutdown();
}

fn blocking_lane_concurrent_producers(c: &mut Criterion) {
    const PRODUCERS: usize = 4;
    const JOBS_PER_PRODUCER: usize = 32;
    let scheduler = ThreadScheduler::new(PRODUCERS, "criterion-blocking-producers").unwrap();
    let completed = Arc::new(AtomicUsize::new(0));
    let expected = PRODUCERS * JOBS_PER_PRODUCER;
    let mut expected_completed = 0usize;

    c.bench_function("blocking_lane_concurrent_producers", |bench| {
        bench.iter(|| {
            std::thread::scope(|scope| {
                for _ in 0..PRODUCERS {
                    let scheduler = scheduler.clone();
                    let completed = Arc::clone(&completed);
                    scope.spawn(move || {
                        for _ in 0..JOBS_PER_PRODUCER {
                            let completed = Arc::clone(&completed);
                            scheduler
                                .schedule::<BlockingTask, _>(Priority::Normal, None, move |_| {
                                    completed.fetch_add(1, Ordering::Relaxed);
                                })
                                .expect("blocking lane admission must remain available");
                        }
                    });
                }
            });
            scheduler.join().expect("blocking lane jobs must complete");
            expected_completed += expected;
            assert_eq!(completed.load(Ordering::Relaxed), expected_completed);
            black_box(expected);
        });
    });

    scheduler.shutdown();
}

criterion_group!(
    blocking_lane,
    blocking_lane_schedule_join,
    blocking_lane_concurrent_producers
);
criterion_main!(blocking_lane);

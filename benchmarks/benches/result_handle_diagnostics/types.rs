use criterion::{black_box, Criterion};
use moirai::Moirai;
use moirai_core::{
    executor::{ExecutorConfig, TaskSpawner},
    TaskError, TaskHandle, TaskId,
};
use moirai_executor::{
    BlockingTask, ExecutorMetrics, HybridExecutor, TaskRegistry, ThreadScheduler,
};
#[cfg(feature = "scheduler-diagnostics")]
use moirai_executor::{ContendedWakeDecision, EmptyWakeDecision, SaturatedWakeDecision};
use std::{
    cell::UnsafeCell,
    future::Future,
    panic::{catch_unwind, AssertUnwindSafe},
    pin::Pin,
    sync::{
        atomic::{AtomicU64, AtomicU8, AtomicUsize, Ordering},
        Arc, Mutex,
    },
    task::{Context, Poll, Wake, Waker},
    thread,
    time::{Duration, Instant},
};

#[cfg(windows)]
use std::sync::atomic::AtomicI64;

pub(crate) const BENCHMARK_SAMPLE_SIZE: usize = 20;
pub(crate) const BENCHMARK_MEASUREMENT_SECONDS: u64 = 2;
pub(crate) const BENCHMARK_WARM_UP_MILLIS: u64 = 500;
const WORKER_THREADS: usize = 4;
const READY_VALUE: usize = 42;
const CAPTURE_WORDS: usize = 10;
const CAPTURED_READY_VALUE: usize = CAPTURE_WORDS;
const MAX_INLINE_CAPTURE_WORDS: usize = 14;
const OVERSIZED_CAPTURE_WORDS: usize = 32;
const OVERSIZED_CAPTURED_READY_VALUE: usize = OVERSIZED_CAPTURE_WORDS;
const TASK_ID: TaskId = TaskId(1);
const BLOCKING_NORMAL_WORKER: usize = 3;
#[cfg(feature = "scheduler-diagnostics")]
const SCHEDULER_JOIN_FAST_SPIN_ATTEMPTS: usize = 256;
const LIFECYCLE_TIMESTAMP_NOT_RECORDED: usize = usize::MAX;

struct DiagnosticLifecycle {
    created_at: Instant,
    started_after_ns: AtomicUsize,
    completed_after_ns: AtomicUsize,
    worker_id: AtomicUsize,
}

struct ElapsedOnlyLifecycle {
    created_at: Instant,
}

struct AtomicOnlyLifecycle {
    started_after_ns: AtomicUsize,
    completed_after_ns: AtomicUsize,
    worker_id: AtomicUsize,
}

struct StartInstantLifecycle {
    created_at: Instant,
    started_after_ns: AtomicUsize,
    completed_after_ns: AtomicUsize,
    worker_id: AtomicUsize,
}

struct StartInstantLifecycleToken<'a> {
    lifecycle: &'a StartInstantLifecycle,
    started_at: Instant,
    started_after_ns: usize,
}

struct CachedClockLifecycle {
    clock: Arc<CachedLifecycleClock>,
    started_after_ns: AtomicUsize,
    completed_after_ns: AtomicUsize,
    worker_id: AtomicUsize,
}

struct CachedLifecycleClock {
    origin: Instant,
    current_after_ns: AtomicUsize,
    stop: AtomicUsize,
}

struct CachedLifecycleClockGuard {
    clock: Arc<CachedLifecycleClock>,
    driver: Option<thread::JoinHandle<()>>,
}

#[cfg(windows)]
struct QpcLifecycle {
    origin_ticks: i64,
    started_after_ns: AtomicUsize,
    completed_after_ns: AtomicUsize,
    worker_id: AtomicUsize,
}

struct DurationOnlyLifecycle {
    started_at: Mutex<Option<Instant>>,
    duration_after_ns: AtomicUsize,
    worker_id: AtomicUsize,
}

#[derive(Default)]
struct WakeOnce {
    observed_pending: bool,
}

impl Future for WakeOnce {
    type Output = usize;

    fn poll(mut self: Pin<&mut Self>, context: &mut Context<'_>) -> Poll<Self::Output> {
        if self.observed_pending {
            Poll::Ready(black_box(READY_VALUE))
        } else {
            self.observed_pending = true;
            context.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

impl DiagnosticLifecycle {
    fn new() -> Self {
        Self {
            created_at: Instant::now(),
            started_after_ns: AtomicUsize::new(LIFECYCLE_TIMESTAMP_NOT_RECORDED),
            completed_after_ns: AtomicUsize::new(LIFECYCLE_TIMESTAMP_NOT_RECORDED),
            worker_id: AtomicUsize::new(usize::MAX),
        }
    }

    fn start(&self, worker_id: usize) -> usize {
        let started_after_ns = self.elapsed_nanos();
        self.started_after_ns
            .store(started_after_ns, Ordering::Release);
        self.worker_id.store(worker_id, Ordering::Release);
        started_after_ns
    }

    fn complete_since(&self, started_after_ns: usize) -> usize {
        let completed_after_ns = self.elapsed_nanos();
        self.completed_after_ns
            .store(completed_after_ns, Ordering::Release);
        completed_after_ns.saturating_sub(started_after_ns)
    }

    fn elapsed_nanos(&self) -> usize {
        elapsed_nanos_since(self.created_at)
    }
}

impl ElapsedOnlyLifecycle {
    fn new() -> Self {
        Self {
            created_at: Instant::now(),
        }
    }

    fn start(&self, worker_id: usize) -> usize {
        black_box(worker_id);
        self.elapsed_nanos()
    }

    fn complete_since(&self, started_after_ns: usize) -> usize {
        let completed_after_ns = self.elapsed_nanos();
        black_box(completed_after_ns.saturating_sub(started_after_ns))
    }

    fn elapsed_nanos(&self) -> usize {
        elapsed_nanos_since(self.created_at)
    }
}

impl AtomicOnlyLifecycle {
    fn new() -> Self {
        Self {
            started_after_ns: AtomicUsize::new(LIFECYCLE_TIMESTAMP_NOT_RECORDED),
            completed_after_ns: AtomicUsize::new(LIFECYCLE_TIMESTAMP_NOT_RECORDED),
            worker_id: AtomicUsize::new(usize::MAX),
        }
    }

    fn start(&self, worker_id: usize) -> usize {
        self.started_after_ns.store(1, Ordering::Release);
        self.worker_id.store(worker_id, Ordering::Release);
        1
    }

    fn complete_since(&self, started_after_ns: usize) -> usize {
        self.completed_after_ns
            .store(started_after_ns.saturating_add(1), Ordering::Release);
        1
    }
}

impl StartInstantLifecycle {
    fn new() -> Self {
        Self {
            created_at: Instant::now(),
            started_after_ns: AtomicUsize::new(LIFECYCLE_TIMESTAMP_NOT_RECORDED),
            completed_after_ns: AtomicUsize::new(LIFECYCLE_TIMESTAMP_NOT_RECORDED),
            worker_id: AtomicUsize::new(usize::MAX),
        }
    }

    fn start(&self, worker_id: usize) -> StartInstantLifecycleToken<'_> {
        let started_at = Instant::now();
        let started_after_ns = duration_nanos_between(self.created_at, started_at);
        self.started_after_ns
            .store(started_after_ns, Ordering::Release);
        self.worker_id.store(worker_id, Ordering::Release);
        StartInstantLifecycleToken {
            lifecycle: self,
            started_at,
            started_after_ns,
        }
    }
}

impl StartInstantLifecycleToken<'_> {
    fn complete(self) -> usize {
        let duration_ns = elapsed_nanos_since(self.started_at);
        let completed_after_ns = self
            .started_after_ns
            .saturating_add(duration_ns)
            .min(LIFECYCLE_TIMESTAMP_NOT_RECORDED - 1);
        self.lifecycle
            .completed_after_ns
            .store(completed_after_ns, Ordering::Release);
        duration_ns
    }
}

impl CachedClockLifecycle {
    fn new(clock: Arc<CachedLifecycleClock>) -> Self {
        Self {
            clock,
            started_after_ns: AtomicUsize::new(LIFECYCLE_TIMESTAMP_NOT_RECORDED),
            completed_after_ns: AtomicUsize::new(LIFECYCLE_TIMESTAMP_NOT_RECORDED),
            worker_id: AtomicUsize::new(usize::MAX),
        }
    }

    fn start(&self, worker_id: usize) -> usize {
        let started_after_ns = self.clock.sample();
        self.started_after_ns
            .store(started_after_ns, Ordering::Release);
        self.worker_id.store(worker_id, Ordering::Release);
        started_after_ns
    }

    fn complete_since(&self, started_after_ns: usize) -> usize {
        let completed_after_ns = self.clock.sample().max(started_after_ns);
        self.completed_after_ns
            .store(completed_after_ns, Ordering::Release);
        completed_after_ns.saturating_sub(started_after_ns)
    }
}

impl CachedLifecycleClock {
    fn new() -> Self {
        let origin = Instant::now();
        Self {
            origin,
            current_after_ns: AtomicUsize::new(0),
            stop: AtomicUsize::new(0),
        }
    }

    fn refresh(&self) {
        self.current_after_ns
            .store(elapsed_nanos_since(self.origin), Ordering::Relaxed);
    }

    fn sample(&self) -> usize {
        self.current_after_ns.load(Ordering::Relaxed)
    }
}

impl CachedLifecycleClockGuard {
    fn start() -> Self {
        let clock = Arc::new(CachedLifecycleClock::new());
        clock.refresh();
        let driver_clock = Arc::clone(&clock);
        let driver = thread::Builder::new()
            .name("result-handle-cached-lifecycle-clock".to_string())
            .spawn(move || {
                while driver_clock.stop.load(Ordering::Relaxed) == 0 {
                    driver_clock.refresh();
                    thread::sleep(Duration::from_micros(50));
                }
            })
            .expect("cached lifecycle clock driver must start");

        Self {
            clock,
            driver: Some(driver),
        }
    }

    fn clock(&self) -> Arc<CachedLifecycleClock> {
        Arc::clone(&self.clock)
    }
}

impl Drop for CachedLifecycleClockGuard {
    fn drop(&mut self) {
        self.clock.stop.store(1, Ordering::Relaxed);
        if let Some(driver) = self.driver.take() {
            driver
                .join()
                .expect("cached lifecycle clock driver must stop without panic");
        }
    }
}

#[cfg(windows)]
impl QpcLifecycle {
    fn new() -> Self {
        Self {
            origin_ticks: query_performance_counter(),
            started_after_ns: AtomicUsize::new(LIFECYCLE_TIMESTAMP_NOT_RECORDED),
            completed_after_ns: AtomicUsize::new(LIFECYCLE_TIMESTAMP_NOT_RECORDED),
            worker_id: AtomicUsize::new(usize::MAX),
        }
    }

    fn start(&self, worker_id: usize) -> usize {
        let started_after_ns = self.elapsed_nanos_from_origin();
        self.started_after_ns
            .store(started_after_ns, Ordering::Release);
        self.worker_id.store(worker_id, Ordering::Release);
        started_after_ns
    }

    fn complete_since(&self, started_after_ns: usize) -> usize {
        let completed_after_ns = self.elapsed_nanos_from_origin().max(started_after_ns);
        self.completed_after_ns
            .store(completed_after_ns, Ordering::Release);
        completed_after_ns.saturating_sub(started_after_ns)
    }

    fn elapsed_nanos_from_origin(&self) -> usize {
        let ticks = query_performance_counter().saturating_sub(self.origin_ticks);
        ticks_to_nanos(ticks, qpc_ticks_per_second())
    }
}

impl DurationOnlyLifecycle {
    fn new() -> Self {
        Self {
            started_at: Mutex::new(None),
            duration_after_ns: AtomicUsize::new(LIFECYCLE_TIMESTAMP_NOT_RECORDED),
            worker_id: AtomicUsize::new(usize::MAX),
        }
    }

    fn start(&self, worker_id: usize) {
        *self
            .started_at
            .lock()
            .expect("duration-only lifecycle lock must not be poisoned") = Some(Instant::now());
        self.worker_id.store(worker_id, Ordering::Release);
    }

    fn complete(&self) -> usize {
        let started_at = self
            .started_at
            .lock()
            .expect("duration-only lifecycle lock must not be poisoned")
            .expect("duration-only lifecycle must be started");
        let elapsed = elapsed_nanos_since(started_at);
        self.duration_after_ns.store(elapsed, Ordering::Release);
        elapsed
    }
}

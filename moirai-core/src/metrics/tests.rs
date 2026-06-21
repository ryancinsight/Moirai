//! Unit tests for metrics collectors and aggregators.

use super::*;
use crate::scheduler::SchedulerId;

#[test]
fn test_counter() {
    let counter = Counter::new();
    assert_eq!(counter.get(), 0);

    counter.increment();
    assert_eq!(counter.get(), 1);

    counter.add(5);
    assert_eq!(counter.get(), 6);

    counter.reset();
    assert_eq!(counter.get(), 0);
}

#[test]
fn test_gauge() {
    let gauge = Gauge::new();
    assert_eq!(gauge.get(), 0);

    gauge.set(10);
    assert_eq!(gauge.get(), 10);

    gauge.increment();
    assert_eq!(gauge.get(), 11);

    gauge.subtract(1);
    assert_eq!(gauge.get(), 10);

    gauge.add(5);
    assert_eq!(gauge.get(), 15);

    gauge.subtract(3);
    assert_eq!(gauge.get(), 12);
}

#[test]
fn test_histogram() {
    let histogram = Histogram::new();
    assert_eq!(histogram.count(), 0);
    assert_eq!(histogram.sum(), 0);
    #[allow(clippy::float_cmp)]
    {
        assert_eq!(histogram.average(), 0.0);
    }

    histogram.record(10);
    histogram.record(20);
    histogram.record(30);

    assert_eq!(histogram.count(), 3);
    assert_eq!(histogram.sum(), 60);
    #[allow(clippy::float_cmp)]
    {
        assert_eq!(histogram.average(), 20.0);
    }
}

#[test]
fn test_task_data() {
    let metrics = TaskData::new();

    metrics.spawned.increment();
    metrics.spawned.increment();
    assert_eq!(metrics.spawned.get(), 2);

    metrics.record_execution(core::time::Duration::from_millis(1));
    assert_eq!(metrics.completed.get(), 1);
    #[allow(clippy::float_cmp)]
    {
        assert_eq!(metrics.completion_rate(), 0.5);
    }

    metrics.record_wait(core::time::Duration::from_millis(1));
}

#[test]
fn test_scheduler_data() {
    let metrics = SchedulerData::new();

    metrics.queue_length.set(5);
    assert_eq!(metrics.queue_length.get(), 5);

    metrics.steal_attempts.increment();
    metrics.steal_attempts.increment();
    metrics.steal_attempts.increment();

    assert_eq!(metrics.steal_attempts.get(), 3);
    assert_eq!(metrics.successful_steals.get(), 0);
    assert!((metrics.steal_success_rate() - 0.0).abs() < f64::EPSILON);

    let utilization = metrics.record_cpu_utilization(0.75);
    assert!((utilization - 75.0).abs() < f32::EPSILON);
    assert_eq!(metrics.cpu_utilization.get(), 75);
}

#[test]
fn test_global_metrics() {
    let mut global = GlobalMetrics::new();

    global.scheduler(SchedulerId::new(1)).queue_length.set(5);
    global.scheduler(SchedulerId::new(2)).queue_length.set(10);
    global.scheduler(SchedulerId::new(3)).queue_length.set(15);

    assert!((global.snapshot().average_queue_length - 10.0).abs() < f64::EPSILON);

    global
        .scheduler(SchedulerId::new(1))
        .steal_attempts
        .increment();
    global
        .scheduler(SchedulerId::new(2))
        .steal_attempts
        .increment();
    global
        .scheduler(SchedulerId::new(3))
        .steal_attempts
        .increment();
    global
        .scheduler(SchedulerId::new(1))
        .steal_attempts
        .increment();

    assert_eq!(global.snapshot().total_steal_attempts, 4);

    global
        .scheduler(SchedulerId::new(1))
        .successful_steals
        .increment();
    global
        .scheduler(SchedulerId::new(2))
        .successful_steals
        .increment();
    global
        .scheduler(SchedulerId::new(3))
        .successful_steals
        .increment();
    global
        .scheduler(SchedulerId::new(1))
        .successful_steals
        .increment();

    assert_eq!(global.snapshot().total_successful_steals, 4);
}

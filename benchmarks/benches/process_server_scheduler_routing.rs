use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use moirai_core::Priority;
use moirai_executor::schedule::{
    AsyncLanesPerProcess, AsyncTask, BlockingTask, HybridRoutePolicy, HybridRouter, ProcessCount,
    RoutePolicy, RouteSummary, RouteTopology, ServerCount, ServerRoutePolicy, SyncTask, WorkClass,
    WorkerCount,
};
use std::time::Duration;

fn route_topology() -> RouteTopology {
    RouteTopology::new(
        WorkerCount::new(8),
        ProcessCount::new(4),
        AsyncLanesPerProcess::new(3),
        ServerCount::new(2),
    )
}

fn priority_weight(priority: Priority) -> usize {
    match priority {
        Priority::Low => 0,
        Priority::Normal => 1,
        Priority::High => 2,
        Priority::Critical => 3,
    }
}

fn route_checksum(
    current: usize,
    route_kind: usize,
    server: usize,
    process: usize,
    thread: usize,
    async_lane: Option<usize>,
) -> usize {
    current
        .wrapping_mul(1_000_003)
        .wrapping_add(route_kind.wrapping_mul(97))
        .wrapping_add(server.wrapping_mul(31))
        .wrapping_add(process.wrapping_mul(17))
        .wrapping_add(thread.wrapping_mul(13))
        .wrapping_add(async_lane.map_or(0, |lane| lane.wrapping_add(1)))
}

fn expected_summary<C, P>(topology: RouteTopology, priority: Priority, count: usize) -> RouteSummary
where
    C: WorkClass,
    P: RoutePolicy,
{
    let mut summary = RouteSummary::default();
    let priority = priority_weight(priority);

    for sequence in 0..count {
        let route_key = sequence
            .wrapping_add(C::AFFINITY_OFFSET)
            .wrapping_add(priority);
        let thread = route_key % topology.worker_threads().get();
        let process = sequence
            .wrapping_div(topology.worker_threads().get())
            .wrapping_add(C::AFFINITY_OFFSET)
            .wrapping_add(priority)
            % topology.processes().get();
        let async_lane = if C::USES_ASYNC_LANE {
            Some(sequence.wrapping_add(process) % topology.async_lanes_per_process().get())
        } else {
            None
        };

        if P::ENABLE_SERVER_ROUTES
            && topology.servers().get() != 0
            && route_key % P::SERVER_PERIOD.max(1) == 0
        {
            summary.server_routes += 1;
            if async_lane.is_some() {
                summary.async_lane_routes += 1;
            }
            summary.checksum = route_checksum(
                summary.checksum,
                3,
                route_key % topology.servers().get(),
                process,
                thread,
                async_lane,
            );
        } else if P::ENABLE_PROCESS_ROUTES
            && topology.processes().get() > 1
            && (C::USES_ASYNC_LANE || route_key % P::PROCESS_PERIOD.max(1) == 0)
        {
            summary.process_routes += 1;
            if async_lane.is_some() {
                summary.async_lane_routes += 1;
            }
            summary.checksum = route_checksum(summary.checksum, 2, 0, process, thread, async_lane);
        } else {
            summary.thread_routes += 1;
            summary.checksum = route_checksum(summary.checksum, 1, 0, process, thread, None);
        }
    }

    summary
}

fn verify_summary<C, P>(router: &HybridRouter<P>, priority: Priority, count: usize) -> RouteSummary
where
    C: WorkClass,
    P: RoutePolicy,
{
    let expected = expected_summary::<C, P>(router.topology(), priority, count);
    let observed = router.summarize::<C>(priority, count);
    assert_eq!(observed, expected);
    assert_eq!(observed.total_routes(), count);
    observed
}

fn benchmark_thread_process_server_summary(c: &mut Criterion) {
    let topology = route_topology();
    let hybrid = HybridRouter::<HybridRoutePolicy>::new(topology);
    let server = HybridRouter::<ServerRoutePolicy>::new(topology);
    let mut group = c.benchmark_group("scheduler_route_thread_process_server_summary");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(1));

    for count in [4_096usize, 65_536] {
        let observed =
            verify_summary::<SyncTask, HybridRoutePolicy>(&hybrid, Priority::Normal, count);
        assert!(observed.thread_routes > 0);
        assert!(observed.process_routes > 0);
        assert!(observed.server_routes > 0);
        assert_eq!(observed.async_lane_routes, 0);
        group.bench_function(BenchmarkId::new("sync_hybrid", count), |b| {
            b.iter(|| {
                black_box(
                    hybrid.summarize::<SyncTask>(black_box(Priority::Normal), black_box(count)),
                )
            });
        });

        let observed =
            verify_summary::<AsyncTask, HybridRoutePolicy>(&hybrid, Priority::High, count);
        assert_eq!(observed.async_lane_routes, count);
        assert_eq!(observed.thread_routes, 0);
        group.bench_function(BenchmarkId::new("async_hybrid", count), |b| {
            b.iter(|| {
                black_box(
                    hybrid.summarize::<AsyncTask>(black_box(Priority::High), black_box(count)),
                )
            });
        });

        let observed =
            verify_summary::<BlockingTask, ServerRoutePolicy>(&server, Priority::High, count);
        assert!(observed.server_routes > 0);
        assert_eq!(observed.async_lane_routes, 0);
        group.bench_function(BenchmarkId::new("blocking_server", count), |b| {
            b.iter(|| {
                black_box(
                    server.summarize::<BlockingTask>(black_box(Priority::High), black_box(count)),
                )
            });
        });
    }

    group.finish();
}

fn benchmark_async_process_lanes(c: &mut Criterion) {
    let topology = route_topology();
    let router = HybridRouter::<HybridRoutePolicy>::new(topology);
    let mut group = c.benchmark_group("scheduler_route_async_process_lanes");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(1));

    for count in [16_384usize, 65_536] {
        let observed =
            verify_summary::<AsyncTask, HybridRoutePolicy>(&router, Priority::Critical, count);
        assert_eq!(observed.total_routes(), observed.async_lane_routes);
        assert!(observed.process_routes > observed.server_routes);

        group.bench_function(BenchmarkId::new("async_process_lane_summary", count), |b| {
            b.iter(|| {
                black_box(
                    router.summarize::<AsyncTask>(black_box(Priority::Critical), black_box(count)),
                )
            });
        });
    }

    group.finish();
}

fn benchmark_policy_overhead(c: &mut Criterion) {
    let topology = route_topology();
    let thread = HybridRouter::<moirai_executor::schedule::ThreadRoutePolicy>::new(topology);
    let hybrid = HybridRouter::<HybridRoutePolicy>::new(topology);
    let server = HybridRouter::<ServerRoutePolicy>::new(topology);
    let mut group = c.benchmark_group("scheduler_route_policy_overhead");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(1));

    for count in [16_384usize, 65_536] {
        let thread_summary = verify_summary::<SyncTask, moirai_executor::schedule::ThreadRoutePolicy>(
            &thread,
            Priority::Normal,
            count,
        );
        assert_eq!(thread_summary.thread_routes, count);
        group.bench_function(BenchmarkId::new("thread_policy_sync", count), |b| {
            b.iter(|| {
                black_box(
                    thread.summarize::<SyncTask>(black_box(Priority::Normal), black_box(count)),
                )
            });
        });

        verify_summary::<SyncTask, HybridRoutePolicy>(&hybrid, Priority::Normal, count);
        group.bench_function(BenchmarkId::new("hybrid_policy_sync", count), |b| {
            b.iter(|| {
                black_box(
                    hybrid.summarize::<SyncTask>(black_box(Priority::Normal), black_box(count)),
                )
            });
        });

        verify_summary::<SyncTask, ServerRoutePolicy>(&server, Priority::Normal, count);
        group.bench_function(BenchmarkId::new("server_policy_sync", count), |b| {
            b.iter(|| {
                black_box(
                    server.summarize::<SyncTask>(black_box(Priority::Normal), black_box(count)),
                )
            });
        });
    }

    group.finish();
}

fn criterion_config() -> Criterion {
    Criterion::default().without_plots()
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets =
        benchmark_thread_process_server_summary,
        benchmark_async_process_lanes,
        benchmark_policy_overhead
}
criterion_main!(benches);

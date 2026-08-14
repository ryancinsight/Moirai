#![expect(
    clippy::unwrap_used,
    reason = "test scope: failed precondition = test failure"
)]

use criterion::{black_box, BenchmarkId, Criterion};
use moirai::{
    FixedRemoteTask, Moirai, RemoteCapabilityToken, RoutedProcessTarget, RoutedServerTarget,
    SumU64Capability,
};
use moirai_core::Priority;
use moirai_executor::schedule::{
    AsyncLanesPerProcess, AsyncTask, HybridRoutePolicy, HybridRouter, ProcessCount, RouteTopology,
    SchedulerRoute, ServerCount, ServerId, ServerRoutePolicy, WorkerCount,
};
use moirai_transport::{
    process::{ProcessDropPolicy, ProcessOutcome, ProcessSpec, ProcessWaitPolicy},
    remote_task::{RemoteTaskId, RemoteTaskOperation, RemoteTaskOutput, RemoteTaskServer},
    route::{
        ProcessEndpoint, RouteAddressBook, RouteNamespace, RouteService, RoutedProcessTaskClient,
        RoutedRemoteTaskClient, ServerEndpoint,
    },
    RemoteAddress,
};
use std::time::{Duration, Instant};

const CHILD_MODE: &str = "MOIRAI_ROUTED_EXECUTION_CHILD";
const CHILD_PORT: &str = "MOIRAI_ROUTED_EXECUTION_PORT";
const CHILD_TASK_ID: &str = "MOIRAI_ROUTED_EXECUTION_TASK_ID";

fn topology(servers: usize) -> RouteTopology {
    RouteTopology::new(
        WorkerCount::new(4),
        ProcessCount::new(3),
        AsyncLanesPerProcess::new(2),
        ServerCount::new(servers),
    )
}

fn loopback_remote_address(service: &'static str) -> RemoteAddress {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    RemoteAddress {
        host: "127.0.0.1".to_string(),
        port,
        service: service.to_string(),
    }
}

fn address_book_for(server_address: &RemoteAddress) -> RouteAddressBook {
    RouteAddressBook::new(
        RouteNamespace::new("routed-execution-bench"),
        vec![ServerEndpoint::new(
            ServerId::new(0),
            server_address.host.clone(),
            server_address.port,
            RouteService::new(server_address.service.clone()),
        )],
    )
}

fn server_route_sequence(router: &HybridRouter<ServerRoutePolicy>) -> usize {
    (0..64)
        .find(|sequence| {
            matches!(
                router.select::<AsyncTask>(Priority::Critical, *sequence),
                SchedulerRoute::Server(_)
            )
        })
        .expect("benchmark topology must produce a server route")
}

fn process_route_sequence(
    router: &HybridRouter<HybridRoutePolicy>,
) -> (usize, moirai_executor::schedule::ProcessId) {
    (0..64)
        .find_map(
            |sequence| match router.select::<AsyncTask>(Priority::High, sequence) {
                SchedulerRoute::Process(route) => Some((sequence, route.process)),
                SchedulerRoute::Thread(_)
                | SchedulerRoute::Server(_)
                | SchedulerRoute::Accelerator(_) => None,
            },
        )
        .expect("benchmark topology must produce a process route")
}

fn run_server_route_once(iteration: u64) {
    let server_address = loopback_remote_address("moirai-routed-server-bench");
    let reply_address = loopback_remote_address("moirai-routed-server-reply");
    let router = HybridRouter::<ServerRoutePolicy>::new(topology(1));
    let sequence = server_route_sequence(&router);
    let task_id = RemoteTaskId::new(10_000 + iteration);
    let server = RemoteTaskServer::new(server_address.clone());
    let server_thread = std::thread::spawn(move || server.serve_one().unwrap());
    let client = RoutedRemoteTaskClient::<ServerRoutePolicy>::new(
        address_book_for(&server_address),
        reply_address,
    );

    let (route, result) = client
        .execute_selected::<AsyncTask>(
            &router,
            Priority::Critical,
            sequence,
            task_id,
            RemoteTaskOperation::SumU64(vec![13, 21, 34, iteration]),
        )
        .unwrap();

    assert!(matches!(route, SchedulerRoute::Server(_)));
    assert_eq!(server_thread.join().unwrap(), task_id);
    assert_eq!(
        result.output,
        RemoteTaskOutput::U64(68u64.wrapping_add(iteration))
    );
}

fn run_process_route_once(iteration: u64) {
    let server_address = loopback_remote_address("moirai-routed-process-bench");
    let reply_address = loopback_remote_address("moirai-routed-process-reply");
    let router = HybridRouter::<HybridRoutePolicy>::new(topology(0));
    let (sequence, process) = process_route_sequence(&router);
    let task_id = RemoteTaskId::new(20_000 + iteration);
    let spec = ProcessSpec::new(std::env::current_exe().unwrap())
        .env(CHILD_MODE, "1")
        .env(CHILD_PORT, server_address.port.to_string())
        .env(CHILD_TASK_ID, task_id.get().to_string());
    let endpoint = ProcessEndpoint::new(process, spec, server_address);
    let client = RoutedProcessTaskClient::<HybridRoutePolicy>::new(
        vec![endpoint],
        reply_address,
        ProcessDropPolicy::TerminateOnDrop,
        ProcessWaitPolicy::new(2_000, Duration::from_millis(1)),
    );

    let (route, output) = client
        .execute_selected::<AsyncTask>(
            &router,
            Priority::High,
            sequence,
            task_id,
            RemoteTaskOperation::SumU64(vec![55, 89, 144, iteration]),
        )
        .unwrap();

    assert!(matches!(route, SchedulerRoute::Process(_)));
    assert_eq!(output.result.task_id, task_id);
    assert_eq!(
        output.result.output,
        RemoteTaskOutput::U64(288u64.wrapping_add(iteration))
    );
    assert_eq!(output.status.outcome, ProcessOutcome::Succeeded);
    assert_eq!(output.status.code, Some(0));
}

fn run_public_server_route_once(runtime: &Moirai, iteration: u64) {
    let server_address = loopback_remote_address("moirai-public-routed-server-bench");
    let reply_address = loopback_remote_address("moirai-public-routed-server-reply");
    let router = HybridRouter::<ServerRoutePolicy>::new(topology(1));
    let sequence = server_route_sequence(&router);
    let task_id = RemoteTaskId::new(30_000 + iteration);
    let server = RemoteTaskServer::new(server_address.clone());
    let server_thread = std::thread::spawn(move || server.serve_one().unwrap());
    let target = RoutedServerTarget::new(address_book_for(&server_address), reply_address);
    let task = FixedRemoteTask::new(
        task_id,
        RemoteCapabilityToken::<SumU64Capability>::new(),
        vec![8, 13, 21, iteration],
    );

    let (route, result) = runtime
        .execute_routed_server_task::<AsyncTask, ServerRoutePolicy, _, _>(
            &router,
            Priority::Critical,
            sequence,
            target,
            task,
        )
        .unwrap();

    assert!(matches!(route, SchedulerRoute::Server(_)));
    assert_eq!(server_thread.join().unwrap(), task_id);
    assert_eq!(
        result.output,
        RemoteTaskOutput::U64(42u64.wrapping_add(iteration))
    );
}

fn run_public_process_route_once(runtime: &Moirai, iteration: u64) {
    let server_address = loopback_remote_address("moirai-public-routed-process-bench");
    let reply_address = loopback_remote_address("moirai-public-routed-process-reply");
    let router = HybridRouter::<HybridRoutePolicy>::new(topology(0));
    let (sequence, process) = process_route_sequence(&router);
    let task_id = RemoteTaskId::new(40_000 + iteration);
    let spec = ProcessSpec::new(std::env::current_exe().unwrap())
        .env(CHILD_MODE, "1")
        .env(CHILD_PORT, server_address.port.to_string())
        .env(CHILD_TASK_ID, task_id.get().to_string());
    let target = RoutedProcessTarget::new(
        vec![ProcessEndpoint::new(process, spec, server_address)],
        reply_address,
        ProcessDropPolicy::TerminateOnDrop,
        ProcessWaitPolicy::new(2_000, Duration::from_millis(1)),
    );
    let task = FixedRemoteTask::new(
        task_id,
        RemoteCapabilityToken::<SumU64Capability>::new(),
        vec![3, 5, 8, iteration],
    );

    let (route, output) = runtime
        .execute_routed_process_task::<AsyncTask, HybridRoutePolicy, _, _>(
            &router,
            Priority::High,
            sequence,
            target,
            task,
        )
        .unwrap();

    assert!(matches!(route, SchedulerRoute::Process(_)));
    assert_eq!(output.result.task_id, task_id);
    assert_eq!(
        output.result.output,
        RemoteTaskOutput::U64(16u64.wrapping_add(iteration))
    );
    assert_eq!(output.status.outcome, ProcessOutcome::Succeeded);
    assert_eq!(output.status.code, Some(0));
}

fn benchmark_routed_execution(c: &mut Criterion) {
    let public_runtime = Moirai::new().expect("public facade runtime must build");

    run_server_route_once(0);
    run_process_route_once(0);
    run_public_server_route_once(&public_runtime, 0);
    run_public_process_route_once(&public_runtime, 0);

    let mut group = c.benchmark_group("process_server_routed_execution");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(200));
    group.measurement_time(Duration::from_secs(1));

    group.bench_function(BenchmarkId::new("server_route_sum_u64", 1), |b| {
        b.iter_custom(|iterations| {
            let start = Instant::now();
            for iteration in 0..iterations {
                run_server_route_once(black_box(iteration));
            }
            start.elapsed()
        });
    });

    group.bench_function(BenchmarkId::new("process_route_sum_u64", 1), |b| {
        b.iter_custom(|iterations| {
            let start = Instant::now();
            for iteration in 0..iterations {
                run_process_route_once(black_box(iteration));
            }
            start.elapsed()
        });
    });

    group.bench_function(BenchmarkId::new("public_server_route_sum_u64", 1), |b| {
        b.iter_custom(|iterations| {
            let start = Instant::now();
            for iteration in 0..iterations {
                run_public_server_route_once(black_box(&public_runtime), black_box(iteration));
            }
            start.elapsed()
        });
    });

    group.bench_function(BenchmarkId::new("public_process_route_sum_u64", 1), |b| {
        b.iter_custom(|iterations| {
            let start = Instant::now();
            for iteration in 0..iterations {
                run_public_process_route_once(black_box(&public_runtime), black_box(iteration));
            }
            start.elapsed()
        });
    });

    group.finish();
    public_runtime.shutdown();
}

fn run_child_server() -> bool {
    if std::env::var_os(CHILD_MODE).is_none() {
        return false;
    }

    let port = std::env::var(CHILD_PORT)
        .expect("child server port must be provided")
        .parse::<u16>()
        .expect("child server port must be a u16");
    let expected_task = std::env::var(CHILD_TASK_ID)
        .expect("child task id must be provided")
        .parse::<u64>()
        .expect("child task id must be a u64");
    let server = RemoteTaskServer::new(RemoteAddress {
        host: "127.0.0.1".to_string(),
        port,
        service: "moirai-routed-process-bench".to_string(),
    });

    assert_eq!(
        server.serve_one().unwrap(),
        RemoteTaskId::new(expected_task)
    );
    true
}

fn criterion_config() -> Criterion {
    Criterion::default().without_plots()
}

fn main() {
    if run_child_server() {
        return;
    }

    let mut criterion = criterion_config().configure_from_args();
    benchmark_routed_execution(&mut criterion);
    criterion.final_summary();
}

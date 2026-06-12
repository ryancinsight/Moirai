use moirai_core::Priority;

use super::{
    AcceleratorCounts, AcceleratorRoutePolicy, AsyncLanesPerProcess, HybridRoutePolicy,
    HybridRouter, ProcessCount, RouteSummary, RouteTopology, SchedulerRoute, ServerCount,
    ServerRoutePolicy, ThreadRoutePolicy, WorkerCount,
};
use crate::schedule::{AsyncTask, BlockingTask, SyncTask};

fn topology() -> RouteTopology {
    RouteTopology::new(
        WorkerCount::new(4),
        ProcessCount::new(3),
        AsyncLanesPerProcess::new(2),
        ServerCount::new(2),
    )
}

#[test]
fn route_policies_are_zero_sized() {
    assert_eq!(core::mem::size_of::<ThreadRoutePolicy>(), 0);
    assert_eq!(core::mem::size_of::<HybridRoutePolicy>(), 0);
    assert_eq!(core::mem::size_of::<ServerRoutePolicy>(), 0);
    assert_eq!(core::mem::size_of::<AcceleratorRoutePolicy>(), 0);
}

#[test]
fn async_work_routes_to_process_async_lanes() {
    let router = HybridRouter::<HybridRoutePolicy>::new(topology());
    let mut observed = RouteSummary::default();

    for sequence in 0..32 {
        let route = router.select::<AsyncTask>(Priority::Normal, sequence);
        match route {
            SchedulerRoute::Process(route) => {
                assert!(route.async_lane.is_some());
                observed.process_routes += 1;
                observed.async_lane_routes += 1;
            }
            SchedulerRoute::Server(route) => {
                assert!(route.async_lane.is_some());
                observed.server_routes += 1;
                observed.async_lane_routes += 1;
            }
            SchedulerRoute::Thread(_) => {
                panic!("async hybrid route must leave thread-only path")
            }
            SchedulerRoute::Accelerator(_) => {
                panic!("hybrid policy must not produce accelerator routes")
            }
        }
    }

    assert_eq!(observed.total_routes(), 32);
    assert_eq!(observed.async_lane_routes, 32);
}

#[test]
fn sync_work_routes_across_thread_process_and_server_targets() {
    let router = HybridRouter::<HybridRoutePolicy>::new(topology());
    let summary = router.summarize::<SyncTask>(Priority::Normal, 128);

    assert!(summary.thread_routes > 0);
    assert!(summary.process_routes > 0);
    assert!(summary.server_routes > 0);
    assert_eq!(summary.async_lane_routes, 0);
    assert_eq!(summary.total_routes(), 128);
    assert_ne!(summary.checksum, 0);
}

#[test]
fn server_policy_increases_server_route_cadence() {
    let hybrid = HybridRouter::<HybridRoutePolicy>::new(topology());
    let server = HybridRouter::<ServerRoutePolicy>::new(topology());

    let hybrid_summary = hybrid.summarize::<BlockingTask>(Priority::High, 512);
    let server_summary = server.summarize::<BlockingTask>(Priority::High, 512);

    assert!(server_summary.server_routes > hybrid_summary.server_routes);
    assert_eq!(hybrid_summary.async_lane_routes, 0);
    assert_eq!(server_summary.async_lane_routes, 0);
}

#[test]
fn accelerator_policy_routes_metadata_across_device_kinds() {
    let topology = topology().with_accelerators(AcceleratorCounts::new(1, 2, 1, 1));
    let router = HybridRouter::<AcceleratorRoutePolicy>::new(topology);
    let summary = router.summarize::<SyncTask>(Priority::Normal, 128);

    assert_eq!(summary.total_routes(), 128);
    assert_eq!(summary.accelerator_routes, 18);
    assert_eq!(summary.cpu_routes, 3);
    assert_eq!(summary.gpu_routes, 8);
    assert_eq!(summary.tpu_routes, 4);
    assert_eq!(summary.npu_routes, 3);
    assert_eq!(summary.async_lane_routes, 0);
    assert_ne!(summary.checksum, 0);
}

#[test]
fn async_accelerator_metadata_retains_async_lane() {
    let topology = topology().with_accelerators(AcceleratorCounts::new(1, 1, 1, 1));
    let router = HybridRouter::<AcceleratorRoutePolicy>::new(topology);
    let summary = router.summarize::<AsyncTask>(Priority::Critical, 64);

    assert_eq!(summary.total_routes(), 64);
    assert_eq!(summary.accelerator_routes, 9);
    assert_eq!(summary.thread_routes, 0);
    assert_eq!(summary.async_lane_routes, 64);
    assert_eq!(
        summary.cpu_routes + summary.gpu_routes + summary.tpu_routes + summary.npu_routes,
        summary.accelerator_routes
    );
}

#[test]
fn thread_policy_stays_thread_local() {
    let router = HybridRouter::<ThreadRoutePolicy>::new(topology());
    let summary = router.summarize::<AsyncTask>(Priority::Critical, 64);

    assert_eq!(summary.thread_routes, 64);
    assert_eq!(summary.process_routes, 0);
    assert_eq!(summary.server_routes, 0);
    assert_eq!(summary.async_lane_routes, 0);
}

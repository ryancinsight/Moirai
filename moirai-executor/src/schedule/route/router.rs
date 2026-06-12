//! Static scheduler router.

use core::marker::PhantomData;

use moirai_core::Priority;

use super::{
    AcceleratorCounts, AcceleratorId, AcceleratorKind, AcceleratorRoute, AsyncLaneId, ProcessId,
    ProcessRoute, RoutePolicy, RouteSummary, RouteTopology, SchedulerRoute, ServerId, ServerRoute,
    ThreadId, ThreadRoute,
};
use crate::schedule::WorkClass;

/// Static hybrid scheduler router.
#[derive(Debug, Clone, Copy)]
pub struct HybridRouter<P: RoutePolicy> {
    topology: RouteTopology,
    _policy: PhantomData<P>,
}

impl<P: RoutePolicy> HybridRouter<P> {
    /// Construct a router for a topology and compile-time policy.
    #[inline]
    pub const fn new(topology: RouteTopology) -> Self {
        Self {
            topology,
            _policy: PhantomData,
        }
    }

    /// Return the router topology.
    #[inline]
    pub const fn topology(self) -> RouteTopology {
        self.topology
    }

    /// Select one concrete scheduler route.
    #[inline]
    pub fn select<C: WorkClass>(&self, priority: Priority, sequence: usize) -> SchedulerRoute {
        let topology = self.topology;
        let priority = priority_weight(priority);
        let route_key = sequence
            .wrapping_add(C::AFFINITY_OFFSET)
            .wrapping_add(priority);
        let thread = ThreadId::new(route_key % topology.worker_threads().get());
        let process = ProcessId::new(
            sequence
                .wrapping_div(topology.worker_threads().get())
                .wrapping_add(C::AFFINITY_OFFSET)
                .wrapping_add(priority)
                % topology.processes().get(),
        );
        let async_lane = async_lane::<C>(topology, sequence, process);

        if P::ENABLE_ACCELERATOR_ROUTES
            && topology.accelerators().total() != 0
            && route_key % P::ACCELERATOR_PERIOD.max(1) == 0
        {
            let accelerator_sequence = route_key / P::ACCELERATOR_PERIOD.max(1);
            let (kind, accelerator) =
                accelerator_target(topology.accelerators(), accelerator_sequence);
            return SchedulerRoute::Accelerator(AcceleratorRoute {
                kind,
                accelerator,
                process,
                thread,
                async_lane,
            });
        }

        if P::ENABLE_SERVER_ROUTES
            && topology.servers().get() != 0
            && route_key % P::SERVER_PERIOD.max(1) == 0
        {
            return SchedulerRoute::Server(ServerRoute {
                server: ServerId::new(route_key % topology.servers().get()),
                process,
                thread,
                async_lane,
            });
        }

        if P::ENABLE_PROCESS_ROUTES
            && topology.processes().get() > 1
            && (C::USES_ASYNC_LANE || route_key % P::PROCESS_PERIOD.max(1) == 0)
        {
            return SchedulerRoute::Process(ProcessRoute {
                process,
                thread,
                async_lane,
            });
        }

        SchedulerRoute::Thread(ThreadRoute { process, thread })
    }

    /// Summarize a deterministic route sequence for one work class.
    #[inline]
    pub fn summarize<C: WorkClass>(&self, priority: Priority, count: usize) -> RouteSummary {
        let mut summary = RouteSummary::default();
        for sequence in 0..count {
            summary.record(self.select::<C>(priority, sequence));
        }
        summary
    }
}

#[inline]
fn async_lane<C: WorkClass>(
    topology: RouteTopology,
    sequence: usize,
    process: ProcessId,
) -> Option<AsyncLaneId> {
    if C::USES_ASYNC_LANE {
        Some(AsyncLaneId::new(
            sequence.wrapping_add(process.get()) % topology.async_lanes_per_process().get(),
        ))
    } else {
        None
    }
}

#[inline]
fn accelerator_target(
    accelerators: AcceleratorCounts,
    accelerator_sequence: usize,
) -> (AcceleratorKind, AcceleratorId) {
    let target = accelerator_sequence % accelerators.total();
    if target < accelerators.cpu() {
        return (AcceleratorKind::Cpu, AcceleratorId::new(target));
    }
    let target = target - accelerators.cpu();
    if target < accelerators.gpu() {
        return (AcceleratorKind::Gpu, AcceleratorId::new(target));
    }
    let target = target - accelerators.gpu();
    if target < accelerators.tpu() {
        return (AcceleratorKind::Tpu, AcceleratorId::new(target));
    }
    (
        AcceleratorKind::Npu,
        AcceleratorId::new(target - accelerators.tpu()),
    )
}

#[inline]
fn priority_weight(priority: Priority) -> usize {
    match priority {
        Priority::Low => 0,
        Priority::Normal => 1,
        Priority::High => 2,
        Priority::Critical => 3,
    }
}

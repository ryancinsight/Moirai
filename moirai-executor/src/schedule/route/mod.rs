//! Static scheduler route topology for thread, process, and server targets.

use core::marker::PhantomData;

use moirai_core::Priority;

use super::class::WorkClass;

mod route_policy {
    pub trait Sealed {}
}

/// Compile-time scheduler route policy.
///
/// Implementors are zero-sized marker types. The router uses the associated
/// constants as structural branch selectors, so inactive routing families are
/// removed by monomorphization and dead-code elimination.
pub trait RoutePolicy: route_policy::Sealed + Copy + Default + Send + Sync + 'static {
    /// Whether route selection may produce process routes.
    const ENABLE_PROCESS_ROUTES: bool;

    /// Whether route selection may produce server routes.
    const ENABLE_SERVER_ROUTES: bool;

    /// Periodic process route cadence for non-async work classes.
    const PROCESS_PERIOD: usize;

    /// Periodic server route cadence.
    const SERVER_PERIOD: usize;
}

/// Route policy that keeps work local to scheduler threads.
#[derive(Debug, Clone, Copy, Default)]
pub struct ThreadRoutePolicy;

/// Route policy that mixes local threads, process routes, and server routes.
#[derive(Debug, Clone, Copy, Default)]
pub struct HybridRoutePolicy;

/// Route policy that gives server routing a higher cadence.
#[derive(Debug, Clone, Copy, Default)]
pub struct ServerRoutePolicy;

impl route_policy::Sealed for ThreadRoutePolicy {}
impl route_policy::Sealed for HybridRoutePolicy {}
impl route_policy::Sealed for ServerRoutePolicy {}

impl RoutePolicy for ThreadRoutePolicy {
    const ENABLE_PROCESS_ROUTES: bool = false;
    const ENABLE_SERVER_ROUTES: bool = false;
    const PROCESS_PERIOD: usize = 1;
    const SERVER_PERIOD: usize = 1;
}

impl RoutePolicy for HybridRoutePolicy {
    const ENABLE_PROCESS_ROUTES: bool = true;
    const ENABLE_SERVER_ROUTES: bool = true;
    const PROCESS_PERIOD: usize = 5;
    const SERVER_PERIOD: usize = 17;
}

impl RoutePolicy for ServerRoutePolicy {
    const ENABLE_PROCESS_ROUTES: bool = true;
    const ENABLE_SERVER_ROUTES: bool = true;
    const PROCESS_PERIOD: usize = 3;
    const SERVER_PERIOD: usize = 11;
}

/// Scheduler worker-thread identifier.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ThreadId(usize);

impl ThreadId {
    /// Construct a thread identifier.
    #[inline]
    pub const fn new(id: usize) -> Self {
        Self(id)
    }

    /// Return the underlying zero-based identifier.
    #[inline]
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Scheduler process identifier.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProcessId(usize);

impl ProcessId {
    /// Construct a process identifier.
    #[inline]
    pub const fn new(id: usize) -> Self {
        Self(id)
    }

    /// Return the underlying zero-based identifier.
    #[inline]
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Scheduler server identifier.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ServerId(usize);

impl ServerId {
    /// Construct a server identifier.
    #[inline]
    pub const fn new(id: usize) -> Self {
        Self(id)
    }

    /// Return the underlying zero-based identifier.
    #[inline]
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Async lane identifier inside a routed process.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct AsyncLaneId(usize);

impl AsyncLaneId {
    /// Construct an async lane identifier.
    #[inline]
    pub const fn new(id: usize) -> Self {
        Self(id)
    }

    /// Return the underlying zero-based identifier.
    #[inline]
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Number of local scheduler worker threads.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WorkerCount(usize);

impl WorkerCount {
    /// Construct a non-zero worker count.
    #[inline]
    pub const fn new(count: usize) -> Self {
        Self(if count == 0 { 1 } else { count })
    }

    /// Return the normalized count.
    #[inline]
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Number of process route targets.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessCount(usize);

impl ProcessCount {
    /// Construct a non-zero process count.
    #[inline]
    pub const fn new(count: usize) -> Self {
        Self(if count == 0 { 1 } else { count })
    }

    /// Return the normalized count.
    #[inline]
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Number of async lanes in each process target.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AsyncLanesPerProcess(usize);

impl AsyncLanesPerProcess {
    /// Construct a non-zero async-lane count.
    #[inline]
    pub const fn new(count: usize) -> Self {
        Self(if count == 0 { 1 } else { count })
    }

    /// Return the normalized count.
    #[inline]
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Number of server route targets.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServerCount(usize);

impl ServerCount {
    /// Construct a server count. Zero disables server route targets.
    #[inline]
    pub const fn new(count: usize) -> Self {
        Self(count)
    }

    /// Return the count.
    #[inline]
    pub const fn get(self) -> usize {
        self.0
    }
}

/// Static route topology used by [`HybridRouter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouteTopology {
    worker_threads: WorkerCount,
    processes: ProcessCount,
    async_lanes_per_process: AsyncLanesPerProcess,
    servers: ServerCount,
}

impl RouteTopology {
    /// Construct a route topology from validated count newtypes.
    #[inline]
    pub const fn new(
        worker_threads: WorkerCount,
        processes: ProcessCount,
        async_lanes_per_process: AsyncLanesPerProcess,
        servers: ServerCount,
    ) -> Self {
        Self {
            worker_threads,
            processes,
            async_lanes_per_process,
            servers,
        }
    }

    /// Return the local worker-thread count.
    #[inline]
    pub const fn worker_threads(self) -> WorkerCount {
        self.worker_threads
    }

    /// Return the process route target count.
    #[inline]
    pub const fn processes(self) -> ProcessCount {
        self.processes
    }

    /// Return the async-lane count per process.
    #[inline]
    pub const fn async_lanes_per_process(self) -> AsyncLanesPerProcess {
        self.async_lanes_per_process
    }

    /// Return the server route target count.
    #[inline]
    pub const fn servers(self) -> ServerCount {
        self.servers
    }
}

/// Local thread route.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ThreadRoute {
    /// Target process containing the thread.
    pub process: ProcessId,
    /// Target local scheduler thread.
    pub thread: ThreadId,
}

/// Process route with an optional async lane.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProcessRoute {
    /// Target process.
    pub process: ProcessId,
    /// Target scheduler thread inside the process.
    pub thread: ThreadId,
    /// Target async lane for async-capable work.
    pub async_lane: Option<AsyncLaneId>,
}

/// Server route with process, thread, and optional async-lane placement.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServerRoute {
    /// Target server.
    pub server: ServerId,
    /// Target process on the server.
    pub process: ProcessId,
    /// Target scheduler thread inside the process.
    pub thread: ThreadId,
    /// Target async lane for async-capable work.
    pub async_lane: Option<AsyncLaneId>,
}

/// Concrete route decision produced by [`HybridRouter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerRoute {
    /// Route remains inside the local scheduler worker set.
    Thread(ThreadRoute),
    /// Route crosses a process boundary.
    Process(ProcessRoute),
    /// Route crosses a server boundary.
    Server(ServerRoute),
}

/// Value summary of a route sequence.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RouteSummary {
    /// Number of local thread routes.
    pub thread_routes: usize,
    /// Number of process routes.
    pub process_routes: usize,
    /// Number of server routes.
    pub server_routes: usize,
    /// Number of routes with an async lane.
    pub async_lane_routes: usize,
    /// Deterministic checksum over route placements.
    pub checksum: usize,
}

impl RouteSummary {
    /// Return the total number of summarized route decisions.
    #[inline]
    pub const fn total_routes(self) -> usize {
        self.thread_routes + self.process_routes + self.server_routes
    }

    #[inline]
    fn record(&mut self, route: SchedulerRoute) {
        match route {
            SchedulerRoute::Thread(route) => {
                self.thread_routes += 1;
                self.checksum = mix_route_checksum(
                    self.checksum,
                    1,
                    0,
                    route.process.get(),
                    route.thread.get(),
                    None,
                );
            }
            SchedulerRoute::Process(route) => {
                self.process_routes += 1;
                if route.async_lane.is_some() {
                    self.async_lane_routes += 1;
                }
                self.checksum = mix_route_checksum(
                    self.checksum,
                    2,
                    0,
                    route.process.get(),
                    route.thread.get(),
                    route.async_lane,
                );
            }
            SchedulerRoute::Server(route) => {
                self.server_routes += 1;
                if route.async_lane.is_some() {
                    self.async_lane_routes += 1;
                }
                self.checksum = mix_route_checksum(
                    self.checksum,
                    3,
                    route.server.get(),
                    route.process.get(),
                    route.thread.get(),
                    route.async_lane,
                );
            }
        }
    }
}

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
        let thread = ThreadId(route_key % topology.worker_threads.get());
        let process = ProcessId(
            sequence
                .wrapping_div(topology.worker_threads.get())
                .wrapping_add(C::AFFINITY_OFFSET)
                .wrapping_add(priority)
                % topology.processes.get(),
        );
        let async_lane = async_lane::<C>(topology, sequence, process);

        if P::ENABLE_SERVER_ROUTES
            && topology.servers.get() != 0
            && route_key % P::SERVER_PERIOD.max(1) == 0
        {
            return SchedulerRoute::Server(ServerRoute {
                server: ServerId(route_key % topology.servers.get()),
                process,
                thread,
                async_lane,
            });
        }

        if P::ENABLE_PROCESS_ROUTES
            && topology.processes.get() > 1
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
        Some(AsyncLaneId(
            sequence.wrapping_add(process.get()) % topology.async_lanes_per_process.get(),
        ))
    } else {
        None
    }
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

#[inline]
fn mix_route_checksum(
    current: usize,
    route_kind: usize,
    server: usize,
    process: usize,
    thread: usize,
    async_lane: Option<AsyncLaneId>,
) -> usize {
    current
        .wrapping_mul(1_000_003)
        .wrapping_add(route_kind.wrapping_mul(97))
        .wrapping_add(server.wrapping_mul(31))
        .wrapping_add(process.wrapping_mul(17))
        .wrapping_add(thread.wrapping_mul(13))
        .wrapping_add(async_lane.map_or(0, |lane| lane.get().wrapping_add(1)))
}

#[cfg(test)]
mod tests {
    use moirai_core::Priority;

    use super::{
        AsyncLanesPerProcess, HybridRoutePolicy, HybridRouter, ProcessCount, RouteSummary,
        RouteTopology, SchedulerRoute, ServerCount, ServerRoutePolicy, ThreadRoutePolicy,
        WorkerCount,
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
    fn thread_policy_stays_thread_local() {
        let router = HybridRouter::<ThreadRoutePolicy>::new(topology());
        let summary = router.summarize::<AsyncTask>(Priority::Critical, 64);

        assert_eq!(summary.thread_routes, 64);
        assert_eq!(summary.process_routes, 0);
        assert_eq!(summary.server_routes, 0);
        assert_eq!(summary.async_lane_routes, 0);
    }
}

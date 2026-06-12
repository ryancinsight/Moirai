//! Unified executor scheduling.
//!
//! The scheduler in this module is the single dispatch engine for synchronous,
//! blocking, and asynchronous jobs. Workload shape is selected by zero-sized
//! marker types so call sites keep static dispatch while heterogeneous jobs are
//! stored at the executor boundary.

pub mod class;
pub mod job;
pub mod queue;
pub mod reduce;
pub mod route;
pub mod runtime;
pub mod wake;

pub use class::{AsyncTask, BlockingTask, SyncTask, WorkClass};
pub use route::{
    AcceleratorCounts, AcceleratorId, AcceleratorKind, AcceleratorRoute, AcceleratorRoutePolicy,
    AsyncLaneId, AsyncLanesPerProcess, HybridRoutePolicy, HybridRouter, ProcessCount, ProcessId,
    ProcessRoute, RoutePolicy, RouteSummary, RouteTopology, SchedulerRoute, ServerCount, ServerId,
    ServerRoute, ServerRoutePolicy, ThreadId, ThreadRoute, ThreadRoutePolicy, WorkerCount,
};
#[cfg(feature = "scheduler-diagnostics")]
pub use runtime::{
    ContendedWakeDecision, DiagnosticWakeDecision, EmptyWakeDecision, SaturatedWakeDecision,
};
pub use runtime::{ScheduleMetrics, SchedulerScope, ThreadScheduler};

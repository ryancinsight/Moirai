//! Static scheduler route topology for thread, process, server, and accelerator
//! targets.

mod decision;
mod ids;
mod policy;
mod router;
mod summary;
mod topology;

#[cfg(test)]
mod tests;

pub use decision::{AcceleratorRoute, ProcessRoute, SchedulerRoute, ServerRoute, ThreadRoute};
pub use ids::{
    AcceleratorCounts, AcceleratorId, AcceleratorKind, AsyncLaneId, AsyncLanesPerProcess,
    ProcessCount, ProcessId, ServerCount, ServerId, ThreadId, WorkerCount,
};
pub use policy::{
    AcceleratorRoutePolicy, HybridRoutePolicy, RoutePolicy, ServerRoutePolicy, ThreadRoutePolicy,
};
pub use router::HybridRouter;
pub use summary::RouteSummary;
pub use topology::RouteTopology;

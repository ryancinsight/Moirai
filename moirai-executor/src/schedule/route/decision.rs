//! Concrete scheduler route decisions.

use super::{AcceleratorId, AcceleratorKind, AsyncLaneId, ProcessId, ServerId, ThreadId};

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

/// Accelerator placement metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AcceleratorRoute {
    /// Target accelerator family.
    pub kind: AcceleratorKind,
    /// Target accelerator within the family.
    pub accelerator: AcceleratorId,
    /// Target process coordinating the device placement.
    pub process: ProcessId,
    /// Target scheduler thread coordinating the device placement.
    pub thread: ThreadId,
    /// Target async lane for async-capable accelerator work.
    pub async_lane: Option<AsyncLaneId>,
}

/// Concrete route decision produced by [`super::HybridRouter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerRoute {
    /// Route remains inside the local scheduler worker set.
    Thread(ThreadRoute),
    /// Route crosses a process boundary.
    Process(ProcessRoute),
    /// Route crosses a server boundary.
    Server(ServerRoute),
    /// Route targets accelerator placement metadata.
    Accelerator(AcceleratorRoute),
}

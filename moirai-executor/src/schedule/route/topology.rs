//! Static route topology definition.

use super::{AcceleratorCounts, AsyncLanesPerProcess, ProcessCount, ServerCount, WorkerCount};

/// Static route topology used by [`super::HybridRouter`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RouteTopology {
    worker_threads: WorkerCount,
    processes: ProcessCount,
    async_lanes_per_process: AsyncLanesPerProcess,
    servers: ServerCount,
    accelerators: AcceleratorCounts,
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
            accelerators: AcceleratorCounts::new(0, 0, 0, 0),
        }
    }

    /// Return a copy of this topology with accelerator route targets attached.
    #[inline]
    pub const fn with_accelerators(mut self, accelerators: AcceleratorCounts) -> Self {
        self.accelerators = accelerators;
        self
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

    /// Return accelerator route target counts.
    #[inline]
    pub const fn accelerators(self) -> AcceleratorCounts {
        self.accelerators
    }
}

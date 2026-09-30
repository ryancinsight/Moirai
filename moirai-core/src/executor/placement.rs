//! Worker placement policy.

/// Whether scheduler workers are confined to logical processors.
///
/// The policy is an opt-in startup contract. It changes what the runtime
/// enforces, and therefore what it may publish about worker placement.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum WorkerPlacement {
    /// Workers run wherever the operating system schedules them.
    ///
    /// The runtime claims no worker-to-processor or worker-to-node assignment,
    /// and the topology-aware victim tier stays inactive.
    #[default]
    Unbound,
    /// Worker `i` is confined to one logical processor for its whole life.
    ///
    /// Processors are taken from the detected topology in ascending id order,
    /// worker `i` receiving processor `i % processor_count`, so a pool larger
    /// than the machine shares processors. Construction fails closed: when the
    /// topology cannot be detected, or the operating system refuses any
    /// worker's binding, the scheduler is not returned and the error names the
    /// lowest-numbered worker that failed. On a host with more than one NUMA
    /// node the bound assignment is what the runtime publishes for
    /// topology-aware stealing.
    Pinned,
}

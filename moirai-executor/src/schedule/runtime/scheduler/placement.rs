//! Worker-to-processor pinning: planning, binding, and outcome collection.
//!
//! Under `WorkerPlacement::Pinned` each worker binds its own thread, so the
//! node table the scheduler publishes describes placement the operating system
//! accepted. Construction waits for every worker's outcome and refuses to
//! return a scheduler that holds a failed binding.

#[cfg(test)]
use std::cell::Cell;
use std::{sync::Arc, thread};

use moirai_core::error::{ExecutorError, ExecutorResult, PlacementFailure};
use themis::BindError;

use super::super::types::WorkerState;

/// One worker's planned processor and the NUMA node that processor belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Assignment {
    pub(super) processor: u32,
    pub(super) node: usize,
}

/// Bind function injected by tests; it receives the worker index and processor.
#[cfg(test)]
pub(in crate::schedule::runtime) type TestBinder =
    Arc<dyn Fn(usize, u32) -> Result<(), BindError> + Send + Sync>;

/// Test-only replacement for topology detection and the operating-system bind.
#[cfg(test)]
pub(in crate::schedule::runtime) struct PinProbe {
    /// Detected `(processor, node)` pairs, in any order.
    pub(in crate::schedule::runtime) pairs: Vec<(u32, usize)>,
    pub(in crate::schedule::runtime) binder: TestBinder,
    /// Dropped only after every scheduler worker has released shared state.
    pub(in crate::schedule::runtime) lifetime_owner: Box<dyn std::any::Any + Send + Sync>,
}

/// Placement request handed to one worker thread.
#[derive(Clone)]
pub(in crate::schedule::runtime) struct WorkerPin {
    #[cfg(test)]
    worker: usize,
    processor: u32,
    #[cfg(test)]
    binder: Option<TestBinder>,
}

impl WorkerPin {
    /// Confine the calling thread to the planned processor.
    pub(in crate::schedule::runtime) fn bind(&self) -> Result<(), PlacementFailure> {
        #[cfg(test)]
        if let Some(binder) = &self.binder {
            return binder(self.worker, self.processor).map_err(placement_failure);
        }
        themis::bind_current_thread(self.processor).map_err(placement_failure)
    }
}

/// The processor assignment of a `Pinned` worker set.
pub(super) struct Pinning {
    assignments: Box<[Assignment]>,
    #[cfg(test)]
    binder: Option<TestBinder>,
    #[cfg(test)]
    lifetime_owner: Cell<Option<Box<dyn std::any::Any + Send + Sync>>>,
}

impl Pinning {
    /// Plan from the processors the operating system reports.
    ///
    /// # Errors
    ///
    /// Returns [`ExecutorError::InvalidConfiguration`] when topology detection
    /// fails or reports no processor with a known NUMA node.
    pub(super) fn detect(worker_count: usize) -> ExecutorResult<Self> {
        let topology = themis::CpuTopology::detect().ok_or(ExecutorError::InvalidConfiguration)?;
        let pairs = topology
            .processor_node_pairs()
            .map(|(processor, node)| (processor, node.index()));
        Ok(Self {
            assignments: plan(pairs, worker_count)?,
            #[cfg(test)]
            binder: None,
            #[cfg(test)]
            lifetime_owner: Cell::new(None),
        })
    }

    /// Plan from a test probe instead of the host topology.
    #[cfg(test)]
    pub(super) fn from_probe(probe: PinProbe, worker_count: usize) -> ExecutorResult<Self> {
        Ok(Self {
            assignments: plan(probe.pairs.iter().copied(), worker_count)?,
            binder: Some(probe.binder),
            lifetime_owner: Cell::new(Some(probe.lifetime_owner)),
        })
    }

    /// The owner a test probe asked the scheduler to retain, once.
    #[cfg(test)]
    pub(super) fn take_lifetime_owner(&self) -> Option<Box<dyn std::any::Any + Send + Sync>> {
        self.lifetime_owner.take()
    }

    /// The NUMA node of every worker, as the table the scheduler publishes.
    pub(super) fn worker_numa_nodes(&self) -> Box<[Option<usize>]> {
        self.assignments
            .iter()
            .map(|assignment| Some(assignment.node))
            .collect()
    }

    /// The placement request for `worker`, which must be below the worker count.
    pub(super) fn worker_pin(&self, worker: usize) -> WorkerPin {
        WorkerPin {
            #[cfg(test)]
            worker,
            processor: self.assignments[worker].processor,
            #[cfg(test)]
            binder: self.binder.clone(),
        }
    }

    /// Wait for every worker to publish its binding outcome.
    ///
    /// # Errors
    ///
    /// Returns [`ExecutorError::WorkerPlacementFailed`] for the
    /// lowest-numbered worker whose binding failed, after all outcomes are in.
    pub(super) fn await_outcomes(&self, workers: &[Arc<WorkerState>]) -> ExecutorResult<()> {
        for worker in workers {
            while worker.placement.get().is_none() {
                thread::yield_now();
            }
        }
        for (worker, (state, assignment)) in workers.iter().zip(&self.assignments).enumerate() {
            if let Some(Err(cause)) = state.placement.get() {
                return Err(ExecutorError::WorkerPlacementFailed {
                    worker,
                    processor: assignment.processor,
                    cause: *cause,
                });
            }
        }
        Ok(())
    }
}

/// Assign processors to workers: worker `i` takes the `i % n`-th processor in
/// ascending id order.
///
/// # Errors
///
/// Returns [`ExecutorError::InvalidConfiguration`] when `pairs` is empty; no
/// placement is invented for a host that reports none.
fn plan(
    pairs: impl Iterator<Item = (u32, usize)>,
    worker_count: usize,
) -> ExecutorResult<Box<[Assignment]>> {
    let mut processors: Vec<Assignment> = pairs
        .map(|(processor, node)| Assignment { processor, node })
        .collect();
    processors.sort_unstable_by_key(|assignment| assignment.processor);
    if processors.is_empty() {
        return Err(ExecutorError::InvalidConfiguration);
    }
    Ok((0..worker_count)
        .map(|worker| processors[worker % processors.len()])
        .collect())
}

/// The one translation from the provider error to Moirai's contract.
fn placement_failure(error: BindError) -> PlacementFailure {
    match error {
        BindError::Unsupported => PlacementFailure::Unsupported,
        BindError::OutOfRange { .. } => PlacementFailure::OutOfRange,
        BindError::Os { code } => PlacementFailure::Os { code },
        // `BindError` is `#[non_exhaustive]`. A refusal this version does not
        // classify means no binding took effect, which is what `Unsupported`
        // reports; unreachable at the locked provider revision.
        _ => PlacementFailure::Unsupported,
    }
}

/// Clear the table unless at least two nodes are represented.
///
/// The same-node steal tier orders victims by node, so a single represented
/// node (or none) gives it nothing to distinguish.
pub(super) fn normalize_worker_numa_nodes(
    mut worker_numa_nodes: Box<[Option<usize>]>,
) -> Box<[Option<usize>]> {
    let mut represented_nodes = worker_numa_nodes.iter().copied().flatten();
    let has_multiple_nodes = represented_nodes
        .next()
        .is_some_and(|first| represented_nodes.any(|node| node != first));

    if !has_multiple_nodes {
        worker_numa_nodes.fill(None);
    }

    worker_numa_nodes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn processors(assignments: &[Assignment]) -> Vec<u32> {
        assignments
            .iter()
            .map(|assignment| assignment.processor)
            .collect()
    }

    #[test]
    fn locality_pass_requires_multiple_represented_nodes() {
        for assignments in [
            vec![None, None, None].into_boxed_slice(),
            vec![Some(0), Some(0), Some(0)].into_boxed_slice(),
            vec![None, Some(3), None].into_boxed_slice(),
        ] {
            assert_eq!(
                &*normalize_worker_numa_nodes(assignments),
                &[None, None, None]
            );
        }

        let multiple = vec![Some(3), None, Some(7), Some(3)].into_boxed_slice();
        assert_eq!(
            &*normalize_worker_numa_nodes(multiple),
            &[Some(3), None, Some(7), Some(3)]
        );
    }

    #[test]
    fn plan_sorts_by_processor_and_wraps_when_workers_exceed_processors() {
        let planned = plan([(5, 1), (2, 0), (9, 1)].into_iter(), 5)
            .expect("invariant: a non-empty topology plans");
        assert_eq!(processors(&planned), [2, 5, 9, 2, 5]);
        assert_eq!(
            planned.iter().map(|a| a.node).collect::<Vec<_>>(),
            [0, 1, 1, 0, 1]
        );
    }

    #[test]
    fn plan_of_an_empty_topology_is_an_invalid_configuration() {
        assert_eq!(
            plan(std::iter::empty(), 2).err(),
            Some(ExecutorError::InvalidConfiguration)
        );
    }

    #[test]
    fn provider_errors_map_onto_the_moirai_contract() {
        assert_eq!(
            placement_failure(BindError::Unsupported),
            PlacementFailure::Unsupported
        );
        assert_eq!(
            placement_failure(BindError::OutOfRange { processor: 40_000 }),
            PlacementFailure::OutOfRange
        );
        assert_eq!(
            placement_failure(BindError::Os { code: 22 }),
            PlacementFailure::Os { code: 22 }
        );
    }
}

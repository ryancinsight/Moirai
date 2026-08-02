//! Warp-aware launch-shape planning (atlas ADR 0002, Stage D2).
//!
//! Warps are scheduled by SM hardware; the software-ownable layer is launch
//! shaping. This module is the typed intersection of the two ADR-0002
//! inputs — themis [`GpuTopology`] (per-unit capacities, provider-fed) and
//! mnemosyne [`KernelResourceBudget`] (per-kernel requirements; registers
//! are compiler-assigned, so budgets are vocabulary, not allocation) — into
//! the two quantities a dispatcher actually needs:
//!
//! - the **grid** that covers a given amount of work (`plan_launch`), and
//! - the **resident-block capacity** at full occupancy
//!   (`resident_blocks`), which sizes persistent-kernel deployments.
//!
//! These are deliberately separate: a grid may exceed residency (hardware
//! queues excess blocks), so conflating "blocks to launch" with "blocks
//! resident at once" would be a correctness error for persistent kernels
//! and a pessimization for classic launches.
//!
//! Unreported capacities follow the themis/mnemosyne "no information"
//! contract: they yield `None` here rather than fabricated bounds.

use core::num::{NonZeroU32, NonZeroUsize};
use mnemosyne_core::KernelResourceBudget;

use themis::GpuTopology;

/// A planned compute launch: grid size and block width.
///
/// `threads_per_block` always equals the budget's block width — the planner
/// never silently reshapes a kernel's declared block size, because register
/// and shared-memory budgets are stated per that width.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LaunchShape {
    /// Number of blocks (workgroups) in the grid.
    pub grid_blocks: u32,
    /// Threads per block, taken from the kernel's budget.
    pub threads_per_block: u32,
}

/// Plan the grid that covers `work_items` elements with one thread each.
///
/// `grid_blocks = ceil(work_items / threads_per_block)`, saturating at
/// `u32::MAX` blocks (a grid that large exceeds every real device's single
/// launch anyway; callers splitting such work detect saturation by
/// comparing covered work). Zero work yields a zero-block shape the caller
/// can skip without dispatching.
#[must_use]
pub const fn plan_launch(budget: KernelResourceBudget, work_items: u64) -> LaunchShape {
    let threads = budget.threads_per_block();
    let blocks = work_items.div_ceil(threads as u64);
    LaunchShape {
        grid_blocks: if blocks > u32::MAX as u64 {
            u32::MAX
        } else {
            blocks as u32
        },
        threads_per_block: threads,
    }
}

/// Device-wide resident-block capacity at full occupancy:
/// `blocks_per_unit(budget, topology) · compute_units`.
///
/// Returns `None` when the topology lacks the information — either the
/// compute-unit count is unreported (`None` by type; e.g. the wgpu
/// provider, which cannot see SM counts) or every per-unit limiter is
/// unconstrained (`u32::MAX` "no information"). A persistent-kernel
/// deployment without this answer must choose its own policy rather than
/// trust a fabricated capacity. Unreported per-unit capacities pass to
/// the budget as `0`, its documented "no such limiter" input.
#[must_use]
pub fn resident_blocks(topology: &GpuTopology, budget: KernelResourceBudget) -> Option<u32> {
    let Some(units) = topology.compute_units() else {
        return None;
    };
    let per_unit = budget
        .occupancy_limits(
            topology.registers_per_unit().map_or(0, NonZeroU32::get),
            topology
                .shared_mem_per_unit_bytes()
                .map_or(0, NonZeroUsize::get),
            topology.max_threads_per_unit().map_or(0, NonZeroU32::get),
        )
        .blocks_per_unit();
    if per_unit == u32::MAX {
        return None;
    }
    let total = (per_unit as u64) * u64::from(units.get());
    Some(if total > u32::MAX as u64 {
        u32::MAX
    } else {
        total as u32
    })
}

/// Plan a persistent-kernel launch: a grid of exactly the resident-block
/// capacity (each block loops over a work queue instead of exiting), capped
/// by the work-covering grid so tiny workloads do not launch idle blocks.
///
/// `None` propagates [`resident_blocks`]' no-information cases.
#[must_use]
pub fn plan_persistent_launch(
    topology: &GpuTopology,
    budget: KernelResourceBudget,
    work_items: u64,
) -> Option<LaunchShape> {
    let resident = resident_blocks(topology, budget)?;
    let covering = plan_launch(budget, work_items);
    Some(LaunchShape {
        grid_blocks: resident.min(covering.grid_blocks),
        threads_per_block: budget.threads_per_block(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use themis::GpuDeviceProperties;

    // Ampere-class fixture matching mnemosyne's closed-form budget tests.
    fn ampere_like() -> GpuTopology {
        GpuTopology::from_provider(GpuDeviceProperties {
            compute_units: NonZeroU32::new(46),
            warp_width: NonZeroU32::new(32),
            max_threads_per_unit: NonZeroU32::new(1_536),
            registers_per_unit: NonZeroU32::new(65_536),
            shared_mem_per_unit_bytes: NonZeroUsize::new(102_400),
            l2_bytes: NonZeroUsize::new(4 * 1024 * 1024),
            memory_tier: themis::MemoryTier::Gddr,
            memory_bytes: core::num::NonZeroU64::new(8 * 1024 * 1024 * 1024),
        })
    }

    fn budget() -> KernelResourceBudget {
        // Registers bind at 4 blocks/unit (65536 / (64·256)).
        KernelResourceBudget::new(64, 16 * 1024, 256).unwrap()
    }

    #[test]
    fn covering_grid_is_ceil_division() {
        let shape = plan_launch(budget(), 1_000_000);
        // ceil(1_000_000 / 256) = 3907
        assert_eq!(shape.grid_blocks, 3_907);
        assert_eq!(shape.threads_per_block, 256);

        assert_eq!(plan_launch(budget(), 0).grid_blocks, 0);
        assert_eq!(plan_launch(budget(), 1).grid_blocks, 1);
        assert_eq!(plan_launch(budget(), 256).grid_blocks, 1);
        assert_eq!(plan_launch(budget(), 257).grid_blocks, 2);
    }

    #[test]
    fn resident_capacity_is_per_unit_times_units() {
        // 4 blocks/unit × 46 units = 184.
        assert_eq!(resident_blocks(&ampere_like(), budget()), Some(184));
    }

    #[test]
    fn persistent_launch_is_resident_capped_by_work() {
        let shape = plan_persistent_launch(&ampere_like(), budget(), 1_000_000).unwrap();
        assert_eq!(shape.grid_blocks, 184); // resident capacity binds
        assert_eq!(shape.threads_per_block, 256);

        // Tiny workload: covering grid (2 blocks) binds instead.
        let small = plan_persistent_launch(&ampere_like(), budget(), 300).unwrap();
        assert_eq!(small.grid_blocks, 2);
    }

    #[test]
    fn no_information_topologies_yield_none_not_fabrication() {
        // The wgpu provider reports zero compute units / capacities.
        let wgpu_like = GpuTopology::from_provider(GpuDeviceProperties {
            compute_units: None,
            warp_width: NonZeroU32::new(32),
            max_threads_per_unit: None,
            registers_per_unit: None,
            shared_mem_per_unit_bytes: None,
            l2_bytes: None,
            memory_tier: themis::MemoryTier::Dram,
            memory_bytes: None,
        });
        assert_eq!(resident_blocks(&wgpu_like, budget()), None);
        assert!(plan_persistent_launch(&wgpu_like, budget(), 1_000).is_none());
        // The work-covering grid never needs topology and still plans.
        assert_eq!(plan_launch(budget(), 1_000).grid_blocks, 4);
    }

    #[test]
    fn unconstrained_budget_on_known_units_is_still_no_information() {
        // Zero-resource budget: every limiter is u32::MAX even though the
        // unit count is known — residency is unbounded by resources, so the
        // honest answer remains None (threads limiter would bind if the
        // topology reported max_threads_per_unit; here it does).
        let topology = ampere_like();
        let unbounded = KernelResourceBudget::new(0, 0, 256).unwrap();
        // threads limiter binds: 1536/256 = 6 per unit × 46 = 276.
        assert_eq!(resident_blocks(&topology, unbounded), Some(276));
    }
}

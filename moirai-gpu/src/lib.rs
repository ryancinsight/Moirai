//! GPU launch-shape planning for Moirai.
//!
//! `moirai-gpu` owns the occupancy planner of atlas ADR 0002: it intersects a
//! themis [`GpuTopology`](themis::GpuTopology) with a mnemosyne
//! [`KernelResourceBudget`] into a [`LaunchShape`]. Device acquisition,
//! buffers, transfers, kernel dispatch, and synchronization belong to the
//! Hephaestus providers; Moirai does not depend on them (ADR 0041). A device
//! operation is scheduled on Moirai as an ordinary task.

#![deny(missing_docs)]

pub mod occupancy;

pub use mnemosyne_core::KernelResourceBudget;
pub use occupancy::{LaunchShape, plan_launch, plan_persistent_launch, resident_blocks};

/// Convenient imports for Moirai GPU planning consumers.
pub mod prelude {
    pub use crate::{
        KernelResourceBudget, LaunchShape, plan_launch, plan_persistent_launch, resident_blocks,
    };
}

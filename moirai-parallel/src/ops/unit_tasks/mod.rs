//! Whole-unit tasks sized by the bytes each unit moves (ADR 0059).
//!
//! A unit is a lane, row or matrix: a fixed run of elements that one closure
//! call transforms together. How many units a task carries, and whether the
//! pass runs in parallel at all, depend on the bytes a unit moves â€” including

mod layout;
mod many;
mod pair;
mod range;
mod single;
mod triple;

#[cfg(test)]
mod tests;

pub use layout::{UNIT_TASK_BYTES, units_per_task};
pub use many::for_each_unit_task_many_mut_with;
pub use pair::for_each_unit_task_pair_mut_with;
pub use range::for_each_unit_task_range_with;
pub use single::for_each_unit_task_mut_with;
pub use triple::for_each_unit_task_triple_mut_with;

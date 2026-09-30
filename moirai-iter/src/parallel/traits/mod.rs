//! Parallel iterator trait family.
//!
//! - `iterator`: [`ParallelIterator`], the item-stream contract and its adapter
//!   and terminal methods.
//! - `consumer`: [`Consumer`] and [`ParallelExtend`], the sink side of the
//!   drive protocol.
//! - `conversion`: [`IntoParallelIterator`] and [`IntoParallelRefIterator`],
//!   the source-side conversions.
//! - `folds`: the sequential and reassociated fold routines the terminal
//!   methods of [`ParallelIterator`] share.

mod consumer;
mod conversion;
mod folds;
mod iterator;

pub use consumer::{Consumer, ParallelExtend};
pub use conversion::{IntoParallelIterator, IntoParallelRefIterator};
pub use iterator::ParallelIterator;

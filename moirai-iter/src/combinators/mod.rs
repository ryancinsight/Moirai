//! Zero-cost iterator combinators for functional programming patterns
//!
//! This module provides advanced iterator combinators that enable functional
//! programming patterns with zero runtime overhead. All combinators are designed
//! to be inlined and optimized away by the compiler.

pub mod cycle;
pub mod ext;
pub mod flat_map;
pub mod inspect;
pub mod peekable;
pub mod scan;
pub mod skip;
pub mod skip_while;
pub mod step_by;

#[cfg(test)]
mod tests;

pub use cycle::Cycle;
pub use ext::CombinatorExt;
pub use flat_map::FlatMap;
pub use inspect::Inspect;
pub use peekable::Peekable;
pub use scan::Scan;
pub use skip::Skip;
pub use skip_while::SkipWhile;
pub use step_by::StepBy;

//! Unified data type abstraction for Moirai concurrency library.

mod base;
mod integer;
mod float;
mod context;

#[cfg(test)]
mod tests;

pub use base::Dtype;
pub use integer::IntegerDtype;
pub use float::FloatDtype;
pub use context::{ComputeContext, DefaultFloat, DefaultInt, DefaultUint};

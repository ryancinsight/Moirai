//! Unified data type abstraction for Moirai concurrency library.

mod base;
mod context;
mod float;
mod integer;

#[cfg(test)]
mod tests;

pub use base::Dtype;
pub use context::{ComputeContext, DefaultFloat, DefaultInt, DefaultUint};
pub use float::FloatDtype;
pub use integer::IntegerDtype;

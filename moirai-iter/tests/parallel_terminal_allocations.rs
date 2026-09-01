//! Allocation contracts for warmed parallel iterator terminals.
//!
//! The counting allocator is shared by operation-focused modules in this one
//! integration binary so the suite pays one link and one executor warmup.

#[path = "parallel_terminal_allocations/context.rs"]
mod context;
#[path = "parallel_terminal_allocations/map.rs"]
mod map;
#[path = "parallel_terminal_allocations/support.rs"]
mod support;
#[path = "parallel_terminal_allocations/terminal.rs"]
mod terminal;

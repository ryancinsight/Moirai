//! Owned Windows handles for atomic job assignment and bounded cleanup.
//! Native declarations follow Windows SDK 10.0.26100.0.

mod attributes;
mod command;
mod environment;
mod ffi;
mod job;
mod pipe;
mod process;
mod status;

pub(super) use process::Process;

//! Priority-aware worker queues.

mod priority;
mod steal;
mod worker;

#[cfg(test)]
mod tests;

pub(crate) use worker::{WorkerQueueOwner, WorkerQueues};

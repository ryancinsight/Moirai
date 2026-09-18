//! The [`IoReactor`]: its type definitions, lifecycle, fd registration,
//! readiness dispatch, and waker registration, each in its own leaf module.

mod event_dispatch;
mod fd_registration;
mod lifecycle;
mod types;
mod waker_registration;

pub use types::{FdInfo, FdKey, IoReactor};

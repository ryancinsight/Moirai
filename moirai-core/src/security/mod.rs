//! Security and safety validation for Moirai concurrency library.
//!
//! This module provides comprehensive security auditing, memory safety validation,
//! and vulnerability assessment capabilities for production deployment.

mod auditor;
mod config;
mod limiter;

#[cfg(test)]
mod tests;

pub use auditor::{Report, SecurityAuditor};
pub use config::{SecurityConfig, SecurityEvent, SecurityLevel};

//! Security and safety validation for Moirai concurrency library.
//!
//! This module provides comprehensive security auditing, memory safety validation,
//! and vulnerability assessment capabilities for production deployment.

mod config;
mod limiter;
mod auditor;

#[cfg(test)]
mod tests;

pub use config::{SecurityLevel, SecurityEvent, SecurityConfig};
pub use auditor::{SecurityAuditor, Report};

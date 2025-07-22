//! Security and safety validation for Moirai concurrency library.
//!
//! This module provides comprehensive security auditing, memory safety validation,
//! and vulnerability assessment capabilities for production deployment.

use crate::{TaskId, Priority, error::ExecutorError};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, atomic::{AtomicU64, AtomicBool, Ordering}},
    time::{Duration, Instant, SystemTime},
};

/// Security audit levels for different deployment environments.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SecurityLevel {
    /// Development environment - minimal security checks
    Development,
    /// Testing environment - moderate security validation
    Testing,
    /// Staging environment - comprehensive security checks
    Staging,
    /// Production environment - maximum security validation
    Production,
}

/// Security audit event types.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SecurityEvent {
    /// Task spawning with security context
    TaskSpawn { 
        /// The ID of the spawned task
        task_id: TaskId, 
        /// The priority of the spawned task
        priority: Priority, 
        /// When the event occurred
        timestamp: SystemTime 
    },
    /// Memory allocation beyond normal bounds
    MemoryAnomalous { 
        /// Size of the allocation in bytes
        size: usize, 
        /// Location where the allocation occurred
        location: String, 
        /// When the event occurred
        timestamp: SystemTime 
    },
    /// Potential race condition detected
    RaceCondition { 
        /// Description of the potential race condition
        description: String, 
        /// When the event occurred
        timestamp: SystemTime 
    },
    /// Resource exhaustion detected
    ResourceExhaustion { 
        /// Name of the exhausted resource
        resource: String, 
        /// Current usage level
        current: u64, 
        /// Maximum allowed limit
        limit: u64, 
        /// When the event occurred
        timestamp: SystemTime 
    },
}

/// Security audit configuration.
#[derive(Debug, Clone)]
pub struct SecurityConfig {
    /// Security level for this deployment
    pub level: SecurityLevel,
    /// Maximum memory allocation size before triggering audit
    pub max_allocation_size: usize,
    /// Maximum number of tasks per second before triggering audit
    pub max_task_spawn_rate: u64,
    /// Enable memory safety validation
    pub enable_memory_validation: bool,
    /// Enable race condition detection
    pub enable_race_detection: bool,
    /// Audit log retention period
    pub audit_retention: Duration,
}

impl Default for SecurityConfig {
    fn default() -> Self {
        Self {
            level: SecurityLevel::Development,
            max_allocation_size: 1024 * 1024 * 1024, // 1GB
            max_task_spawn_rate: 10_000, // 10k tasks/sec
            enable_memory_validation: true,
            enable_race_detection: true,
            audit_retention: Duration::from_secs(7 * 24 * 3600), // 7 days
        }
    }
}

impl SecurityConfig {
    /// Create a production-ready security configuration.
    pub fn production() -> Self {
        Self {
            level: SecurityLevel::Production,
            max_allocation_size: 512 * 1024 * 1024, // 512MB
            max_task_spawn_rate: 5_000, // 5k tasks/sec
            enable_memory_validation: true,
            enable_race_detection: true,
            audit_retention: Duration::from_secs(30 * 24 * 3600), // 30 days
        }
    }
}

/// Security auditor for monitoring and validating system security.
pub struct SecurityAuditor {
    config: SecurityConfig,
    events: Arc<Mutex<Vec<SecurityEvent>>>,
    task_spawn_counter: AtomicU64,
    last_spawn_reset: Arc<Mutex<Instant>>,
    enabled: AtomicBool,
}

impl SecurityAuditor {
    /// Create a new security auditor with the specified configuration.
    pub fn new(config: SecurityConfig) -> Self {
        Self {
            config,
            events: Arc::new(Mutex::new(Vec::new())),
            task_spawn_counter: AtomicU64::new(0),
            last_spawn_reset: Arc::new(Mutex::new(Instant::now())),
            enabled: AtomicBool::new(true),
        }
    }
    
    /// Enable or disable the security auditor.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }
    
    /// Check if the security auditor is enabled.
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }
    
    /// Audit a task spawn operation.
    pub fn audit_task_spawn(&self, task_id: TaskId, priority: Priority) -> Result<(), ExecutorError> {
        if !self.is_enabled() {
            return Ok(());
        }
        
        // Rate limiting check
        let current_count = self.task_spawn_counter.fetch_add(1, Ordering::Relaxed);
        let mut last_reset = self.last_spawn_reset.lock().unwrap();
        let now = Instant::now();
        
        if now.duration_since(*last_reset) >= Duration::from_secs(1) {
            self.task_spawn_counter.store(0, Ordering::Relaxed);
            *last_reset = now;
        } else if current_count > self.config.max_task_spawn_rate {
            self.record_event(SecurityEvent::ResourceExhaustion {
                resource: "task_spawn_rate".to_string(),
                current: current_count,
                limit: self.config.max_task_spawn_rate,
                timestamp: SystemTime::now(),
            });
            return Err(ExecutorError::ResourceExhausted("Task spawn rate exceeded".to_string()));
        }
        
        // Record the task spawn event
        self.record_event(SecurityEvent::TaskSpawn {
            task_id,
            priority,
            timestamp: SystemTime::now(),
        });
        
        Ok(())
    }
    
    /// Audit a memory allocation operation.
    pub fn audit_memory_allocation(&self, size: usize, location: &str) -> Result<(), ExecutorError> {
        if !self.is_enabled() || !self.config.enable_memory_validation {
            return Ok(());
        }
        
        // Check allocation size limits
        if size > self.config.max_allocation_size {
            self.record_event(SecurityEvent::MemoryAnomalous {
                size,
                location: location.to_string(),
                timestamp: SystemTime::now(),
            });
            return Err(ExecutorError::ResourceExhausted(
                format!("Memory allocation {} bytes exceeds limit {}", size, self.config.max_allocation_size)
            ));
        }
        
        Ok(())
    }
    
    /// Audit for potential race conditions.
    pub fn audit_race_condition(&self, description: &str) {
        if !self.is_enabled() || !self.config.enable_race_detection {
            return;
        }
        
        self.record_event(SecurityEvent::RaceCondition {
            description: description.to_string(),
            timestamp: SystemTime::now(),
        });
    }
    
    /// Record a security event.
    fn record_event(&self, event: SecurityEvent) {
        let mut events = self.events.lock().unwrap();
        events.push(event);
        
        // Clean up old events based on retention policy
        let cutoff = SystemTime::now() - self.config.audit_retention;
        events.retain(|event| {
            let timestamp = match event {
                SecurityEvent::TaskSpawn { timestamp, .. } => *timestamp,
                SecurityEvent::MemoryAnomalous { timestamp, .. } => *timestamp,
                SecurityEvent::RaceCondition { timestamp, .. } => *timestamp,
                SecurityEvent::ResourceExhaustion { timestamp, .. } => *timestamp,
            };
            timestamp > cutoff
        });
    }
    
    /// Get all security events.
    pub fn get_events(&self) -> Vec<SecurityEvent> {
        let events = self.events.lock().unwrap();
        events.clone()
    }
    
    /// Generate a security audit report.
    pub fn generate_report(&self) -> SecurityReport {
        let events = self.events.lock().unwrap();
        let total_events = events.len();
        
        let mut event_counts = HashMap::new();
        for event in events.iter() {
            let event_type = match event {
                SecurityEvent::TaskSpawn { .. } => "TaskSpawn",
                SecurityEvent::MemoryAnomalous { .. } => "MemoryAnomalous",
                SecurityEvent::RaceCondition { .. } => "RaceCondition",
                SecurityEvent::ResourceExhaustion { .. } => "ResourceExhaustion",
            };
            *event_counts.entry(event_type.to_string()).or_insert(0) += 1;
        }
        
        SecurityReport {
            config: self.config.clone(),
            total_events,
            event_counts,
            generated_at: SystemTime::now(),
        }
    }
}

/// Security audit report.
#[derive(Debug, Clone)]
pub struct SecurityReport {
    /// Configuration used for this audit
    pub config: SecurityConfig,
    /// Total number of security events recorded
    pub total_events: usize,
    /// Count of each type of security event
    pub event_counts: HashMap<String, usize>,
    /// When this report was generated
    pub generated_at: SystemTime,
}

impl SecurityReport {
    /// Check if the system passes security validation.
    pub fn is_secure(&self) -> bool {
        // No resource exhaustion events
        self.event_counts.get("ResourceExhaustion").unwrap_or(&0) == &0
    }
    
    /// Get security score (0-100, higher is better).
    pub fn security_score(&self) -> u8 {
        let critical_events = *self.event_counts.get("ResourceExhaustion").unwrap_or(&0);
        let warning_events = *self.event_counts.get("MemoryAnomalous").unwrap_or(&0) +
                            *self.event_counts.get("RaceCondition").unwrap_or(&0);
        
        if critical_events > 0 {
            0 // Critical security issues
        } else if self.total_events > 10 && warning_events > (self.total_events / 10) {
            50 // Many warning events (>10% of total)
        } else if warning_events > 0 {
            80 // Some warning events
        } else {
            100 // No security issues
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_security_config_defaults() {
        let config = SecurityConfig::default();
        assert_eq!(config.level, SecurityLevel::Development);
        assert!(config.max_allocation_size > 0);
        assert!(config.max_task_spawn_rate > 0);
    }
    
    #[test]
    fn test_security_auditor_basic() {
        let auditor = SecurityAuditor::new(SecurityConfig::default());
        assert!(auditor.is_enabled());
        
        // Test task spawn audit
        let task_id = TaskId::new(1);
        let result = auditor.audit_task_spawn(task_id, Priority::Normal);
        assert!(result.is_ok());
        
        // Check that event was recorded
        let events = auditor.get_events();
        assert!(!events.is_empty());
    }
    
    #[test]
    fn test_memory_allocation_audit() {
        let auditor = SecurityAuditor::new(SecurityConfig::production());
        
        // Normal allocation should pass
        let result = auditor.audit_memory_allocation(1024, "test_location");
        assert!(result.is_ok());
        
        // Large allocation should fail
        let result = auditor.audit_memory_allocation(1024 * 1024 * 1024, "test_location");
        assert!(result.is_err());
    }
    
    #[test]
    fn test_security_report() {
        let auditor = SecurityAuditor::new(SecurityConfig::production());
        
        // Generate some events
        let _ = auditor.audit_task_spawn(TaskId::new(1), Priority::Normal);
        auditor.audit_race_condition("test race condition");
        
        let report = auditor.generate_report();
        assert!(report.total_events > 0);
        assert!(report.event_counts.contains_key("TaskSpawn"));
        assert!(report.event_counts.contains_key("RaceCondition"));
        
        // Should be secure with only normal events
        assert!(report.is_secure());
        
        // Debug the security score calculation
        let score = report.security_score();
        println!("Security score: {}, Total events: {}, Event counts: {:?}", 
                 score, report.total_events, report.event_counts);
        
        // With only 2 events (1 TaskSpawn, 1 RaceCondition), and 1 warning event,
        // the score should be 80 (some warning events)
        assert!(score >= 80);
    }
}
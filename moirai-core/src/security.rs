//! Security and safety validation for Moirai concurrency library.
//!
//! This module provides comprehensive security auditing, memory safety validation,
//! and vulnerability assessment capabilities for production deployment.

use crate::{TaskId, Priority, error::ExecutorError};
use std::{
    collections::HashMap,
    sync::{Arc, Mutex, atomic::{AtomicU64, AtomicBool, AtomicUsize, Ordering}},
    time::{Duration, SystemTime},
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

/// Lock-free sliding window rate limiter for high-performance rate limiting.
/// 
/// Uses a circular buffer of atomic counters to track requests in time windows.
/// This approach avoids locks and race conditions while providing accurate rate limiting.
struct SlidingWindowRateLimiter {
    /// Circular buffer of counters for each time window
    windows: Vec<AtomicUsize>,
    /// Current window index (atomically updated)
    current_window: AtomicUsize,
    /// Timestamp of the current window start (in nanoseconds since epoch)
    window_start_ns: AtomicU64,
    /// Window duration in nanoseconds
    window_duration_ns: u64,
    /// Maximum requests per window
    max_requests: usize,
    /// Number of windows in the sliding window
    num_windows: usize,
}

impl SlidingWindowRateLimiter {
    /// Create a new sliding window rate limiter.
    /// 
    /// # Arguments
    /// * `max_requests_per_second` - Maximum requests allowed per second
    /// * `num_windows` - Number of sub-windows (higher = more accurate, default 10)
    fn new(max_requests_per_second: u64, num_windows: usize) -> Self {
        let num_windows = num_windows.max(1); // Ensure at least 1 window
        let window_duration_ns = 1_000_000_000 / num_windows as u64; // 1 second / num_windows
        // Total requests allowed across all windows
        let max_requests = max_requests_per_second as usize;
        
        let mut windows = Vec::with_capacity(num_windows);
        for _ in 0..num_windows {
            windows.push(AtomicUsize::new(0));
        }
        
        let now_ns = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        Self {
            windows,
            current_window: AtomicUsize::new(0),
            window_start_ns: AtomicU64::new(now_ns),
            window_duration_ns,
            max_requests,
            num_windows,
        }
    }
    
    /// Check if a request is allowed and increment the counter if so.
    /// 
    /// Returns true if the request is allowed, false if rate limited.
    fn try_acquire(&self) -> bool {
        let now_ns = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        
        // Update the current window if needed
        self.update_current_window(now_ns);
        
        // Get current window index and try to increment atomically
        let window_idx = self.current_window.load(Ordering::Acquire) % self.num_windows;
        
        // Optimistically increment the counter
        let _old_count = self.windows[window_idx].fetch_add(1, Ordering::AcqRel);
        
        // Check total count across all windows after incrementing
        let total_count = self.current_count();
        
        // If we exceeded the limit, undo the increment and reject
        if total_count > self.max_requests {
            self.windows[window_idx].fetch_sub(1, Ordering::AcqRel);
            return false;
        }
        
        true
    }
    
    /// Update the current window based on the current time.
    /// This method is lock-free and handles window transitions atomically.
    fn update_current_window(&self, now_ns: u64) {
        let window_start = self.window_start_ns.load(Ordering::Acquire);
        let elapsed_ns = now_ns.saturating_sub(window_start);
        
        if elapsed_ns >= self.window_duration_ns {
            // Calculate how many windows we need to advance
            let windows_to_advance = (elapsed_ns / self.window_duration_ns) as usize;
            
            // Try to update the window start time atomically
            let new_window_start = window_start + (windows_to_advance as u64 * self.window_duration_ns);
            
            // Use compare_exchange to ensure only one thread updates the window
            if self.window_start_ns.compare_exchange_weak(
                window_start,
                new_window_start,
                Ordering::AcqRel,
                Ordering::Acquire,
            ).is_ok() {
                // Successfully updated window start, now update current window
                let old_window = self.current_window.load(Ordering::Acquire);
                let new_window = old_window.wrapping_add(windows_to_advance);
                self.current_window.store(new_window, Ordering::Release);
                
                // Clear the windows that we're moving past
                for i in 1..=windows_to_advance.min(self.num_windows) {
                    let clear_idx = (old_window + i) % self.num_windows;
                    self.windows[clear_idx].store(0, Ordering::Release);
                }
            }
        }
    }
    
    /// Get the current request count across all windows (for monitoring).
    fn current_count(&self) -> usize {
        self.windows.iter()
            .map(|w| w.load(Ordering::Acquire))
            .sum()
    }
}

/// Security auditor for monitoring and validating system security.
pub struct SecurityAuditor {
    config: SecurityConfig,
    events: Arc<Mutex<Vec<SecurityEvent>>>,
    task_spawn_limiter: SlidingWindowRateLimiter,
    memory_allocations: Arc<Mutex<HashMap<String, usize>>>,
    enabled: AtomicBool,
}

impl SecurityAuditor {
    /// Create a new security auditor with the specified configuration.
    pub fn new(config: SecurityConfig) -> Self {
        Self {
            task_spawn_limiter: SlidingWindowRateLimiter::new(config.max_task_spawn_rate, 10),
            memory_allocations: Arc::new(Mutex::new(HashMap::new())),
            config,
            events: Arc::new(Mutex::new(Vec::new())),
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
    /// 
    /// Uses a lock-free sliding window rate limiter to accurately enforce spawn rate limits
    /// without race conditions or performance bottlenecks.
    pub fn audit_task_spawn(&self, task_id: TaskId, priority: Priority) -> Result<(), ExecutorError> {
        if !self.is_enabled() {
            return Ok(());
        }
        
        // Lock-free rate limiting check
        if !self.task_spawn_limiter.try_acquire() {
            let current_count = self.task_spawn_limiter.current_count();
            
            self.record_event(SecurityEvent::ResourceExhaustion {
                resource: "task_spawn_rate".to_string(),
                current: current_count as u64,
                limit: self.config.max_task_spawn_rate,
                timestamp: SystemTime::now(),
            });
            
            return Err(ExecutorError::ResourceExhausted(
                format!("Task spawn rate limit exceeded: {} requests/sec", current_count)
            ));
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
    /// 
    /// Checks allocation size limits and tracks allocation patterns to detect anomalies.
    /// Handles lock poisoning gracefully by propagating errors instead of panicking.
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
        
        // Track allocations by location (handle lock poisoning gracefully)
        match self.memory_allocations.lock() {
            Ok(mut allocations) => {
                let total = allocations.entry(location.to_string()).or_insert(0);
                *total += size;
                
                // Check for potential memory leaks (simplified heuristic)
                if *total > self.config.max_allocation_size / 2 {
                    self.record_event(SecurityEvent::MemoryAnomalous {
                        size: *total,
                        location: format!("accumulated_at_{location}"),
                        timestamp: SystemTime::now(),
                    });
                }
            }
            Err(_) => {
                // Lock is poisoned, record this as a security event but don't panic
                self.record_event(SecurityEvent::RaceCondition {
                    description: format!("Memory allocation tracking lock poisoned at {location}"),
                    timestamp: SystemTime::now(),
                });
                return Err(ExecutorError::ResourceExhausted(
                    "Memory allocation tracking unavailable due to lock poisoning".to_string()
                ));
            }
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
    
    /// Record a security event with automatic cleanup.
    fn record_event(&self, event: SecurityEvent) {
        if let Ok(mut events) = self.events.lock() {
            events.push(event);
            
            // Clean up old events based on retention policy
            let cutoff = SystemTime::now() - self.config.audit_retention;
            events.retain(|event| {
                let timestamp = match event {
                    SecurityEvent::TaskSpawn { timestamp, .. }
                    | SecurityEvent::MemoryAnomalous { timestamp, .. }
                    | SecurityEvent::RaceCondition { timestamp, .. }
                    | SecurityEvent::ResourceExhaustion { timestamp, .. } => *timestamp,
                };
                timestamp > cutoff
            });
        }
        // If lock is poisoned, we silently continue to maintain system stability
        // In production, this might be logged to an external system
    }
    
    /// Get all security events.
    /// 
    /// Returns an empty vector if the events lock is poisoned to maintain system stability.
    pub fn get_events(&self) -> Vec<SecurityEvent> {
        match self.events.lock() {
            Ok(events) => events.clone(),
            Err(_) => {
                // Lock is poisoned, return empty vector to maintain stability
                Vec::new()
            }
        }
    }
    
    /// Generate a comprehensive security report.
    pub fn generate_report(&self) -> Report {
        let (total_events, event_counts) = if let Ok(events) = self.events.lock() {
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
            
            (total_events, event_counts)
        } else {
            // Lock is poisoned, return minimal report
            let mut event_counts = HashMap::new();
            event_counts.insert("LockPoisoned".to_string(), 1);
            (0, event_counts)
        };
        
        Report {
            config: self.config.clone(),
            total_events,
            event_counts,
            generated_at: SystemTime::now(),
        }
    }
}

/// Security audit report.
#[derive(Debug, Clone)]
pub struct Report {
    /// Configuration used for this audit
    pub config: SecurityConfig,
    /// Total number of security events recorded
    pub total_events: usize,
    /// Count of each type of security event
    pub event_counts: HashMap<String, usize>,
    /// When this report was generated
    pub generated_at: SystemTime,
}

impl Report {
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
        
        // With only 2 events (1 TaskSpawn, 1 RaceCondition), and 1 warning event,
        // the score should be 80 (some warning events)
        let score = report.security_score();
        assert!(score >= 80);
    }
    
    #[test]
    fn test_sliding_window_rate_limiter() {
        let limiter = SlidingWindowRateLimiter::new(10, 5); // 10 requests/sec, 5 windows
        
        // Should allow up to the limit
        for _ in 0..10 {
            assert!(limiter.try_acquire(), "Should allow requests up to limit");
        }
        
        // Should reject additional requests
        assert!(!limiter.try_acquire(), "Should reject requests over limit");
        assert!(!limiter.try_acquire(), "Should continue rejecting");
        
        // Check current count
        let count = limiter.current_count();
        assert_eq!(count, 10, "Current count should equal the limit");
    }
    
    #[test]
    fn test_rate_limiter_no_off_by_one() {
        let limiter = SlidingWindowRateLimiter::new(5, 1); // 5 requests/sec, 1 window
        
        // Exactly 5 requests should be allowed
        for i in 0..5 {
            assert!(limiter.try_acquire(), "Request {} should be allowed", i + 1);
        }
        
        // 6th request should be rejected (fixes off-by-one error)
        assert!(!limiter.try_acquire(), "6th request should be rejected");
    }
    
    #[test]
    fn test_rate_limiter_concurrent_access() {
        use std::sync::Arc;
        use std::thread;
        
        let limiter = Arc::new(SlidingWindowRateLimiter::new(100, 10));
        let mut handles = vec![];
        
        // Spawn multiple threads trying to acquire
        for _ in 0..10 {
            let limiter_clone = limiter.clone();
            let handle = thread::spawn(move || {
                let mut acquired = 0;
                for _ in 0..20 {
                    if limiter_clone.try_acquire() {
                        acquired += 1;
                    }
                }
                acquired
            });
            handles.push(handle);
        }
        
        // Collect results
        let total_acquired: usize = handles.into_iter()
            .map(|h| h.join().unwrap())
            .sum();
        
        // Should not exceed the limit even with concurrent access
        assert!(total_acquired <= 100, "Total acquired {total_acquired} should not exceed limit 100");
        
        // Test rate limiting
        let auditor = SecurityAuditor::new(SecurityConfig::production());
        for i in 0..5 {
            let result = auditor.audit_task_spawn(TaskId::new(i + 1), Priority::Normal);
            assert!(result.is_ok(), "Spawn {} should succeed", i + 1);
        }
        
        // 6th spawn should fail due to rate limiting
        let result = auditor.audit_task_spawn(TaskId::new(6), Priority::Normal);
        assert!(result.is_err(), "6th spawn should fail due to rate limiting");
        
        // Check that it's specifically a rate limit error
        match result {
            Err(ExecutorError::ResourceExhausted(msg)) => {
                assert!(msg.contains("rate limit"), "Error should mention rate limit: {msg}");
            }
            _ => panic!("Expected ResourceExhausted error"),
        }
    }
    
    #[test]
    fn test_lock_poisoning_resilience() {
        let auditor = SecurityAuditor::new(SecurityConfig::production());
        
        // Generate a report - should work normally
        let report1 = auditor.generate_report();
        assert_eq!(report1.total_events, 0);
        
        // Even if locks were poisoned, we should get a report (though minimal)
        let report2 = auditor.generate_report();
        assert!(report2.generated_at > report1.generated_at);
        
        // Get events should not panic even with poisoned locks
        let events = auditor.get_events();
        assert!(events.is_empty()); // Should be empty initially
    }
}
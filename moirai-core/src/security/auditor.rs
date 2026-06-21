use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::{error::ExecutorError, Priority, TaskId};
use super::config::{
    SecurityConfig, SecurityEvent, DEFAULT_RATE_LIMITER_WINDOWS, MAX_REPRESENTABLE_UNIX_NANOS,
};
use super::limiter::SlidingWindowRateLimiter;

/// Security auditor for monitoring and enforcing security policies.
#[allow(clippy::module_name_repetitions)]
pub struct SecurityAuditor {
    config: SecurityConfig,
    events: Arc<Mutex<Vec<SecurityEvent>>>,
    task_spawn_limiter: SlidingWindowRateLimiter,
    memory_allocations: Arc<Mutex<HashMap<String, usize>>>,
    enabled: AtomicBool,
    last_report_unix_ns: AtomicU64,
}

impl SecurityAuditor {
    /// Creates a new security auditor with the given configuration.
    ///
    /// # Arguments
    /// * `config` - Security configuration settings
    ///
    /// # Returns
    /// A new security auditor instance
    #[must_use]
    pub fn new(config: SecurityConfig) -> Self {
        Self {
            task_spawn_limiter: SlidingWindowRateLimiter::new(
                config.max_task_spawn_rate,
                DEFAULT_RATE_LIMITER_WINDOWS,
            ),
            memory_allocations: Arc::new(Mutex::new(HashMap::new())),
            config,
            events: Arc::new(Mutex::new(Vec::new())),
            enabled: AtomicBool::new(true),
            last_report_unix_ns: AtomicU64::new(0),
        }
    }

    /// Enable or disable the security auditor.
    pub fn set_enabled(&self, enabled: bool) {
        self.enabled.store(enabled, Ordering::Relaxed);
    }

    /// Check if the security auditor is enabled.
    #[must_use]
    pub fn is_enabled(&self) -> bool {
        self.enabled.load(Ordering::Relaxed)
    }

    /// Audits a task spawn request for security compliance.
    ///
    /// # Arguments
    /// * `task_id` - The unique identifier for the task being spawned
    /// * `priority` - The requested priority level for the task
    ///
    /// # Returns
    /// `Ok(())` if the spawn request is approved, `Err` if denied.
    ///
    /// # Errors
    /// Returns `ExecutorError` in the following cases:
    /// - `ResourceExhausted` if rate limits are exceeded
    /// - `SecurityViolation` if the request violates security policies
    /// - `InvalidConfiguration` if security settings are malformed
    pub fn audit_task_spawn(
        &self,
        task_id: TaskId,
        priority: Priority,
    ) -> Result<(), ExecutorError> {
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

            return Err(ExecutorError::ResourceExhausted(format!(
                "Task spawn rate limit exceeded: {current_count} requests/sec"
            )));
        }

        // Record the task spawn event
        self.record_event(SecurityEvent::TaskSpawn {
            task_id,
            priority,
            timestamp: SystemTime::now(),
        });

        Ok(())
    }

    /// Audits a memory allocation request.
    ///
    /// # Arguments
    /// * `size` - The size of the memory allocation in bytes
    /// * `location` - A string identifying the allocation location
    ///
    /// # Returns
    /// `Ok(())` if the allocation is approved, `Err` if denied.
    ///
    /// # Errors
    /// Returns `ExecutorError` in the following cases:
    /// - `ResourceExhausted` if memory limits are exceeded
    /// - `SecurityViolation` if the allocation pattern is suspicious
    /// - `SystemError` if internal tracking fails
    pub fn audit_memory_allocation(
        &self,
        size: usize,
        location: &str,
    ) -> Result<(), ExecutorError> {
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
            return Err(ExecutorError::ResourceExhausted(format!(
                "Memory allocation {} bytes exceeds limit {}",
                size, self.config.max_allocation_size
            )));
        }

        // Track allocations by location (handle lock poisoning gracefully)
        if let Ok(mut allocations) = self.memory_allocations.lock() {
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
        } else {
            // Lock is poisoned, record this as a security event but don't panic
            self.record_event(SecurityEvent::RaceCondition {
                description: format!("Memory allocation tracking lock poisoned at {location}"),
                timestamp: SystemTime::now(),
            });
            return Err(ExecutorError::ResourceExhausted(
                "Memory allocation tracking unavailable due to lock poisoning".to_string(),
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
    }

    /// Get all security events.
    ///
    /// Returns an empty vector if the events lock is poisoned to maintain system stability.
    #[must_use]
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
    #[must_use]
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
            generated_at: self.next_report_time(),
        }
    }

    fn next_report_time(&self) -> SystemTime {
        let observed_now = unix_nanos(SystemTime::now());

        loop {
            let previous = self.last_report_unix_ns.load(Ordering::Acquire);
            let next = observed_now.max(previous.saturating_add(1));

            if self
                .last_report_unix_ns
                .compare_exchange(previous, next, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return UNIX_EPOCH + Duration::from_nanos(next);
            }
        }
    }
}

fn unix_nanos(time: SystemTime) -> u64 {
    time.duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO)
        .as_nanos()
        .min(u128::from(MAX_REPRESENTABLE_UNIX_NANOS)) as u64
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
    /// Checks if the current security state is considered secure.
    ///
    /// # Returns
    /// `true` if the system is in a secure state, `false` otherwise
    #[must_use]
    pub fn is_secure(&self) -> bool {
        // No resource exhaustion events
        self.event_counts.get("ResourceExhaustion").unwrap_or(&0) == &0
    }

    /// Calculates a security score based on current metrics.
    ///
    /// # Returns
    /// A security score from 0-100, where 100 is most secure
    #[must_use]
    pub fn security_score(&self) -> u8 {
        let critical_events = *self.event_counts.get("ResourceExhaustion").unwrap_or(&0);
        let warning_events = *self.event_counts.get("MemoryAnomalous").unwrap_or(&0)
            + *self.event_counts.get("RaceCondition").unwrap_or(&0);

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

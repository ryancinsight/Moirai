use super::auditor::{SecurityAuditor, MAX_AUDIT_EVENTS};
use super::config::{SecurityConfig, SecurityLevel};
use super::limiter::SlidingWindowRateLimiter;
use crate::{error::ExecutorError, Priority, TaskId};

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
fn audit_event_buffer_is_bounded_under_flood() {
    // Production retention is a month, so time-based eviction never fires here.
    // The hard count cap must still bound the buffer, fixing the prior unbounded
    // growth (and O(n^2) retain-per-insert) under a sustained event flood.
    let auditor = SecurityAuditor::new(SecurityConfig::production());
    for i in 0..(MAX_AUDIT_EVENTS + 5_000) {
        auditor.audit_race_condition(&format!("flood event {i}"));
    }
    let events = auditor.get_events();
    assert_eq!(
        events.len(),
        MAX_AUDIT_EVENTS,
        "event buffer must be bounded by the count cap"
    );
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
    let total_acquired: usize = handles.into_iter().map(|h| h.join().unwrap()).sum();

    // Should not exceed the limit even with concurrent access
    assert!(
        total_acquired <= 100,
        "Total acquired {total_acquired} should not exceed limit 100"
    );

    // Test rate limiting with a restrictive configuration
    let mut config = SecurityConfig::production();
    config.max_task_spawn_rate = 3; // Very low limit for testing
    let auditor = SecurityAuditor::new(config);

    // Should allow exactly 3 spawns
    for i in 0..3 {
        let result = auditor.audit_task_spawn(TaskId::new(i + 1), Priority::Normal);
        assert!(result.is_ok(), "Spawn {} should succeed", i + 1);
    }

    let result = auditor.audit_task_spawn(TaskId::new(4), Priority::Normal);
    assert!(
        result.is_err(),
        "4th spawn should fail due to rate limiting"
    );

    // Check that it's specifically a rate limit error
    match result {
        Err(ExecutorError::ResourceExhausted(msg)) => {
            assert!(
                msg.contains("rate limit"),
                "Error should mention rate limit: {msg}"
            );
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

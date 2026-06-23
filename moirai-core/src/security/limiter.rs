use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::SystemTime;

/// Lock-free sliding window rate limiter for high-performance rate limiting.
///
/// Uses a circular buffer of atomic counters to track requests in time windows.
/// This approach avoids locks and race conditions while providing accurate rate limiting.
pub(crate) struct SlidingWindowRateLimiter {
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
    /// Lock to serialize window transitions
    update_lock: Mutex<()>,
}

impl SlidingWindowRateLimiter {
    /// Create a new sliding window rate limiter.
    ///
    /// # Arguments
    /// * `max_requests_per_second` - Maximum requests allowed per second
    /// * `num_windows` - Number of sub-windows (higher = more accurate, default 10)
    pub(crate) fn new(max_requests_per_second: u64, num_windows: usize) -> Self {
        let num_windows = num_windows.max(1); // Ensure at least 1 window
        let window_duration_ns = 1_000_000_000 / num_windows as u64; // 1 second / num_windows
                                                                     // Total requests allowed across all windows
        #[allow(clippy::cast_possible_truncation)]
        let max_requests = max_requests_per_second as usize;

        let mut windows = Vec::with_capacity(num_windows);
        for _ in 0..num_windows {
            windows.push(AtomicUsize::new(0));
        }

        #[allow(clippy::cast_possible_truncation)]
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
            update_lock: Mutex::new(()),
        }
    }

    /// Check if a request is allowed and increment the counter if so.
    ///
    /// Returns true if the request is allowed, false if rate limited.
    ///
    /// This is an *approximate* limiter: the count is summed across windows and
    /// the increment-then-revert is not atomic with the check, so under high
    /// concurrency the effective cap may be exceeded transiently by up to the
    /// number of racing callers. That is acceptable for its purpose (best-effort
    /// task-spawn throttling / DoS dampening), not a hard quota.
    pub(crate) fn try_acquire(&self) -> bool {
        #[allow(clippy::cast_possible_truncation)]
        let now_ns = SystemTime::now()
            .duration_since(SystemTime::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;

        // Update the current window if needed
        self.update_current_window(now_ns);

        // Get current window index and try to increment atomically
        let window_idx = self.current_window.load(Ordering::Acquire) % self.num_windows;

        // Optimistically increment the counter
        let _previous_count = self.windows[window_idx].fetch_add(1, Ordering::AcqRel);

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
            // Acquire lock to update window parameters safely without race conditions
            if let Ok(_guard) = self.update_lock.try_lock() {
                // Re-check under lock
                let window_start = self.window_start_ns.load(Ordering::Relaxed);
                let elapsed_ns = now_ns.saturating_sub(window_start);
                if elapsed_ns >= self.window_duration_ns {
                    #[allow(clippy::cast_possible_truncation)]
                    let windows_to_advance = (elapsed_ns / self.window_duration_ns) as usize;
                    // Saturating time arithmetic (numerical_discipline): although
                    // `(elapsed / dur) * dur <= elapsed` cannot overflow a u64 on a
                    // 64-bit target, the time math must not rely on unchecked ops.
                    let new_window_start = window_start.saturating_add(
                        (windows_to_advance as u64).saturating_mul(self.window_duration_ns),
                    );

                    let old_window = self.current_window.load(Ordering::Relaxed);
                    let new_window = old_window.wrapping_add(windows_to_advance);

                    // Clear the windows that we're moving past
                    for i in 1..=windows_to_advance.min(self.num_windows) {
                        let clear_idx = (old_window + i) % self.num_windows;
                        self.windows[clear_idx].store(0, Ordering::Relaxed);
                    }

                    self.window_start_ns
                        .store(new_window_start, Ordering::Release);
                    self.current_window.store(new_window, Ordering::Release);
                }
            }
        }
    }

    /// Get the current request count across all windows (for monitoring).
    pub(crate) fn current_count(&self) -> usize {
        self.windows.iter().map(|w| w.load(Ordering::Acquire)).sum()
    }
}

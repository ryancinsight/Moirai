use crate::{Duration, Instant};
use num_traits::ToPrimitive;

fn nanoseconds_as_f64(value: u128) -> f64 {
    // Every u128 lies within f64's finite exponent range. The integer may round, as it did in
    // the original estimator, but cannot make the conversion unavailable.
    value.to_f64().expect("u128 is finite in f64")
}

fn work_items_as_f64(value: usize) -> f64 {
    value.to_f64().expect("usize is finite in f64")
}

fn saturated_work_count(value: f64) -> usize {
    // Match Rust's saturating float-to-unsigned conversion: truncate finite in-range values,
    // clamp positive overflow to the maximum, and map negative values and NaN to zero.
    if value.is_nan() || value <= 0.0 {
        0
    } else {
        value.to_usize().unwrap_or(usize::MAX)
    }
}

/// Limits the amount of time spent on a certain type of work in a cycle
///
/// The limiter works dynamically: For a sampled subset of cycles it measures
/// the time that is approximately required for fulfilling 1 work item, and
/// calculates the amount of allowed work items per cycle.
/// The estimates are smoothed over all cycles where the exact duration is measured.
///
/// In cycles where no measurement is performed the previously determined work limit
/// is used.
///
/// For the limiter the exact definition of a work item does not matter.
/// It could for example track the amount of transmitted bytes per cycle,
/// or the amount of transmitted datagrams per cycle.
/// It will however work best if the required time to complete a work item is
/// constant.
#[derive(Debug)]
pub(crate) struct WorkLimiter {
    /// Whether to measure the required work time, or to use the previous estimates
    mode: Mode,
    /// The current cycle number
    cycle: u16,
    /// The time the cycle started - only used in measurement mode
    start_time: Option<Instant>,
    /// How many work items have been completed in the cycle
    completed: usize,
    /// The amount of work items which are allowed for a cycle
    allowed: usize,
    /// The desired cycle time
    desired_cycle_time: Duration,
    /// The estimated and smoothed time per work item in nanoseconds
    smoothed_time_per_work_item_nanos: f64,
}

impl WorkLimiter {
    pub(crate) fn new(desired_cycle_time: Duration) -> Self {
        Self {
            mode: Mode::Measure,
            cycle: 0,
            start_time: None,
            completed: 0,
            allowed: 0,
            desired_cycle_time,
            smoothed_time_per_work_item_nanos: 0.0,
        }
    }

    /// Starts one work cycle
    pub(crate) fn start_cycle(&mut self, now: impl Fn() -> Instant) {
        self.completed = 0;
        if let Mode::Measure = self.mode {
            self.start_time = Some(now());
        }
    }

    /// Returns whether more work can be performed inside the `desired_cycle_time`
    ///
    /// Requires that previous work was tracked using `record_work`.
    pub(crate) fn allow_work(&mut self, now: impl Fn() -> Instant) -> bool {
        match self.mode {
            Mode::Measure => self
                .start_time
                .is_some_and(|start| (now() - start) < self.desired_cycle_time),
            Mode::HistoricData => self.completed < self.allowed,
        }
    }

    /// Records that `work` additional work items have been completed inside the cycle
    ///
    /// Must be called between `start_cycle` and `finish_cycle`.
    pub(crate) fn record_work(&mut self, work: usize) {
        self.completed += work;
    }

    /// Finishes one work cycle
    ///
    /// For cycles where the exact duration is measured this will update the estimates
    /// for the time per work item and the limit of allowed work items per cycle.
    /// The estimate is updated using the same exponential averaging (smoothing)
    /// mechanism which is used for determining QUIC path rtts: The last value is
    /// weighted by 1/8, and the previous average by 7/8.
    pub(crate) fn finish_cycle(&mut self, now: impl Fn() -> Instant) {
        // If no work was done in the cycle drop the measurement, it won't be useful
        if self.completed == 0 {
            return;
        }

        if let Mode::Measure = self.mode {
            // A measurement cannot be derived without a cycle start. In particular, do not
            // introduce a made-up zero-duration sample after a caller skips `start_cycle`.
            let Some(start) = self.start_time else {
                return;
            };
            let elapsed = now() - start;

            let time_per_work_item_nanos =
                nanoseconds_as_f64(elapsed.as_nanos()) / work_items_as_f64(self.completed);

            // Calculate the time per work item. We set this to at least 1ns to avoid
            // dividing by 0 when calculating the allowed amount of work items.
            self.smoothed_time_per_work_item_nanos = if self.allowed == 0 {
                // Initial estimate
                time_per_work_item_nanos
            } else {
                // Smoothed estimate
                (7.0 * self.smoothed_time_per_work_item_nanos + time_per_work_item_nanos) / 8.0
            }
            .max(1.0);

            // Allow at least 1 work item in order to make progress
            self.allowed = saturated_work_count(
                nanoseconds_as_f64(self.desired_cycle_time.as_nanos())
                    / self.smoothed_time_per_work_item_nanos,
            )
            .max(1);
            self.start_time = None;
        }

        self.cycle = self.cycle.wrapping_add(1);
        self.mode = match self.cycle % SAMPLING_INTERVAL {
            0 => Mode::Measure,
            _ => Mode::HistoricData,
        };
    }
}

/// We take a measurement sample once every `SAMPLING_INTERVAL` cycles
const SAMPLING_INTERVAL: u16 = 256;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    Measure,
    HistoricData,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::RefCell;

    #[test]
    fn missing_cycle_start_cannot_admit_work_or_update_sample() {
        let now = Instant::now();
        let mut limiter = WorkLimiter::new(Duration::from_millis(500));
        assert!(!limiter.allow_work(|| now));
        limiter.record_work(3);
        limiter.finish_cycle(|| now);
        assert_eq!(limiter.mode, Mode::Measure);
        assert_eq!(limiter.allowed, 0);
        assert_eq!(limiter.smoothed_time_per_work_item_nanos, 0.0);
        assert!(limiter.start_time.is_none());

        limiter.start_cycle(|| now);
        assert!(limiter.allow_work(|| now));
    }

    #[test]
    fn work_count_conversion_preserves_truncation_and_saturation() {
        assert_eq!(saturated_work_count(f64::NAN), 0);
        assert_eq!(saturated_work_count(f64::NEG_INFINITY), 0);
        assert_eq!(saturated_work_count(-0.5), 0);
        assert_eq!(saturated_work_count(12.75), 12);
        assert_eq!(saturated_work_count(f64::INFINITY), usize::MAX);
        assert_eq!(
            saturated_work_count(nanoseconds_as_f64(u128::MAX)),
            usize::MAX
        );
    }

    #[test]
    fn limit_work() {
        const CYCLE_TIME: Duration = Duration::from_millis(500);
        const BATCH_WORK_ITEMS: usize = 12;
        const BATCH_TIME: Duration = Duration::from_millis(100);

        let expected_initial_batches =
            usize::try_from(CYCLE_TIME.as_nanos() / BATCH_TIME.as_nanos())
                .expect("the fixed test ratio fits usize");
        let expected_allowed_work_items = expected_initial_batches * BATCH_WORK_ITEMS;

        let mut limiter = WorkLimiter::new(CYCLE_TIME);
        reset_time();

        // The initial cycle is measuring
        limiter.start_cycle(get_time);
        let mut initial_batches = 0;
        while limiter.allow_work(get_time) {
            limiter.record_work(BATCH_WORK_ITEMS);
            advance_time(BATCH_TIME);
            initial_batches += 1;
        }
        limiter.finish_cycle(get_time);

        assert_eq!(initial_batches, expected_initial_batches);
        assert_eq!(limiter.allowed, expected_allowed_work_items);
        let initial_time_per_work_item = limiter.smoothed_time_per_work_item_nanos;

        // The next cycles are using historic data
        const BATCH_SIZES: [usize; 4] = [1, 2, 3, 5];
        for &batch_size in &BATCH_SIZES {
            limiter.start_cycle(get_time);
            let mut allowed_work = 0;
            while limiter.allow_work(get_time) {
                limiter.record_work(batch_size);
                allowed_work += batch_size;
            }
            limiter.finish_cycle(get_time);

            assert_eq!(allowed_work, expected_allowed_work_items);
        }

        // After `SAMPLING_INTERVAL`, we get into measurement mode again
        for _ in 0..(usize::from(SAMPLING_INTERVAL) - BATCH_SIZES.len() - 1) {
            limiter.start_cycle(get_time);
            limiter.record_work(1);
            limiter.finish_cycle(get_time);
        }

        // We now do more work per cycle, and expect the estimate of allowed
        // work items to go up
        const BATCH_WORK_ITEMS_2: usize = 96;
        let time_per_work_items_2_nanos = nanoseconds_as_f64(CYCLE_TIME.as_nanos())
            / work_items_as_f64(expected_initial_batches * BATCH_WORK_ITEMS_2);

        let expected_updated_time_per_work_item =
            (initial_time_per_work_item * 7.0 + time_per_work_items_2_nanos) / 8.0;
        let expected_updated_allowed_work_items = saturated_work_count(
            nanoseconds_as_f64(CYCLE_TIME.as_nanos()) / expected_updated_time_per_work_item,
        );

        limiter.start_cycle(get_time);
        let mut initial_batches = 0;
        while limiter.allow_work(get_time) {
            limiter.record_work(BATCH_WORK_ITEMS_2);
            advance_time(BATCH_TIME);
            initial_batches += 1;
        }
        limiter.finish_cycle(get_time);

        assert_eq!(initial_batches, expected_initial_batches);
        assert_eq!(limiter.allowed, expected_updated_allowed_work_items);
    }

    thread_local! {
        /// Mocked time
        pub static TIME: RefCell<Instant> = RefCell::new(Instant::now());
    }

    fn reset_time() {
        TIME.with(|t| {
            *t.borrow_mut() = Instant::now();
        })
    }

    fn get_time() -> Instant {
        TIME.with(|t| *t.borrow())
    }

    fn advance_time(duration: Duration) {
        TIME.with(|t| {
            *t.borrow_mut() += duration;
        })
    }
}

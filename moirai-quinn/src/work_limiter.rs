use crate::{Duration, Instant};
use eunomia::{convert::IntegerTarget, F64};

fn rounded_nanoseconds(value: u128) -> f64 {
    if let Ok(count) = usize::try_from(value) {
        return F64::from_count(count).0;
    }

    // Duration::as_nanos is u128, while Eunomia's count conversion accepts usize. Construct
    // the IEEE binary64 normal form directly so the entire u128 range rounds exactly as the
    // original integer-to-float conversion did (nearest, ties to even), on 32- and 64-bit hosts.
    let width = u128::BITS - value.leading_zeros();
    let mut exponent = width - 1;
    let mut significand = if width > f64::MANTISSA_DIGITS {
        let discarded_bits = width - f64::MANTISSA_DIGITS;
        let kept = value >> discarded_bits;
        let remainder = value & ((1_u128 << discarded_bits) - 1);
        let halfway = 1_u128 << (discarded_bits - 1);
        kept + u128::from(remainder > halfway || (remainder == halfway && kept & 1 == 1))
    } else {
        value << (f64::MANTISSA_DIGITS - width)
    };
    if significand == (1_u128 << f64::MANTISSA_DIGITS) {
        significand >>= 1;
        exponent += 1;
    }
    let fraction = u64::try_from(significand & ((1_u128 << (f64::MANTISSA_DIGITS - 1)) - 1))
        .expect("binary64 fraction has 52 bits");
    let exponent_bits = u64::from(exponent + 1023) << (f64::MANTISSA_DIGITS - 1);
    f64::from_bits(exponent_bits | fraction)
}

fn rounded_work_items(value: usize) -> f64 {
    F64::from_count(value).0
}

fn saturated_work_count(value: f64) -> usize {
    usize::from_truncated(value)
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
                rounded_nanoseconds(elapsed.as_nanos()) / rounded_work_items(self.completed);

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
                rounded_nanoseconds(self.desired_cycle_time.as_nanos())
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
            saturated_work_count(rounded_nanoseconds(u128::MAX)),
            usize::MAX
        );
    }

    #[test]
    fn wide_nanosecond_counts_match_independent_decimal_rounding() {
        let precise = 1_u128 << f64::MANTISSA_DIGITS;
        let edges = [
            0,
            1,
            precise - 1,
            precise,
            precise + 1,
            precise + 3,
            1_u128 << 64,
            (1_u128 << 64) + (1_u128 << 11),
            u128::MAX - 1,
            u128::MAX,
        ];
        let mut state = 0x8d3f_5b79_2a6e_c413_f109_8642_b5d7_0ca1_u128;
        for value in edges.into_iter().chain((0..1024).map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 43;
            state
        })) {
            let oracle = value
                .to_string()
                .parse::<f64>()
                .expect("a decimal u128 is a valid finite binary64 input");
            assert_eq!(
                rounded_nanoseconds(value).to_bits(),
                oracle.to_bits(),
                "{value}"
            );
        }
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
        let time_per_work_items_2_nanos = rounded_nanoseconds(CYCLE_TIME.as_nanos())
            / rounded_work_items(expected_initial_batches * BATCH_WORK_ITEMS_2);

        let expected_updated_time_per_work_item =
            (initial_time_per_work_item * 7.0 + time_per_work_items_2_nanos) / 8.0;
        let expected_updated_allowed_work_items = saturated_work_count(
            rounded_nanoseconds(CYCLE_TIME.as_nanos()) / expected_updated_time_per_work_item,
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

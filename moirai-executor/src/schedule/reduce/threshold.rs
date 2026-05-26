//! Reduction scheduling thresholds derived from cache-line work volume.

use std::mem;

use moirai_core::constants::CACHE_LINE_SIZE;

const INLINE_REDUCTION_CACHE_LINES_PER_WORKER: usize = 2;

pub(crate) fn inline_reduction_limit<T>(worker_count: usize) -> usize {
    let value_size = mem::size_of::<T>().max(1);
    let values_per_cache_line = (CACHE_LINE_SIZE / value_size).max(1);

    worker_count
        .max(1)
        .saturating_mul(values_per_cache_line)
        .saturating_mul(INLINE_REDUCTION_CACHE_LINES_PER_WORKER)
}

#[cfg(test)]
mod tests {
    use super::inline_reduction_limit;

    #[test]
    fn threshold_scales_with_worker_count_and_value_width() {
        assert_eq!(inline_reduction_limit::<usize>(4), 64);
        assert_eq!(inline_reduction_limit::<u8>(4), 512);
        assert_eq!(inline_reduction_limit::<[u8; 64]>(4), 8);
    }
}

use super::ParallelStreamExt;
use futures::StreamExt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

#[test]
fn par_map_yields_every_item_with_correct_values() {
    // Unordered, so sort before comparing — every input must map exactly once.
    let mut results: Vec<u64> = futures::executor::block_on(
        futures::stream::iter(0..200u64)
            .par_map(8, |x| async move { x * 2 })
            .collect(),
    );
    results.sort_unstable();

    let expected: Vec<u64> = (0..200u64).map(|x| x * 2).collect();
    assert_eq!(results, expected);
}

#[test]
fn par_map_bounds_in_flight_concurrency_to_limit() {
    const LIMIT: usize = 4;
    const ITEMS: u64 = 40;

    let in_flight = Arc::new(AtomicUsize::new(0));
    let peak = Arc::new(AtomicUsize::new(0));
    let in_flight_for_items = Arc::clone(&in_flight);
    let peak_for_items = Arc::clone(&peak);

    let processed: Vec<u64> = futures::executor::block_on(
        futures::stream::iter(0..ITEMS)
            .par_map(LIMIT, move |x| {
                let in_flight = Arc::clone(&in_flight_for_items);
                let peak = Arc::clone(&peak_for_items);
                async move {
                    // The decrement happens before the result is yielded, so
                    // `buffer_unordered` cannot start a replacement item until an
                    // active one has left — `now` therefore never exceeds LIMIT.
                    let now = in_flight.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(now, Ordering::SeqCst);
                    std::thread::sleep(Duration::from_millis(15));
                    in_flight.fetch_sub(1, Ordering::SeqCst);
                    x
                }
            })
            .collect(),
    );

    assert_eq!(processed.len(), ITEMS as usize);
    let observed_peak = peak.load(Ordering::SeqCst);
    assert!(
        observed_peak <= LIMIT,
        "in-flight peak {observed_peak} exceeded the bound {LIMIT}"
    );
    // The 15 ms hold forces overlap on the multi-worker scheduler, proving the
    // items run in parallel rather than serially.
    assert!(
        observed_peak >= 2,
        "expected parallel overlap across workers, saw peak {observed_peak}"
    );
}

#[test]
fn par_for_each_visits_every_item_exactly_once() {
    let count = Arc::new(AtomicUsize::new(0));
    let count_for_items = Arc::clone(&count);

    futures::executor::block_on(futures::stream::iter(0..150u64).par_for_each(8, move |_| {
        let count = Arc::clone(&count_for_items);
        async move {
            count.fetch_add(1, Ordering::Relaxed);
        }
    }));

    assert_eq!(count.load(Ordering::Relaxed), 150);
}

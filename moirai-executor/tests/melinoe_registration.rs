//! Startup registration for the Moirai-to-Melinoe partition bridge.

use melinoe::{MelinoeCell, brand_scope};
use moirai_executor::schedule::SyncTask;

#[test]
fn initialize_restores_bridge_for_direct_partition() {
    // Warm the global pool before clearing the bridge so first-use registration
    // cannot satisfy the refresh check.
    let executor = moirai_executor::global();
    executor
        .for_each_indexed::<SyncTask, _>(1, |_| {
            // Moirai runs nested indexed work on this lane. Melinoe's fallback
            // creates scoped threads, even though it produces the same shards.
            let outer_thread = std::thread::current().id();
            let partition = || {
                brand_scope(|token| {
                    let mut cells: Vec<MelinoeCell<'_, usize>> =
                        (0..32).map(|_| MelinoeCell::new(0)).collect();
                    let results =
                        melinoe::sync::partition_map(&mut cells, 4, |start, mut shard| {
                            for (offset, cell) in shard.iter_mut().enumerate() {
                                *cell = start + offset;
                            }
                            (shard.len(), std::thread::current().id() == outer_thread)
                        });

                    let snapshot = token.share();
                    let values: Vec<_> = cells.iter().map(|cell| *cell.borrow(snapshot)).collect();
                    assert_eq!(values, (0..32).collect::<Vec<_>>());
                    results
                })
            };

            melinoe::sync::clear_parallel_executor();
            let fallback = partition();
            assert_eq!(fallback.len(), 4);
            assert!(fallback.iter().all(|&(len, _)| len == 8));
            assert!(
                fallback.iter().any(|&(_, on_outer_lane)| !on_outer_lane),
                "scoped fallback should run work off the outer lane"
            );
            moirai_executor::initialize();
            let restored = partition();
            melinoe::sync::clear_parallel_executor();
            assert_eq!(restored, vec![(8, true); 4], "Moirai bridge");
        })
        .expect("outer indexed region");
}

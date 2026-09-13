//! Startup registration for the Moirai-to-Melinoe partition bridge.

use melinoe::{MelinoeCell, brand_scope};

#[test]
fn initialize_registers_bridge_before_direct_partition() {
    // Start from the same empty slot a process has before any scheduler access.
    melinoe::sync::clear_parallel_executor();
    moirai_executor::initialize();

    brand_scope(|token| {
        let mut cells: Vec<MelinoeCell<'_, usize>> = (0..32).map(|_| MelinoeCell::new(0)).collect();
        let results = melinoe::sync::partition_map(&mut cells, 4, |start, mut shard| {
            for (offset, cell) in shard.iter_mut().enumerate() {
                *cell = start + offset;
            }
            shard.len()
        });

        assert_eq!(results, vec![8, 8, 8, 8]);
        let snapshot = token.share();
        let values: Vec<_> = cells.iter().map(|cell| *cell.borrow(snapshot)).collect();
        assert_eq!(values, (0..32).collect::<Vec<_>>());
    });

    // Do not leak the process-global provider into another integration test.
    melinoe::sync::clear_parallel_executor();
}

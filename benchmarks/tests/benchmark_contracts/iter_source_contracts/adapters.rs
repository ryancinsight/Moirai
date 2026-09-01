/// Adapter files whose `drive` implementations this audit governs.
const PARALLEL_ADAPTER_SOURCES: [&str; 13] = [
    "../moirai-iter/src/parallel/adapters/blocks.rs",
    "../moirai-iter/src/parallel/adapters/chunks.rs",
    "../moirai-iter/src/parallel/adapters/filter.rs",
    "../moirai-iter/src/parallel/adapters/flat.rs",
    "../moirai-iter/src/parallel/adapters/map.rs",
    "../moirai-iter/src/parallel/adapters/pair.rs",
    "../moirai-iter/src/parallel/adapters/position.rs",
    "../moirai-iter/src/parallel/adapters/ref_ops.rs",
    "../moirai-iter/src/parallel/adapters/side_effect.rs",
    "../moirai-iter/src/parallel/adapters/slice_ops.rs",
    "../moirai-iter/src/parallel/adapters/stride.rs",
    "../moirai-iter/src/parallel/adapters/window.rs",
    "../moirai-iter/src/parallel/sources.rs",
];

/// The `drive` body that collects the whole logical stream before any split.
const COLLECT_BEFORE_SPLIT: &str = "consumer.consume(VecParIter::new(self.seq_items()))";

/// Adapter `drive` bodies still permitted to collect before splitting.
///
/// A non-increasing baseline, on the model of the repository's other
/// conformance ratchets. Twenty-six adapter and source `drive` implementations
/// collected the whole logical stream into one vector before any split,
/// discarding the source's shards for every chain containing them; six were
/// converted to push their transform into a consumer instead. The remaining
/// twenty each carry the reason they cannot be at the `drive` that stays —
/// an absent logical offset, a prefix dependency, a fixed combine order, two
/// sources split in lockstep, or threaded state.
///
/// This number only decreases. Raising it means either a converted adapter
/// regressed to collecting or a newly written one was authored in the shape
/// this item removed; both are the same defect and both fail here.
const COLLECT_BEFORE_SPLIT_BASELINE: usize = 20;

#[test]
fn parallel_adapters_push_transforms_into_consumers_rather_than_collecting() {
    let drives: Vec<String> = PARALLEL_ADAPTER_SOURCES
        .iter()
        .map(|path| read_benchmark(path))
        .collect();
    // Only the drive-bearing files feed the counts below; the consumers they
    // push into live in their own module and hold no `drive` of their own.
    let source = drives.join("\n");
    let with_consumers = format!(
        "{source}\n{}",
        read_benchmark("../moirai-iter/src/parallel/consumers.rs")
    );

    // The converted `drive` shapes. A revert to collecting removes the marker,
    // so the conversion cannot silently come back out.
    for required in [
        // `filter_map` is per-item and stateless, so a shard decides its own
        // items alone.
        ".drive(FilterMapConsumer::new(consumer, self.filter_map_fn))",
        "pub struct FilterMapConsumer<C, F>",
        // One input expanding to many outputs does not block the push: each
        // expansion depends on its own item.
        ".drive(FlatMapConsumer::new(consumer, self.flat_map_fn))",
        "pub struct FlatMapConsumer<C, F>",
        // `flatten` is `flat_map` with the identity expansion and reuses that
        // consumer rather than duplicating the split and combine forwarding.
        ".drive(FlatMapConsumer::new(consumer, |item: I::Item| item))",
        // `update` is one input to one output with no state between items.
        ".drive(MapConsumer::new(consumer, move |mut item: I::Item| {",
        // The block adapters leave the item stream identical to the base's.
        "self.base.drive(consumer)",
    ] {
        assert!(
            with_consumers.contains(required),
            "converted parallel adapter must keep pushing its transform into a consumer \
             through {required}"
        );
    }

    // Every `drive` that still collects carries its reason, so a future reader
    // finds why rather than assuming the conversion was merely unfinished.
    let recorded_reasons = source.matches("# Why this stays sequential").count();
    let collecting = source.matches(COLLECT_BEFORE_SPLIT).count();
    assert!(
        recorded_reasons >= collecting,
        "{collecting} parallel adapter `drive` implementations collect the logical stream \
         before splitting but only {recorded_reasons} record why; an unexplained collect is \
         indistinguishable from an unfinished conversion"
    );

    assert!(
        collecting <= COLLECT_BEFORE_SPLIT_BASELINE,
        "{collecting} parallel adapter `drive` implementations collect the whole logical \
         stream before any split, above the non-increasing baseline of \
         {COLLECT_BEFORE_SPLIT_BASELINE}; a converted adapter regressed or a new one was \
         written in the shape that discards the source's shards"
    );
}

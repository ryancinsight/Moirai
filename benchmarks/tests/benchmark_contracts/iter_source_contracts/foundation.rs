#[test]
fn iterator_base_does_not_expose_boxed_future_execution_trait() {
    let source = format!(
        "{}\n{}",
        read_benchmark("../moirai-iter/src/base.rs"),
        read_benchmark("../moirai-iter/src/base/tests.rs")
    );

    for required in [
        "pub const fn inner(&self) -> &I",
        "pub fn context(&self) -> &Arc<C>",
        "pub fn into_parts(self) -> (I, Arc<C>)",
        "pub const fn function(&self) -> &F",
        "pub const fn predicate(&self) -> &F",
        "pub const fn size(&self) -> usize",
        "#[path = \"base/tests.rs\"]",
        "base_adapters_expose_components_without_dead_fields",
    ] {
        assert!(
            source.contains(required),
            "iterator base adapters must expose live fields through {required}"
        );
    }

    for prohibited in [
        "pub trait ExecutionBase: Send + Sync + 'static",
        "Pin<Box<dyn Future",
        "Box::pin(async move",
        "execute_each<T, F>",
        "execute_map<T, R, F>",
        "execute_filter<T, F>",
        "#[allow(dead_code)]",
    ] {
        assert!(
            !source.contains(prohibited),
            "iterator base must not reintroduce unused boxed-future execution trait shape through {prohibited}"
        );
    }
}
#[test]
fn channel_fusion_uses_typed_channels_without_placeholder_pipeline() {
    let source = read_benchmark("../moirai-iter/src/channel_fusion.rs");

    for required in [
        "pub struct ChannelSplitter<T, I, C>",
        "channels: Vec<C>",
        "C: FusableChannel<T>",
        "pub fn add_channel(mut self, channel: C) -> Self",
        "pub struct ChannelMerger<T, C>",
        "buffer: VecDeque<T>",
        "pop_front()",
        "fn split_channels<C>(",
        "test_channel_merger_fair_merge_uses_fifo_order",
        "test_channel_splitter_broadcast_clones_to_every_channel",
    ] {
        assert!(
            source.contains(required),
            "channel fusion must retain typed zero-cost channel structure through {required}"
        );
    }

    for prohibited in [
        "Vec<Box<dyn FusableChannel<T>>>",
        "Box<dyn FusableChannel<T>>",
        "SplitStrategy::Hash",
        "pub struct Pipeline",
        "PipelineStage",
        "let hash = 0",
        "remove(0)",
        ".add_channel(Box::new",
    ] {
        assert!(
            !source.contains(prohibited),
            "channel fusion must not reintroduce dynamic or placeholder structure through {prohibited}"
        );
    }
}

#[test]
fn streaming_iter_uses_monomorphized_producer_and_fifo_buffer() {
    let source = format!(
        "{}\n{}\n{}",
        read_benchmark("../moirai-iter/src/iter_ops.rs"),
        read_benchmark("../moirai-iter/src/iter_ops/streaming.rs"),
        read_benchmark("../moirai-iter/src/iter_ops/tests.rs")
    );

    for required in [
        "mod streaming;",
        "streaming::StreamingIter",
        "pub struct StreamingIter<T, F>",
        "buffer: VecDeque<T>",
        "producer: F",
        "F: FnMut() -> Option<T>",
        "VecDeque::with_capacity(capacity)",
        "capacity: capacity.max(1)",
        "push_back(item)",
        "pop_front()",
        "streaming_iter_preserves_fifo_order",
    ] {
        assert!(
            source.contains(required),
            "streaming iterator must retain monomorphized producer/FIFO shape through {required}"
        );
    }

    for prohibited in [
        "producer: Box<dyn FnMut() -> Option<T>>",
        "producer: Box::new(producer)",
        "Box<dyn FnMut",
        "self.buffer.remove(0)",
    ] {
        assert!(
            !source.contains(prohibited),
            "streaming iterator must not reintroduce boxed producer or shifting FIFO through {prohibited}"
        );
    }
}

# moirai-metrics

[![crates.io](https://img.shields.io/crates/v/moirai-metrics.svg)](https://crates.io/crates/moirai-metrics)
[![docs.rs](https://docs.rs/moirai-metrics/badge.svg)](https://docs.rs/moirai-metrics)

Performance metrics for the [Moirai](https://github.com/ryancinsight/Moirai)
concurrency library. Metric handles are cloneable and backed by shared atomic or
bounded mutex state; snapshots are value copies, so exporting never retains a
collector lock.

Types: `Counter`, `Gauge`, `Histogram` (with `HistogramStats`), `Metrics` /
`MetricsCollector`, `MetricsSnapshot`, and `PrometheusExporter`.

```toml
[dependencies]
moirai-metrics = "0.5"
```

```rust
use moirai_metrics::{Counter, Gauge};

let tasks_completed = Counter::new();
tasks_completed.increment();
tasks_completed.add(4);
assert_eq!(tasks_completed.get(), 5);

let queue_depth = Gauge::new();
queue_depth.set(3);
queue_depth.decrement();
assert_eq!(queue_depth.get(), 2);
```

Full documentation: <https://docs.rs/moirai-metrics>

## License

Licensed under either of [Apache-2.0](../LICENSE-APACHE) or
[MIT](../LICENSE-MIT) at your option.

### Fixed
- Fixed example all-target clippy coverage by documenting intentionally broad demo-domain fields, replacing index/range and loop patterns, handling map insert results, and factoring an IoT event-handler alias.
- Fixed `async_tcp_comparison` persistent stream setup so persistent sockets are created immediately before the persistent-stream benchmark group instead of idling through the preceding loopback group.
- Fixed the scheduler-diagnostics wake-decision path to pass the concrete `SchedulerInner` reference into the static `ContendedWakable` boundary.
- Fixed strict clippy findings in zero-copy iterator helper pointer captures without changing borrowed chunk semantics.
- Fixed async `RwLock` release-handoff coverage by adding value-semantic tests for final-reader-to-writer and writer-to-multiple-reader grant paths.
- Fixed utility SIMD benchmark coverage so the wide vector-addition row asserts expected values before timing and benchmark contracts reject reintroduced type-suffixed private SIMD routing.

[Version 0.3.0 index](../index.md) · [Current changelog](../../../../CHANGELOG.md)

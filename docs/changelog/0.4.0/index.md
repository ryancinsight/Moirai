## [0.4.0] - 2026-07-17

### Changed

- Confirm the RITK masked Parzen-cache consumer path against the merged Moirai
  scheduler pin, closing the indexed caller-region downstream gate.
- [major] Direct Atlas providers follow their merged default branches. The
  workspace removes revision quarantine and the local Melinoe patch, and adopts
  Mnemosyne 0.5/Core 0.2; `Cargo.lock` is the sole reproducibility pin.
- [major] Dual channel consolidation (TREE-DUP-002): fold `unified_channel`
  into `channel/` module. Extended `Channel<T>` trait with `send_batch`,
  `recv_batch`, `close`, `is_closed`, `len`, `stats` (default impls). Moved
  `ChannelConfig` → `channel/config.rs`, `ChannelStatistics` →
  `channel/stats.rs`. Added `InvalidConfig` to `ChannelError`. `UnifiedChannel`
  now implements `Channel<T>` under `channel::unified`. Deleted the
  `unified_channel/` module. Migration: replace `unified_channel::*` imports
  with `channel::unified::*`; match `ChannelError::InvalidConfig` in error
  handlers.

### Fixed

- Make async waiter handoff retain FIFO behavior after cancellation without
  linear waiter scans; reclaim broadcast messages after every live receiver
  consumes them; derive timer-compaction coverage from the active heap state.
- Apply warning-free iterator and arithmetic idioms in affected examples and
  HTTP framing fixtures.

### Breaking

- Moirai 0.4.0 requires Rust 1.95. Migration: update the consumer toolchain
  before resolving the default Mnemosyne provider graph.
- **Breaking:** removed the `unified_channel` public module. Use
  `moirai_core::channel::unified` instead.
- **Breaking:** `ChannelError` gains `InvalidConfig` variant — update
  exhaustive matches. `Channel<T>` trait adds `send_batch`/`recv_batch`/
  `close`/`is_closed`/`len`/`stats` — implementors must provide or accept
  default impls.

[Current changelog](../../../CHANGELOG.md)

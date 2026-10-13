### Changed
- `moirai-transport` `MessageRouter` is generic over its backing `Transport`
  and takes the transport at construction (`MessageRouter::new(transport)`), so
  delivery is real and zero-cost. (Pre-1.0 breaking change to a previously
  non-functional API.)
- `moirai-core` `UnifiedChannel::send` now delegates to `try_send` (single SSOT
  for the send path, ~45 lines of duplicate overflow logic removed) and
  documents that it consumes the message on `Full`/`Closed`; callers needing the
  value back to retry use `try_send`. `overflow_count` is documented as advisory.

[Version 0.3.0 index](../index.md) · [Current changelog](../../../../CHANGELOG.md)

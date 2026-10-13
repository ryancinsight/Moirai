### Breaking

- Remove the async surface of `moirai_pal::fs`:
  - `AsyncFile`, with all of its methods: the async `open`, `open_with`,
    `open_with_options`, `read`, `read_to_string`, `read_to_end`, `write`,
    `flush`, `seek`, `sync_all`, `sync_data` and `metadata`, and the
    `poll_read`, `poll_write` and `poll_flush` methods.
  - `YieldFuture` and `yield_now`.
  - The async path functions `write`, `append`, `copy`, `metadata`,
    `rename`, `remove_file`, `create_dir`, `create_dir_all`, `remove_dir` and
    `remove_dir_all`.

  Add `moirai_pal::fs::File`, a blocking handle whose methods take `&self`:
  `open_with`, `read`, `read_to_end`, `write`, `write_all`, `seek`,
  `sync_all`, `sync_data`, `metadata` and `read_at`. Migrate async callers to
  `moirai_async::fs`, which runs these calls on its blocking pool. Also
  remove the `moirai_async::fs::options` module. `FileOpenOptions` is now
  `moirai_pal::fs::FileOpenOptions`, and `moirai_async::fs::FileOpenOptions`
  still names it.

- Remove `moirai_iter::ThreadPool` and its `moirai::ThreadPool` re-export. The
  crate's own FIFO thread pool was a second runtime beside the unified
  scheduler, kept only as the fallback for a shutting-down executor — which is
  when starting worker threads is least defensible, and where the caller's own
  thread does just as well for a flat index domain. The indexed operations now
  re-run their work on the caller, and `ParallelContext` schedules through the
  process-wide executor rather than owning a pool, so several contexts share
  one worker set instead of over-subscribing the machine. Callers wanting a
  fan-out use `moirai_parallel`'s data-parallel operators or
  `moirai_executor::global()`.

[Version 0.6.1 index](../index.md) · [Current changelog](../../../../CHANGELOG.md)

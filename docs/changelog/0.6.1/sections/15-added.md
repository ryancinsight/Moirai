### Added

- `moirai_core::channel::SpscRing` with `split(&mut self)`, yielding
  `SpscProducer`/`SpscConsumer` halves that borrow the ring instead of sharing
  an `Arc`. For pairs living inside a scope — a `thread::scope`, a frame loop, a
  pipeline stage — the scope already proves the ring outlives the halves, so the
  refcount enforces statically-checkable lifetimes at runtime cost. The borrowed
  pair drops that cost: no allocation beyond the ring's buffer, and dropping a
  half is one store rather than a refcount decrement and a conditional free.
  Both halves implement `Producer`/`Consumer`, so generic code takes either
  flavour with no `'static` bound. `spsc(capacity)` is unchanged. See ADR-026.

- `moirai_core::channel::{Producer, Consumer}` — the sending and receiving
  halves of a channel as separate contracts, neither requiring `Sync`. This
  restores generic use of the SPSC halves, which ADR-024 had excluded: they
  cannot implement `Channel<T>`, whose `Send + Sync` supertrait is precisely
  what a single-producer half must not satisfy. Implemented on both SPSC
  halves, both MPMC halves, and `MpmcChannel` itself, so a shareable channel is
  usable whole or split. See ADR-025.

- Expose a borrowing `moirai_parallel::scope` surface over the unified
  scheduler so downstream parallel regions can spawn an arbitrary number of
  non-`'static` tasks and return a value only after every task completes.
- Build, install, attest, and attach locked `moirai-python` wheels for CPython
  3.10 through 3.13 on Linux, Windows, and macOS when a matching GitHub Release
  is published, then publish the same artifacts to PyPI through OIDC Trusted
  Publishing.

[Version 0.6.1 index](../index.md) · [Current changelog](../../../../CHANGELOG.md)

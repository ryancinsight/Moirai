# Channels

Moirai provides three channel families for inter-task communication, all
designed for zero-copy, lock-free message passing.

## SPSC Channel

Single-producer/single-consumer bounded ring buffer. The ring is
`SpscChannel<T>` (crate-private); callers access it through
`SpscSender`/`SpscReceiver` pairs that enforce ownership rules at compile time.

```rust,ignore
use moirai_core::channel::spsc;

let (sender, receiver) = spsc::channel::<u64>(1024)?;
sender.send(42)?;
let value = receiver.recv()?;
```

- Lock-free; atomic `head`/`tail` counters with cache-line padding
- Falls back to `thread::yield_now` after spin-loop hints
- `SpscSender` and `SpscReceiver` are neither `Clone` nor `Sync`

## MPMC Channel

Multi-producer/multi-consumer bounded queue:

```rust,ignore
use moirai_core::channel::mpmc;
let (sender, receiver) = mpmc::channel::<u64>(1024)?;
let tx2 = sender.clone();  // MpmcSender is Clone
```

## Unified Channel

`unified_channel` presents a single `UnifiedSender`/`UnifiedReceiver` facade
that routes to either SPSC or MPMC based on `ChannelConfig`.

## Channel Selection (`Select`)

`Select` multiplexes multiple receivers into a single ready-notification.

## `ChannelStatistics`

Accumulates per-channel metrics (sent, received, dropped, backpressure events).

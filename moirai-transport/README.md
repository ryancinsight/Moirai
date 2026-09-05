# moirai-transport

[![crates.io](https://img.shields.io/crates/v/moirai-transport.svg)](https://crates.io/crates/moirai-transport)
[![docs.rs](https://docs.rs/moirai-transport/badge.svg)](https://docs.rs/moirai-transport)

Transport layer for the [Moirai](https://github.com/ryancinsight/Moirai)
concurrency library. One `Transport` trait spans thread, process, and machine
boundaries, so the same addressing works for local and remote communication.

- `InMemoryTransport` — same-process messaging over the core channels.
- `IpcTransport` — shared-memory same-machine IPC (Unix and Windows).
- `TcpTransport` / `NetworkTransport` — machine-to-machine (`network` feature).
- `ArchivedUniversalSender<T: ArchiveSerialize>` /
  `ArchivedUniversalReceiver<T: ArchiveView>` — the canonical typed
  cross-boundary channel, using rkyv-style archive serialization with
  zero-copy borrowed views validated on receive.
- `route` and `remote_task` — endpoint address books and fixed-format remote
  tasks admitted through sealed capability tokens (`scheduler-routes` /
  `distributed` features). Arbitrary closure remoting is intentionally rejected.

```toml
[dependencies]
moirai-transport = "0.5"
```

```rust
use moirai_transport::{Address, InMemoryTransport, Transport};

let transport = InMemoryTransport::new();
let address = Address::Local("worker".to_string());

transport.send(&address, b"payload".to_vec()).unwrap();
assert_eq!(transport.recv(&address).unwrap(), b"payload".to_vec());
```

Full documentation: <https://docs.rs/moirai-transport>

## Process lifecycle

`process::ProcessSupervisor` owns process spawning, optional stdin/stdout
pipes, finite waits, and deadline-aware termination. `ProcessSpec` can clear
the child environment and require process-tree containment. Windows 10 or
later assigns jobs atomically at process creation with explicit inherited
handles; terminate-on-drop jobs kill descendants on their last handle close.
`terminate_timeout` confirms zero active processes in the Windows job before
reporting cleanup complete. Other platforms reject requested tree containment
and retain direct-child supervision only. Portable Drop is a best-effort,
nonblocking termination request; explicit termination reports OS failures.

`ManagedProcess::wait()` now has a finite 30-second default and
`terminate()` a finite one-second cleanup budget. Callers requiring another
finite budget use `wait_timeout` or `terminate_timeout`. `ProcessStatus` now
retains the original OS exit status, exposed through `exit_status()`; callers
constructing status literals must consume provider-produced statuses instead.

These contracts are described in [ADR 0043](../docs/adr/0043-contained-process-lifecycle.md).
A Windows job controls process lifetime, not filesystem, network, or privilege
access. Pipe I/O and synchronous application handlers need caller-owned policy.

## License

Licensed under either of [Apache-2.0](../LICENSE-APACHE) or
[MIT](../LICENSE-MIT) at your option.

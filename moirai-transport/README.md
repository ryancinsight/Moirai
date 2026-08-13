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

## License

Licensed under either of [Apache-2.0](../LICENSE-APACHE) or
[MIT](../LICENSE-MIT) at your option.

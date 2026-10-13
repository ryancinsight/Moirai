### Fixed
- `moirai-transport` `MessageRouter::publish` now actually delivers messages.
  The previous implementation constructed a throwaway `InMemoryTransport` per
  send and silently discarded every published message; it now routes through a
  shared transport and returns the delivered-subscriber count.


# ADR-008 Implementation Checklist: Scheduler Route Consumption

## Completed

- [x] Route-to-transport address binding: `moirai-transport` exposes a feature-gated transport route consumer that maps `SchedulerRoute` values to concrete `Address` values.
- [x] Archived local route roundtrip: routed send/receive uses transport-owned archive bytes and borrowed `ArchiveView` validation for the local in-memory backend.
- [x] Static route policy consumption: `RoutedArchivedSender<P>` and `RoutedArchivedReceiver<P>` remain generic over sealed ZST `RoutePolicy` markers.
- [x] Server route endpoint resolution: known `ServerId` values resolve to `RemoteAddress` metadata without claiming remote execution.
- [x] Remote byte transport: `NetworkTransport` sends and receives remote payload bytes through a bounded length-prefixed TCP frame with value tests.

## Open

- [ ] OS process executor lifecycle: define child process creation, supervision, shutdown, backpressure, and failure propagation.
- [ ] Remote task execution: define task envelopes, result envelopes, failure propagation, and scheduler integration over the remote byte transport.
- [ ] Server transport execution: add persistent connection lifecycle, bounded queues, and backpressure policy before claiming production server execution.
- [ ] Mnemosyne allocator ownership handoff: define which allocation region owns archived task payloads across thread, process, and server boundaries.
- [ ] End-to-end routed execution benchmark: benchmark real process/server execution only after the transport backend performs real work and value results return through the scheduler contract.

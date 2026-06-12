# ADR-008 Implementation Checklist: Scheduler Route Consumption

## Completed

- [x] Route-to-transport address binding: `moirai-transport` exposes a feature-gated transport route consumer that maps `SchedulerRoute` values to concrete `Address` values.
- [x] Archived local route roundtrip: routed send/receive uses transport-owned archive bytes and borrowed `ArchiveView` validation for the local in-memory backend.
- [x] Static route policy consumption: `RoutedArchivedSender<P>` and `RoutedArchivedReceiver<P>` remain generic over sealed ZST `RoutePolicy` markers.
- [x] Server route endpoint resolution: known `ServerId` values resolve to `RemoteAddress` metadata without claiming remote execution.
- [x] Remote byte transport: `NetworkTransport` sends and receives remote payload bytes through a bounded length-prefixed TCP frame with value tests.
- [x] Remote task envelopes/results: fixed-format `RemoteTaskEnvelope` and `RemoteTaskResult` archives execute explicit built-in `EchoBytes` and `SumU64` operations through `RemoteTaskServer::serve_one` and `RemoteTaskClient::execute`.
- [x] Route-to-remote-task scheduler integration: `RoutedRemoteTaskClient<P>` selects `SchedulerRoute::Server` through `HybridRouter<P>`, resolves it through `RouteAddressBook`, and executes a fixed-format remote task.
- [x] OS process executor lifecycle: `ProcessSupervisor` spawns real OS child processes from `ProcessSpec`, observes `try_wait`/`wait`/bounded wait status, terminates live children, and applies explicit `ProcessDropPolicy` cleanup.
- [x] Route-to-process task execution: `RoutedProcessTaskClient<P>` binds selected `SchedulerRoute::Process` values to registered `ProcessEndpoint` entries, launches the child process, executes a fixed-format remote task through the child task server, waits for child completion, and returns the value-checked task result with process status.
- [x] Server transport backpressure: `BoundedRemoteTaskServer` owns one listener lifecycle, accepts fixed-format remote task frames into a bounded `sync_channel`, executes them on a bounded worker set, and returns accepted/completed counts.
- [x] Arbitrary closure remoting boundary: `RemoteCapabilityToken<C>` uses sealed zero-sized capability markers to admit only fixed-format built-in operations; arbitrary Rust closures and dynamic task traits remain outside the process/server transport contract.
- [x] Mnemosyne allocator ownership handoff: `TransportPayload<R>` tags owned archive bytes with sealed thread/process/server/device payload regions, moves buffers between regions without cloning, rejects pointer transfer across process/server/device regions, and relies on the top-level Mnemosyne global allocator feature for process-local allocations.
- [x] End-to-end routed execution benchmark: `process_server_routed_execution` measures selected server-route and process-route fixed-format `SumU64` execution with real `RemoteTaskServer`, TCP request/result frames, supervised child process execution, and value assertions.

## Open

No open ADR-008 implementation items remain.

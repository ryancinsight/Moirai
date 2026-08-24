# Routes: from scheduler decision to wire address

Chapters 1–3 gave us transports, region-tagged payloads, and typed
endpoints. One vocabulary gap remains: the scheduler thinks in
`SchedulerRoute` values — `Thread`, `Process`, `Accelerator`, `Server`,
each carrying abstract ids like `ProcessId` and `ServerId` — while
transports speak `Address::Local(String)` and `Address::Remote`. This
chapter covers `route.rs`, the resolver between those vocabularies, and
the failure semantics of each path through it.

## Resolution keeps its provenance

```rust,ignore
pub struct RouteResolution {
    route: SchedulerRoute,
    address: Address,
    placement: RoutePlacement,
}
```

A resolved route is *not* just an address. It carries the scheduler's
original decision beside the wire address derived from it, plus a
placement tag (`Local`, `Remote`, `Accelerator(route)`). Callers can ask
`accelerator()` whether work targets a device without re-parsing the
route — the kind of accessor that prevents every downstream consumer
from re-deriving the same answer by pattern matching.

## The address grammar

Every in-process route shape funnels into one string format:

```text
{namespace}/process/{pid}/thread/{tid}
{namespace}/process/{pid}/thread/{tid}/async-lane/{lane}
```

Two properties make this safe as a channel key: the leading namespace
isolates unrelated systems sharing one transport manager, and async lanes
extend the leaf rather than creating a parallel naming scheme — a thread
and its async lane are visibly the same endpoint family.

## Precedence and the server fallback

`RouteAddressBook::resolve` matches the four route shapes in order, and
three of them are total: thread, process, and accelerator routes always
resolve locally, because their ids name things inside this process by
construction.

The `Server` shape is different. Its resolver searches the static
endpoint catalog (`Vec<ServerEndpoint>` keyed by `ServerId`) and:

- on a hit, produces the registered `RemoteAddress`;
- on a miss, **falls back to the local thread address** instead of
  failing.

That fallback is the one place in Part IV where resolution degrades
silently, and it deserves its tradeoff stated plainly: it lets code
written against a server topology run unchanged on a machine with no
catalog — useful for tests and single-node deployments — at the cost of
masking a missing registration in production. If you need missing
endpoints to be loud, the current design says so by omission: strictness
would be a `Result`-returning `resolve_strict` beside it. This asymmetry
is documented behaviour, not an accident; treat changing it as a policy
decision, not a bug fix.

## Region tagging rides the route

`archive_route_payload` closes the loop with Chapter 2: after archiving
into a `ThreadPayloadRegion`, it re-tags the buffer by route shape —

```rust,ignore
Ok(match route {
    SchedulerRoute::Thread(_)     => payload.into_bytes(),
    SchedulerRoute::Process(_)    => payload.handoff::<ProcessPayloadRegion>().into_bytes(),
    SchedulerRoute::Server(_)     => payload.handoff::<ServerPayloadRegion>().into_bytes(),
    SchedulerRoute::Accelerator(_) => payload.handoff::<DevicePayloadRegion>().into_bytes(),
})
```

The ownership region of every routed message is derived mechanically from
the route that carries it. Nobody hand-labels a payload "this will cross
a process boundary"; crossing one *is* what the selected route means,
and the type records it.

## Three clients, three lifecycle contracts

`RoutedArchivedSender`/`Receiver` are Chapter 3 endpoints driven by
resolved addresses; `send_route` returns the `RouteResolution` so callers
learn where their message went, and `send_selected` composes with the
scheduler's `HybridRouter::select::<C>(priority, sequence)` so routing
policy stays in the scheduler, not the transport layer.

The two remote-task clients add real lifecycle semantics:

- `RoutedRemoteTaskClient::execute_route` requires a *remote* resolution
  (`let Address::Remote(server) = ... else { Closed }`) — asking for a
  server task against a local address is a caller bug surfaced as an
  error, not coerced.
- `RoutedProcessTaskClient` owns the whole child-process arc per task:
  spawn under `ProcessSupervisor`, run the remote task against the
  child's task server, then `wait_bounded(wait_policy)` — terminating the
  child when the bound lapses rather than waiting forever. Its dedicated
  error enum separates `NonProcessRoute`, `MissingProcessEndpoint`,
  `Process(_)`, and `Transport(_)`, because "you asked the wrong client"
  and "the child crashed" demand different responses.

## Failure semantics summary

| Path | On failure |
| --- | --- |
| Thread/Process/Accelerator resolve | cannot fail — ids are local by construction |
| Server resolve, no catalog entry | silent local fallback (policy, see above) |
| Remote-task execute on non-remote address | `TransportError::Closed` |
| Process task on non-process route | `RoutedProcessTaskError::NonProcessRoute` |
| Process task, unregistered process | `MissingProcessEndpoint` |
| Child overruns wait bound | terminated; status reports the outcome |

The pattern across all six: structural mismatches are errors, capacity
and lifecycle limits are errors, and exactly one downgrade exists — the
documented server fallback.

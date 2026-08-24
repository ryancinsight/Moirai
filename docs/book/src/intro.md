# Introduction

Moirai is a runtime for work that spans threads, processes, and machines.
The transport and routing layers answer one question at several scales:
*how does a unit of work reach the executor that should run it, when the
answer is not known until runtime?*

This book teaches that design from first principles. It is the pedagogical
companion to the API documentation (docs.rs) and the source; it explains
*why* the pieces exist before *how* they are spelled.

## The problem stack

Three forces shape everything in `moirai-transport`:

1. **Heterogeneous endpoints.** The same logical send may cross a thread
   boundary, a process boundary through shared memory (`IpcTransport`),
   or a network link (`TcpTransport`). Code above the transport must not
   know which.
2. **Capability-scoped authority.** A sender may act only on destinations
   it was granted. Routing is therefore an authorization decision as much
   as an addressing one — this is what `remote_task/capability.rs`
   negotiates.
3. **Backpressure without blocking.** Moirai never blocks an executor
   thread on a slow peer; every crossing carries explicit capacity and
   failure semantics instead of implicit waits.

Each chapter takes one layer of that stack, states its invariant first,
and then maps the invariant onto the module that enforces it.

| Chapter | Source of truth | Status |
| --- | --- | --- |
| Transports and their capability contract | `transport.rs`, `network.rs`, `process.rs` | planned |
| Payload framing | `payload.rs` | planned |
| Safe channels: typed endpoints over raw links | `safe_channel.rs` | planned |
| Routes: resolution, precedence, failure semantics | `route.rs` | planned |
| The router: dispatch, retries, backpressure | `router.rs` | planned |
| Remote tasks: capabilities and server lifecycle | `remote_task/` | planned |

Planned entries gain links as their teaching content lands under
`MOI-AUDIT-DOC-009`; this page never ships placeholder chapters or dead
links.

Chapters are written against the current tree and carry worked examples
that compile against the public API. Where behaviour is verified by a test
suite or fuzz target, the chapter links it rather than restating it.

## Status

The introduction is the only completed chapter so far; the six listed
chapters are sequenced as board follow-ups under `MOI-AUDIT-DOC-009` and
land when their teaching content exists — no placeholders.

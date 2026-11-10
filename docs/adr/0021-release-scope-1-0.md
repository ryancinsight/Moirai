# ADR 0021: Moirai release scope for 1.0

- Status: Accepted
- Date: 2026-11-10
- Class: `[arch]` `[minor]`

## Context

Moirai (workspace 0.11.3, `moirai-runtime`) is a unified scheduler/router for
Rust work placement with async, parallel, transport, and platform integrations.
The backlog contains open items across execution, async I/O, QUIC, WebView, and
verification, and no milestone ladder exists to 1.0.

## Decision

Adopt a three-milestone ladder to 1.0 recorded in `backlog/milestones.md`:

- **M-core-execution** (0.12.0): users spawn tasks, join handles, and run
  parallel map/reduce through the unified scheduler with stable contracts.
- **M-async-io-transport** (0.13.0): users use async I/O and transport
  primitives (including native QUIC paths where verified) with bounded behavior.
- **M-1.0** (1.0.0): stable unified execution surface (authority-gated) with
  documented contracts and release gates satisfied.

## Alternatives

- Plan 1.0 as a single step: rejected because significant open correctness and
  verification work remains.
- Keep roadmap only in backlog items: rejected because stack policy requires a
  runnable capability-oriented milestone ladder.

## Consequences

- Open items carry `milestone:` assignments and feature items name the exit
  check they serve.
- Milestones close only when exit checks pass on fetched default commits; 1.0
  release/publication remains authority-gated.
- This ADR is rewritten in place as scope evidence changes.

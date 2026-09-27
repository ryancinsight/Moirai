# ADR 0008 delivered-contract snapshot

This compatibility document is embedded by the benchmark-contract test. It
contains no execution state; active work lives in [`../backlog.md`](../backlog.md).

The delivered contract retains:

- Route-to-transport address binding through the transport route consumer.
- Remote byte transport and fixed-format Remote task envelopes/results.
- Route-to-remote-task scheduler integration.
- OS process executor lifecycle and Route-to-process task execution.
- Mnemosyne allocator ownership handoff.
- Arbitrary closure remoting remains outside the admitted capability set.
- The End-to-end routed execution benchmark uses real process and server routes.

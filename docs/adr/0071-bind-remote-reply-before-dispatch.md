# 0071 — Bind the remote reply receiver before dispatch

Status: Accepted

Date: 2026-10-12

## Contract and decision

`RemoteTaskClient::execute` owns the reply listener before it publishes a
request carrying that reply address. The order is bind → send → receive →
decode → task-ID validation. A failed bind dispatches no remote work. The
listener drops on every exit; message bounds and returned task checks remain
the existing transport contracts.

Previously, send preceded bind. A fast server could attempt its reply while
the client had no listening socket and exhaust the bounded connection attempts.
The protected benchmark gate observes `Closed` at that seam. Moving listener
ownership before publication removes that interleaving without sleeps, larger
retry limits, duplicate algorithms or a new public API.

The existing `NetworkTransport::listen` supplies the mechanism. A regression
holds the actual reply endpoint, invokes the actual client, and checks both
`Closed` and the server socket's `WouldBlock`: no request reaches the peer.
The old implementation delivers a request and fails that oracle. Existing
sum/echo round trips and the full routed execution benchmark retain their
computed outputs and workloads.

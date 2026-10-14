# ADR 0080: Recovery trace evidence

Date: 2026-10-14
Status: Accepted
Change class: [patch], diagnostic investigation within unreleased workspace 0.10.0.

## Specification

Continue MOI-QUIC-RECOVERY-001 without adopting an unproved correction.
Retain ADR 0079's full 108 cases per stock policy, all 30 competing-flow
experiments, real mutual TLS, exact application bytes, acknowledged FIN,
5-second operation deadlines and 60-second pair deadlines. The independent
original CUBIC promotion gate remains authoritative.

The probe may opt into aggregate tracing through
`MOIRAI_QUIC_RECOVERY_TRACE=1`. Capture begins before endpoint construction
and continues through handshake, stream admission, exchange and endpoint
drain. Keep phase transitions and final failure observations, including
connections that never reach stream admission. Record pinned Quinn's
congestion, pacing, anti-amplification, flow-blocked and packet-number-space
PTO events. Record shared timer-driver wake lateness. Counters remain
monotonic; final observations retain success or failure explicitly.

Event counts are not blocked-time fractions. PTO flight bytes describe only
PTO instants, not the entire connection's flight history. Timer lateness is
shared-driver wake time minus requested deadline, including QUIC, operation
and observation timers. It excludes later protocol scheduling delay and does
not isolate pacing timers. Quinn checks monotonic time before polling its
async timer, so counting ready polls does not measure its timer wakes. No unobserved flight
value or continuous queue bound is inferred. Instrumented measurements
remain diagnostic observations and do not substitute for the original gate.

## Architecture and alternatives

One example-only tracing layer aggregates the pinned engine's existing events.
The existing shared timer driver emits a disabled-by-default trace event
when it wakes a live registration. Transport configuration, public contracts, packet
algorithms, authentication and application admission remain unchanged.
Snapshots and phase transitions use structured tracing; raw keys and message
payloads are never captured. Use one parser and independent numerical audit
over the original archived inputs.
Drain the server's stdout and stderr concurrently with client execution;
increased diagnostic volume must not fill a pipe and block the transport.
Retain the original pair deadline and bounded task-owned output-reader join.

Enabling qlog would require additional feature/configuration surface and
synchronous per-packet output. Converting a pending server connection to
0.5-RTT would weaken the authentication boundary. Neither is required for
this observation increment. A controller change still requires explained
causal evidence and an acceptable measured fairness/latency/queue tradeoff.

## Verification

Verify aggregation from real tracing events, disabled capture, packet-space
separation, phase order, monotonic counters and rejection of corrupt or
incomplete diagnostic evidence. Exercise actual authenticated process pairs
locally. Then retain fresh full Linux matrix and shared-queue evidence, with
the exact source, binary, lock, platform and opt-in mode recorded. Compare
with retained uninstrumented studies and keep unsuccessful experiments.

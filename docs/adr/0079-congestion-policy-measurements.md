# ADR 0079: Congestion policy measurement boundary

Date: 2026-10-13
Status: Accepted

Change class: [minor]. Target: unreleased workspace 0.9.0 / moirai-quic 0.1.0.
This decision authorizes policy evaluation, not a production default change.

## Intent and constraints

MOI-QUIC-RECOVERY-001 records three 60-second failures among the original 108
mutually authenticated process pairs. Continued validated progress and zero
kernel UDP error deltas distinguish the observed limitation from demonstrated
socket overflow, but do not identify its complete cause.

Expose one bounded configuration choice over the pinned Quinn controllers:
CUBIC (the unchanged default), NewReno, and explicitly experimental BBR.
The pinned stock initial windows differ: CUBIC/NewReno use 12,000 bytes and
experimental BBR uses 240,000 bytes. These values remain unchanged; the study
compares complete stock configurations rather than attributing all differences
to the steady-state recovery algorithm. BBR also randomizes its gain-cycle
offset internally; the shared netem seed does not seed that upstream RNG.
Select Quinn's factory once when constructing transport configuration. Quinn
continues to own the controller algorithms and its erased controller ABI;
Moirai adds neither a parallel transport nor a cloned recovery algorithm.
TLS, ALPN, stream/frame admission, flow-control windows, native UDP settings,
operation deadlines, reliable delivery and acknowledged FIN remain invariant.

## Measurement specification

For every policy, retain ADR 0075's 4 profiles x 3 stream counts x 3 frame
sizes x 3 repetitions = 108 process pairs. Every successful pair proves
S * 128 * B complete application-message bytes in each direction. Nearest-rank
latencies and integer goodput are recomputed from the original samples.
Five-second operation and 60-second process-pair deadlines do not change.
The seeded netem implementation, rate, delay, loss, queue limits and packet
aggregation remain identical. Policy order rotates between repetitions on
the same runner to expose order effects. The unmodified CUBIC acceptance job
continues to enforce the original promotion gate independently.

Supplementary competing-flow experiments run two real process pairs through
the same pair of kernel queues, with a readiness barrier before client launch.
Both retain the original maximum workload and deadlines. Compare CUBIC/CUBIC,
NewReno/CUBIC, NewReno/NewReno, experimental BBR/CUBIC and BBR/BBR under
100 Mbit/s symmetric 10 ms delay, with and without 1% loss. Retain raw progress
timestamps, completed-message latency diagnostics, per-direction queue samples,
UDP errors and final complete reports or explicit failures.

For a common observation interval, verified byte increments b_i determine
Jain's index J = (sum b_i)^2 / (n * sum b_i^2). Store numerator/denominator;
zero progress has no defined fairness index. Only fully validated messages
contribute. A failed finite workload has no completed-exchange goodput or RTT
percentile. Its progress and completed-message diagnostics remain lower-bound
observations, never successful partial reports. A sample of queue backlog is
an observation, not a proof of its continuous-time maximum.

## Acceptance and rejection

A recovery correction requires matched evidence, all original cases passing,
and an explained fairness/latency/queue-growth tradeoff. No numerical regression
tolerance is supplied, so measurements alone do not authorize inventing one.
Any original failure keeps promotion blocked. BBR's upstream experimental label
is retained even if a finite experiment completes. No default changes merely
because an alternative is faster; no larger deadline, smaller payload, socket
bypass, changed loss model or suppressed failure establishes acceptance.

## Alternatives and verification

Hard-coding a controller in the probe would fail to exercise the consumer's
configuration path. A generic endpoint family would propagate variation through
unrelated security/framing types without removing Quinn's existing erased ABI.
An enum selects existing supported factories at construction and preserves one
transport API. Tests check policy selection and real authenticated bytes/FIN;
the full Linux matrix and shared-queue experiments provide performance evidence.

## Collector lifecycle evidence

The first full comparison exposes two cleanup failures while a queue observer
can still own short-lived namespace readers. Join that observer before
namespace-wide process cleanup. An inventoried process can exit before kill;
accept a failed kill only after a fresh kernel inventory proves the owned
namespace empty. A remaining process preserves the actual cleanup error.
Log primary probe errors before endpoint drain and retain drain errors as
additional diagnostics. None of these lifecycle corrections changes workload,
security, congestion algorithms or successful-exchange timing. Archive the
unsuccessful experiment and rerun all original cases, without reconstructing
successful reports or splicing adjacent repetitions into an acceptance matrix.

References: [Quinn's controller catalog](https://docs.rs/quinn/0.11.12/quinn/congestion/index.html),
[RFC 9002](https://www.rfc-editor.org/rfc/rfc9002),
[ADR 0075](0075-quic-multiplexed-sessions.md).

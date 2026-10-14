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
The timer timestamp is taken immediately before the existing waker dispatch;
trace processing and subsequent protocol scheduling can add delay. PTO
`backoff` stores Quinn's count before increment (the binary-backoff exponent),
not a duration or multiplier. `success` before drain describes exchange
completion; a later drain failure still makes the process pair fail.

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
Pass the single public diagnostic switch explicitly after `sudo ip netns exec`;
sudo does not preserve the controller's environment by default. Every traced
shard first executes an extra real original-domain 1-stream/24-byte/128-message
mutual-TLS preflight and audits both child traces. Preflight is separate from
the complete acceptance inventory; each subsequent case resets the original
qdisc and seed. Missing child traces reject the study before the full run.

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

## Complete evidence and remaining decision

The [October 14 archive](../../moirai-quic/measurements/recovery/2026-10-14/README.md)
retains security-patched 140d60cf's complete 108 cases per policy, all 30 shared-queue
experiments and independently audited child preflights/traces. CUBIC/NewReno
remain 105/108; experimental BBR remains 108/108. The independent uninstrumented
original gate has the same source tree and reproduces the three deadlines.
Failed environment-filtered and canceled inputs remain excluded from diagnostic
acceptance; no case inventory is assembled across studies.

Six early server failures never reach established state while clients record
one Handshake-space PTO followed by repeated Data-space PTOs. Established
throughput failures instead retain authenticated exchange, congestion/pacing
events and stream-flow-credit notifications. These are distinct failure classes.
Driver dispatch lateness does not measure later protocol execution, and event
counts do not identify the dominant blocked duration.

BBR reaches a sampled 10,313,912-byte congestion window in the patched original
loss workload; preceding 733afc9f records 133,608,526 bytes. A congestion window
is an allowed flight limit, not actual flight or allocated memory.
Pinned BBR compares dimensionless cwnd_gain with a byte-valued target window in
its startup growth guard. While that branch repeats, w(n)=w(0)+sum(acked bytes).
[Upstream PR 2798](https://github.com/quinn-rs/quinn/pull/2798) addresses startup
growth and application-limited bandwidth samples but remains open/unmerged.
Internal mode/target/filter state is unobserved, so the study does not prove
that guard explains all observed growth or unfairness. No upstream patch or
default replacement is adopted. Obtain per-space ACK/CRYPTO, flight, flow-credit
and pacing timelines before selecting and measuring a focused correction.

## Security-patch revalidation boundary

Completed required CI at 733afc9f rejects Rustls 0.23.45 for four published
advisories. Adopt the released 0.23.46 security patch through one shared minimum;
the standalone QUIC provider depends on and exposes that same Rustls crate.
Caller trust, TLS/ALPN, early-data policy, resource bounds and deadlines remain
unchanged. Preserve the measured preceding stack and repeat every original
policy case, competing-flow experiment and independent CUBIC gate after the
crypto dependency changes. No advisory suppression or cross-source case splicing
can establish acceptance. Public API compatibility is verified with the patch
override; transport security tests exercise actual peers.

Revalidation at 140d60cf repeats all six shards and the independent original
CUBIC gate without substituting preceding-stack cases. CUBIC/NewReno remain
105/108 and BBR 108/108. Every applicable non-performance required job passes,
including Linux/Windows native QUIC, workspace, supply-chain and lock integrity.
The performance gate and aggregate result remain failed. Current mixed BBR/CUBIC
sampled Jain indices span 0.557124–0.581728; BBR completes while CUBIC reaches
the deadline in all three loss experiments. Six early server handshakes and
15 other competing pair deadlines remain distinct. All controlled-loss and
competing UDP-error deltas are zero; three receive-buffer drops in one successful
unshaped NewReno client do not explain the controlled-loss failures. Both complete
studies, independent gates and every unsuccessful input remain independently
replayable and hashed. No recovery-policy correction is accepted.

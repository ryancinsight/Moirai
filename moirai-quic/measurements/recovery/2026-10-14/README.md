# Recovery trace investigation — 2026-10-14

MOI-QUIC-RECOVERY-001 remains open; CUBIC remains the production default and
the independent original 108-case promotion gate remains enforced.
[ADR 0080](../../../../docs/adr/0080-recovery-trace-evidence.md) specifies the
opt-in observation path. No recovery correction is selected by local tests.

`loopback-trace.json.gz` retains all 81 real authenticated local process pairs:
3 stream counts x 3 payload sizes x 3 repetitions x 3 stock policies. Every
pair compares every application-message byte and completes acknowledged FIN.
All 162 peer traces contain established/authenticated/exchange/completion/drain
evidence; the shared timer driver records 503 wakes. The archive records its
binary digest and the hashes of the measured source files. It is a pre-commit
macOS diagnostic verification, not shaped-network or promotion evidence.
Two parser tests recompute complete reports and reject missing observations,
changed phase histories, counter corruption and missing PTO inventories.

The initial local observation experiment counts ready polls of Quinn's async
timer and records zero across 81 complete pairs. Pinned Quinn first compares
monotonic time with the protocol deadline and normally handles expiration
without polling a ready timer. That observation path is rejected and removed.
The retained implementation observes actual shared timer-driver wakes instead.
It includes QUIC, operation and observation timers and excludes later protocol
scheduling. Event counts do not measure blocked duration, and PTO flight bytes
do not supply a continuous flight history.

The recovery workflow enables `MOIRAI_QUIC_RECOVERY_TRACE=1` for the complete
matched study: 108 original pairs per policy and all 30 shared-queue experiments.
The switch is passed explicitly inside each owned namespace after sudo. Each
shard first validates both real peers' traces in an additional original-domain
1-stream/24-byte/128-message mutual-TLS preflight. That check never contributes
to acceptance counts; each later case resets the original queue and seed.
The collector drains server output during client execution so trace volume
cannot apply pipe backpressure to transport. Network profiles, queue limits,
aggregation, seed, mutual TLS, byte validation and 5/60-second operation/pair
deadlines remain unchanged. `scripts/quic_trace_audit.py` first applies the
existing full-inventory/network/byte/rate/latency/queue audit, then verifies
trace counters, event inventories, phase histories and primary outcomes.

The first Linux diagnostic run, [38583505199](https://github.com/ryancinsight/Moirai/actions/runs/38583505199),
uses integrated source d946459c. Its controller records the opt-in intent but
sudo removes the environment variable before either probe starts. Independent
trace auditing rejects the absence of child observations. This is an unsuccessful
diagnostic experiment; its actual byte-validating stock-policy runs do not
provide handshake/flight/pacing evidence. The explicit switch and authenticated
preflight correct collection only, with no transport-policy or deadline change.

`archives.json` records hashes of retained compressed and decompressed inputs.
The [preceding studies](../2026-10-13/README.md) retain the uninstrumented stock
comparison and all unsuccessful experiments. Instrumented measurements cannot
replace the independent original CUBIC acceptance run.

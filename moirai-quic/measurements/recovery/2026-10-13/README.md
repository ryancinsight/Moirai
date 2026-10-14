# QUIC congestion-policy investigation — 2026-10-13

MOI-QUIC-RECOVERY-001 remains open. CUBIC remains the production default and
the original 108-case promotion gate remains enforced. No congestion-recovery
correction or controller replacement is established by these experiments.

## Corrected complete comparison

Source `8e9dd137e7cbf37c643b67b724c4f1a945bb6021`,
[Actions run 38504359446](https://github.com/ryancinsight/Moirai/actions/runs/38504359446).
This is a fresh complete study after the collector lifecycle correction. Its
six shards use one clean source tree, dependency lock, platform and pinned
netem tool. No case or repetition is imported from the preceding study.
The independent audit verifies all original settings and every complete report.
There are no collector failures; every role of all 384 pairs retains zero
UDP-error deltas. Both peers still compare every application-message byte
and complete acknowledged FIN; failures never emit successful partial reports.

| Stock policy | Complete verified pairs | Original large loss case | Complete application bytes per direction |
|---|---:|---|---:|
| CUBIC | 105 / 108 | all three reach 60 seconds | 1,844,797,440 |
| NewReno | 105 / 108 | all three reach 60 seconds | 1,844,797,440 |
| Experimental BBR | 108 / 108 | all three complete | 2,247,598,080 |

For 16 streams and 65,560-byte messages, medians over all three complete pairs
are below. A failed case has no completed-exchange rate or percentile.

| Profile | Policy | Application Mbit/s | Client RTT p95, ms |
|---|---|---:|---:|
| 10 ms, 100 Mbit/s | CUBIC | 79.058477 | 151.896802 |
| 10 ms, 100 Mbit/s | NewReno | 79.066128 | 162.152973 |
| 10 ms, 100 Mbit/s | Experimental BBR | 77.021295 | 161.515831 |
| 50 ms, 100 Mbit/s | CUBIC | 35.776186 | 317.812962 |
| 50 ms, 100 Mbit/s | NewReno | 33.234380 | 321.042282 |
| 50 ms, 100 Mbit/s | Experimental BBR | 37.059911 | 226.006527 |
| 10 ms, 1% loss, 100 Mbit/s | CUBIC | deadline exceeded | unavailable |
| 10 ms, 1% loss, 100 Mbit/s | NewReno | deadline exceeded | unavailable |
| 10 ms, 1% loss, 100 Mbit/s | Experimental BBR | 74.241825 | 181.234076 |

BBR's median large-loss exchange takes 14.468058079 seconds. The last one-second
CUBIC client observations count 1389/1315/1401 complete validated echoes;
NewReno counts 832/814/844, out of 2048. Large-loss sampled queue maxima are
55,278 bytes for CUBIC and NewReno and 808,029 bytes for BBR. Continuing progress
and zero socket errors remain consistent with the observed recovery limitation,
without establishing its complete cause.

The corrected shared-queue study again executes all 30 experiments / 60 pairs:
39 complete pair reports and 21 failures. Seventeen reach the pair deadline;
four fail early. All four early failures now retain the originating server
`QUIC accept deadline elapsed` error. `QuicEndpoint::accept` includes incoming
admission and handshake completion; this failure precedes server stream
admission. The logs distinguish that stage from subsequent client receive or
peer-close errors, but do not prove the underlying handshake recovery cause.

| Corrected loss-profile pairing | Sampled Jain index, repetitions 0/1/2 | Largest observed backlog per repetition, bytes |
|---|---|---|
| CUBIC / CUBIC | 0.999126, 0.999087, 0.998519 | 107,568; 86,652; 111,176 |
| NewReno / CUBIC | 0.500000, 0.500000, 0.955338 | 69,088; 44,820; 94,122 |
| NewReno / NewReno | 0.995376, 0.500000, 0.500000 | 105,289; 56,772; 64,242 |
| Experimental BBR / CUBIC | 0.598001, 0.568881, 0.572124 | 766,133; 813,562; 797,691 |
| Experimental BBR / BBR | 0.999540, 0.999866, 0.999976 | 774,674; 1,048,345; 1,037,137 |

Every loss-profile BBR/CUBIC experiment completes BBR's workload while its
CUBIC competitor reaches the deadline. BBR's three complete client RTT p95
values range from 191.554117 to 198.542538 ms. BBR/BBR completes both workloads,
with client RTT p95 from 293.268283 to 333.671371 ms. Every competing queue
reports zero backlog after cleanup. The exact sampled fairness fractions,
common intervals, observation ages, queue slopes and both peer diagnostics
remain in the input archives and computed audit. Sampling and finite-workload
limitations below apply to both studies.

**Decision:** NewReno does not close recovery. Experimental BBR improves the
original loss workload but supplies a mixed-policy fairness and queue tradeoff
that does not justify a production default change. No regression tolerance is
invented. The stock initial-window difference and upstream experimental status
remain explicit. Keep CUBIC as default, retain MOI-QUIC-RECOVERY-001 and block
provider promotion. Metis receives no provider-pin or transport-policy change.
Further recovery work needs handshake/flight/pacing evidence and a justified
candidate measured under these same original acceptance conditions.

The independent [original CUBIC acceptance run 38504359933](https://github.com/ryancinsight/Moirai/actions/runs/38504359933)
also executes all 108 pairs: 105 complete exact-byte reports and the same three
large-loss 60-second timeouts. Its checkout is merge revision
`b76ca0af52bcb52597ecf74403f2098c560a591f`, whose Git tree
`65afa7712e2850b0d70bc47b384b06b5270efa3b` equals provider `8e9dd137`.
`8e9dd137-default-matrix.json.gz` preserves that independent failed gate;
both peers' report values and all successful kernel byte deltas are verified.
The required Verification result fails. This is a measured promotion block.

## First complete comparison

Source `3e353caf4f2c805fd30b3556c11162461f3b958e`,
[Actions run 38501872894](https://github.com/ryancinsight/Moirai/actions/runs/38501872894).
Three matrix jobs each compare all policies within the same runner, rotating
policy order between repetitions. Together they execute all 108 original real
mutual-TLS process pairs per policy. Three supplementary jobs execute 30
shared-queue experiments containing 60 real process pairs. Ubuntu 24.04,
Linux 6.17.0-1022-azure, Rust 1.97.0, the original iproute2 commit, netem seed,
packet aggregation, profiles, 128 messages, sizes, stream counts, five-second
operation deadlines and 60-second pair deadlines remain unchanged.

| Stock policy | Complete verified experiment reports | Failed experiments | Complete application bytes per direction |
|---|---:|---:|---:|
| CUBIC | 105 / 108 | 3 | 1,844,797,440 |
| NewReno | 104 / 108 | 4 | 1,844,785,152 |
| Experimental BBR | 107 / 108 | 1 | 2,247,595,008 |

All three CUBIC and NewReno maximum-frame/16-stream loss repetitions exceed
60 seconds. The last one-second client observations count 1425/1370/1348
validated echoes for CUBIC and 964/903/822 for NewReno, out of 2048. These
are lower-bound progress observations, not completed-exchange rates or RTTs.

The extra NewReno failure (50 ms, four streams, 24 bytes, repetition 2) and
BBR failure (50 ms, one stream, 24 bytes, repetition 2) report collector
`kill -KILL` errors after a namespace process exits. The queue observer can
still own short-lived counter-reader processes during pair cleanup. These
are failed experiments with obscured pair outcomes, not established controller
failures. No successful report is reconstructed from queue bytes or adjacent
repetitions. The corrected study above supplies a fresh independent comparison.

For the successfully completed BBR maximum-frame/16-stream loss case, median
client application goodput is 75.413700 Mbit/s, exchange duration is
14.243234703 s and client RTT p95 is 182.209223 ms. The sampled queue maximum
across its repetitions/directions is 843,772 bytes; CUBIC's corresponding
sampled maximum is 71,712 bytes, with no completed-exchange percentile.

## Competing-flow evidence

Every experiment uses two original maximum workloads on the same pair of
100 Mbit/s queues, symmetric 10 ms delay, both without loss and with 1% loss.
Client launch follows a readiness barrier. Thirty experiments preserve all
39 complete pair reports and 21 failures (17 pair timeouts and four early
peer application closes). Every loss-profile BBR/CUBIC repetition completes
BBR's workload while the CUBIC competitor reaches the pair deadline.

| Loss-profile pairing | Sampled Jain index, three repetitions | Largest observed backlog in each repetition, bytes |
|---|---|---|
| CUBIC / CUBIC | 0.996119, 0.500000, 0.986380 | 82,170; 55,278; 109,140 |
| NewReno / CUBIC | 0.930353, 0.940075, 0.500000 | 85,158; 80,676; 77,688 |
| NewReno / NewReno | 0.500000, 0.995285, 0.500000 | 118,026; 88,146; 50,796 |
| Experimental BBR / CUBIC | 0.559527, 0.571102, 0.586669 | 661,038; 819,750; 662,674 |
| Experimental BBR / BBR | 1.000000, 0.999964, 0.983725 | 1,249,126; 1,141,321; 1,055,436 |

Jain's index uses observed fully validated byte-counter increments over the
common sampled interval. The audit retains exact numerator/denominator,
interval endpoints and boundary observation ages; it does not interpolate
unobserved progress. Early failed flows can limit the common interval to a
few seconds. These are finite-workload observations, not long-lived TCP or
wide-area fairness guarantees. The first archive's four early client closes
do not establish the originating server error because its stderr is empty.

BBR/BBR loss experiments complete both workloads. Across their six complete
client reports, RTT p95 ranges from 287.739685 to 347.909047 ms. BBR/CUBIC's
last observed CUBIC completed-message p95 diagnostics range from 1.635760543
to 1.988461813 s; these are partial-run diagnostics, not completed-exchange
percentiles. All competing queues report zero backlog after cleanup. One-second
sampling and retained least-squares slopes supply queue-growth evidence; the
sampled maxima neither bound continuous-time peaks nor prove unbounded growth.

All controlled delay/loss and competing experiments record zero UDP error
deltas. One unshaped CUBIC maximum-frame/16-stream repetition records
`InErrors=2` and `RcvbufErrors=2` at the server; these counters overlap and do
not establish four separate drops. That pair still validates every byte. This
does not reclassify the loss-profile recovery failures as socket exhaustion.

## Interpretation and corrections

Stock NewReno does not close the measured recovery limit. Experimental BBR
improves the maximum loss workload, while its mixed-policy fairness and queue
tradeoff prevent selecting it as the default from this evidence. Upstream's
experimental classification remains binding. Stock initial windows differ:
12,000 bytes for CUBIC/NewReno versus 240,000 for BBR; BBR also randomizes its
gain-cycle offset internally. The netem seed does not seed that RNG. The study
therefore compares complete stock configurations, not isolated causal effects.

The admitted maximum workload needs 134,266,880 application bytes per direction;
even excluding setup/drain, finishing in 60 seconds requires more than
17,902,250 bits/s. Contracting congestion windows and continued verified progress
remain consistent with recovery-limited throughput, but do not establish its
complete cause. No buffer enlargement, altered workload/deadline, disabled
validation or application replay is introduced.

The collector correction joins the queue observer before namespace-wide process
cleanup. A failed kill is accepted only when a fresh kernel inventory proves
that the owned namespace is empty; remaining processes preserve cleanup errors.
The probe logs originating failure before endpoint drain and preserves that
error if drain also fails. A real native mutual-TLS negative probe with
deliberately mismatched stream counts records the receive-message deadline,
both failed peers and no successful partial report. This is diagnostic
verification, not a substitute workload or performance result.

## Reproduction and integrity

The twelve `3e353caf-*.json.gz` and `8e9dd137-*.json.gz` matrix/fairness input
archives retain both complete studies, including the first study's cleanup
failures. `archives.json` records canonical decompressed
JSON hashes, original downloaded JSON hashes, source tree and Actions run.
Canonical JSON uses sorted keys, compact separators and a final newline; gzip
uses `mtime=0`. Each source prefix also has reproducible `*-audit.json.gz`
computed evidence.
The thirteenth input archive preserves the separate original default gate.

```sh
cargo build --locked --release -p moirai-quic --example quic_load
# On Linux with namespace administration; repeat for repetition 0, 1 and 2:
python3 scripts/quic_recovery.py --binary target/release/examples/quic_load \
  --tc /absolute/path/to/pinned/iproute2/tc/tc --suite matrix --repetition 0 \
  --output output/quic-recovery/matrix-0.json
python3 scripts/quic_recovery.py --binary target/release/examples/quic_load \
  --tc /absolute/path/to/pinned/iproute2/tc/tc --suite fairness --repetition 0 \
  --output output/quic-recovery/fairness-0.json
python3 scripts/quic_recovery_audit.py \
  moirai-quic/measurements/recovery/2026-10-13/8e9dd137-matrix-*.json.gz \
  moirai-quic/measurements/recovery/2026-10-13/8e9dd137-fairness-*.json.gz \
  --output output/quic-recovery/audit.json
```

The independent auditor requires every original case and all six shards from
one clean source tree/lock/platform/tool. It recalculates every successful
report, checks actual queue seed/rate/delay/loss/limit, verifies kernel byte
coverage, retains UDP errors and rejects missing inventory or partial success.
The separate original CUBIC acceptance workflow continues to enforce promotion.

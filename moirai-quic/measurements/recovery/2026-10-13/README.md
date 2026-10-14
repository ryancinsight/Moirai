# QUIC congestion-policy investigation — 2026-10-13

MOI-QUIC-RECOVERY-001 remains open. CUBIC remains the production default and
the original 108-case promotion gate remains enforced. No congestion-recovery
correction or controller replacement is established by these experiments.

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
repetitions. A fresh complete comparison is required after collector correction.

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

The six `3e353caf-*.json.gz` input archives retain the first comparison,
including both cleanup failures. `archives.json` records canonical decompressed
JSON hashes, original downloaded JSON hashes, source tree and Actions run.
Canonical JSON uses sorted keys, compact separators and a final newline; gzip
uses `mtime=0`. `3e353caf-audit.json.gz` is reproducible computed evidence.

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
  moirai-quic/measurements/recovery/2026-10-13/3e353caf-matrix-*.json.gz \
  moirai-quic/measurements/recovery/2026-10-13/3e353caf-fairness-*.json.gz \
  --output output/quic-recovery/audit.json
```

The independent auditor requires every original case and all six shards from
one clean source tree/lock/platform/tool. It recalculates every successful
report, checks actual queue seed/rate/delay/loss/limit, verifies kernel byte
coverage, retains UDP errors and rejects missing inventory or partial success.
The separate original CUBIC acceptance workflow continues to enforce promotion.

# Controlled QUIC measurements — 2026-10-13

This directory preserves canonical JSON in deterministic gzip streams. It is
measurement evidence for ADR 0075; archived failures remain failures.

## Initial baseline

Source: `545685574acb385f08e1c6fa875f0ccf6cf4cca3`;
[CI run 38459278652](https://github.com/ryancinsight/Moirai/actions/runs/38459278652).
Ubuntu 24.04, Linux 6.17.0-1022-azure, Rust 1.97.0 and pinned iproute2 6.17
measure real mutual-TLS process pairs in isolated namespaces. Netem runs on
both directions with the declared seed; counters preserve packet aggregation.

The initial controller stops at the first failure: 105 pairs validate every
byte, percentile and integer rate; pair 106 exceeds the 60-second budget.
Two repetitions remain unmeasured in this initial archive. Successful pairs
validate exactly 1,844,797,440 application-message bytes in each direction.
Independent kernel queue-byte deltas cover at least that declared payload for
every successful pair and direction.

For 65,560-byte messages, median client application goodput in Mbit/s over
three repetitions is:

| Profile | 1 stream | 4 streams | 16 streams |
|---|---:|---:|---:|
| unshaped | 1029.511 | 1819.586 | 2258.210 |
| delay-10ms | 8.955 | 34.847 | 78.076 |
| delay-50ms | 1.813 | 8.451 | 36.921 |
| delay-10ms-loss-1pct | 7.160 | 11.821 | deadline exceeded |

These are application-message rates, excluding setup and endpoint drain.
The raw archive includes all sizes (24, 4,096 and 65,560 bytes), every sample,
nearest-rank percentiles, peer reports and actual queue counters. This matrix
supplies no TCP comparison or wide-area performance guarantee.

## Loss diagnosis

Source: `60db5f8c14940e85b8c25e0dc07c011f9c85bfe1`;
[CI run 38464082170](https://github.com/ryancinsight/Moirai/actions/runs/38464082170).
The diagnostic runs all three repetitions of the original 16-stream,
128-message, 65,560-byte, symmetric 10 ms/1% loss/100 Mbit/s case. Each exceeds
its unchanged 60-second budget. No successful partial-load report is emitted.

The last one-second client snapshots observe 1378, 1339, 1385
complete validated echoes out of 2,048. These are progress lower bounds before
the deadline, not completed-exchange results. Every role/repetition records zero
UDP input, receive-buffer, send-buffer, checksum and memory errors. Recovery
snapshots show packet loss, repeated congestion events and contracting CUBIC
windows while messages continue. This distinguishes the observed recovery limit
from local socket overflow or a stopped driver; it does not prove fairness or
justify selecting another controller for production.

## Complete matrix

[CI run 38464090433](https://github.com/ryancinsight/Moirai/actions/runs/38464090433)
measures all 108 declared process pairs. The checked-out merge revision is
`8168e21f06cb7d53b3c8a42340a8e0b7c05c515d`; its Git tree
`42dac33fed607a92f505ca5b495f319254e649ef` equals published provider
`60db5f8c14940e85b8c25e0dc07c011f9c85bfe1`. This is one coherent measured
source tree, not a combination of the earlier baseline and diagnostic runs.

All 105 successful pairs pass independent count, percentile and rate
recalculation, and both kernel queue-byte deltas cover the declared payload.
They validate exactly 1,844,797,440 application-message bytes per direction.
All three remaining pairs are the unchanged maximum-frame, 16-stream loss case
and reach the 60-second deadline. Every role/case has zero kernel UDP input,
receive-buffer, send-buffer, checksum and memory-error deltas. The gate remains
failed; failed pairs never become successful partial reports.

For 16 streams and 65,560-byte messages, the median across three complete
process-pair samples is:

| Profile | Client goodput (Mbit/s) | Client RTT p95 (ms) |
|---|---:|---:|
| unshaped | 1667.214 | 8.905 |
| delay-10ms | 75.604 | 167.894 |
| delay-50ms | 36.186 | 323.757 |
| delay-10ms-loss-1pct | deadline exceeded | no completed-exchange percentile |

The unshaped values use a virtual link; they do not establish physical-link
capacity. These measurements include probe validation and the stated workload.

## Archive integrity

SHA-256 below hashes decompressed canonical UTF-8 JSON (sorted keys, compact
separators and one final newline). Gzip uses `mtime=0` for reproducibility.

| Archive | JSON SHA-256 |
|---|---|
| `54568557-initial.json.gz` | `95482fa39e77d9292390ba4155b1d553b9140e6fd744f39a5eae037de6076af3` |
| `60db5f8c-diagnostic.json.gz` | `8b369ea17425a3c60b22126e5de5df460f0eeebf731972d29759e59596a30a4d` |
| `8168e21f-matrix.json.gz` | `151590a16ee8c45ffe07b626a5e119bab89558f2417e349f990ec85e9b0bb001` |

Reproduce the complete matrix from a standalone checkout:

```sh
cargo build --locked --release -p moirai-quic --example quic_load
python3 scripts/quic_profiles.py --binary target/release/examples/quic_load \
  --network netem --tc "$RUNNER_TEMP/quic-iproute2/tc/tc" --output output/quic/profiles.json
```

The runner requires namespace administration and the pinned tool from the CI
workflow. Public fixture credentials authorize only these controlled probes.
For diagnosis, append `--case delay-10ms-loss-1pct:16:65560`; normal CI still
runs the complete matrix. The full collector preserves every failure and fails
its gate if any declared pair fails. Promotion remains gated by this measured
limit and required review/checks.

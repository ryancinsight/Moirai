# Recovery trace investigation — 2026-10-14

MOI-QUIC-RECOVERY-001 remains open. CUBIC remains the production default;
the original 108-case promotion gate remains enforced. No recovery correction
or production controller replacement is justified by this study. Metis receives
no provider-pin or transport-policy change.

The 733afc9f measurements below use Rustls 0.23.45. That source's completed
supply-chain job subsequently reports RUSTSEC-2026-0346/0347/0348/0349.
The shared dependency and standalone QUIC provider now require the released
[Rustls 0.23.46 security patch](https://github.com/rustls/rustls/releases/tag/v%2F0.23.46).
Both locks change only that external package identity. TLS/ALPN/trust, early-data
policy, flow budgets and deadlines remain unchanged. The patched stack requires
a fresh complete study and independent original gate; these earlier measurements
do not stand in for security-patched acceptance.

## Corrected complete study

Source `733afc9fcce3b70b4ceef512914e83f6e6142841`,
[run 38585867253](https://github.com/ryancinsight/Moirai/actions/runs/38585867253).
All six shards contain a clean source tree, the same lock, platform, pinned
seeded-netem tool and actual original network settings. Every shard first
verifies both child traces in an extra authenticated original-domain preflight.
Those six preflight pairs do not contribute to acceptance counts; each later
case resets the original qdisc/seed. No earlier case fills this study's inventory.

[ADR 0080](../../../../docs/adr/0080-recovery-trace-evidence.md) specifies capture
and inference limits. All 384 matrix/competing pairs retain both peers' actual
protocol observations. All namespace UDP-error deltas are zero. Successful
pairs still compare every application-message byte and acknowledge FIN;
failures never become successful partial reports. Five-second operation and
60-second process-pair deadlines, profiles, aggregation, queue limits, 128
messages and stream/payload domains remain unchanged.

| Stock policy | Complete original pairs | Complete application bytes per direction |
|---|---:|---:|
| CUBIC | 105 / 108 | 1,844,797,440 |
| NewReno | 105 / 108 | 1,844,797,440 |
| Experimental BBR | 108 / 108 | 2,247,598,080 |

Stock initial windows remain 12,000 bytes for CUBIC/NewReno and 240,000 for
BBR. BBR randomizes its gain-cycle offset internally; the netem seed does not
seed that RNG. This compares complete stock configurations rather than isolating
the steady-state algorithm from startup behavior.

All three CUBIC and NewReno maximum-frame/16-stream loss cases reach the pair
deadline. Last client observations count 1329/1419/1353 validated echoes for
CUBIC and 837/962/877 for NewReno, out of 2048. These are progress observations,
not complete-exchange rates or latencies. BBR's three complete loss cases have
median application goodput 73.365963 Mbit/s, exchange duration 14.640781568 s,
and client RTT p95 182.192083 ms.

For the same 16-stream/65,560-byte workload, complete-pair medians are:

| Profile | Policy | Application Mbit/s | Client RTT p95, ms |
|---|---|---:|---:|
| 10 ms, 100 Mbit/s | CUBIC | 77.819178 | 162.146473 |
| 10 ms, 100 Mbit/s | NewReno | 81.087784 | 164.350747 |
| 10 ms, 100 Mbit/s | Experimental BBR | 73.860042 | 161.027896 |
| 50 ms, 100 Mbit/s | CUBIC | 34.468097 | 328.713770 |
| 50 ms, 100 Mbit/s | NewReno | 36.777243 | 321.344431 |
| 50 ms, 100 Mbit/s | Experimental BBR | 37.528307 | 224.112104 |

## Established-session observations

Both peers of every failed maximum loss case reach the authenticated exchange
phase. None records an Initial- or Handshake-space PTO. Per-peer counter ranges
at the last pre-drain outcome or last sample across all three repetitions are
below; they count events rather than blocked
durations. PTO flight bytes describe only PTO instants.

| Policy | Congestion blocks | Pacing blocks | Stream-data-blocked notifications | Data-space PTOs | Largest driver lateness, ns |
|---|---:|---:|---:|---:|---:|
| CUBIC | 23,051–26,750 | 11,274–11,896 | 1,149–1,257 | 5–10 | 446,750 |
| NewReno | 13,647–15,375 | 4,905–6,164 | 728–838 | 26–29 | 426,575 |

Recovery, pacing and flow-credit pressure coexist; these counts do not identify
their respective wall-time costs or prove a complete cause. Shared-driver
lateness is sampled at dispatch, before trace handling and subsequent protocol
scheduling. It cannot exclude delayed protocol execution or incorrect timer
selection. No larger flow-control window, socket queue or deadline is adopted.

Experimental BBR's largest sampled congestion window in this original loss
workload is 133,608,526 bytes at the server in repetition 2. A congestion window
is an allowed flight limit, not a measurement of allocated memory or actual
in-flight bytes. The pinned startup calculation compares dimensionless
`cwnd_gain` with byte-valued `target_window` before increasing the window by
acknowledged bytes. While that branch repeats, w(n) = w(0) + sum(acked bytes).
Upstream [PR 2798](https://github.com/quinn-rs/quinn/pull/2798) addresses startup
window growth and admission of application-limited bandwidth samples; it remains
open and unmerged when checked on October 14. This study does not capture BBR's
internal mode/target/filter state and does not attribute all growth or unfairness
to that guard. No unreviewed upstream patch is adopted.

The pinned CUBIC implementation already includes the
[fast-convergence correction](https://github.com/quinn-rs/quinn/pull/2641).
Reapplying that fix does not supply a new candidate.

## Shared queues and early handshake failures

All 30 experiments / 60 pairs run: 39 pairs complete, 18 reach the process-pair
deadline, and three fail early at the server accept/handshake deadline. Each
early failure has no server established event, one client Handshake-space PTO
followed by five or six Data-space PTOs, and no validated client echo. The
servers record five shared-driver wakes each; their largest observed dispatch
lateness is 109,881 ns. This distinguishes an uncompleted server handshake from
the established-session throughput failures. Pending CRYPTO bytes, per-space
ACKs, packet ordering and complete flight state remain unobserved.

Every loss-profile BBR/CUBIC experiment completes BBR while CUBIC reaches the
pair deadline. Sampled Jain indices, rounded to six decimals, and largest
observed per-direction backlogs are:

| Pairing | Repetition 0 / 1 / 2 sampled Jain index | Largest backlog per repetition, bytes |
|---|---|---|
| CUBIC / CUBIC | 0.500000 / 0.998870 / 0.500000 | 69,079 / 92,628 / 68,724 |
| NewReno / CUBIC | 0.959742 / 0.967970 / 0.500000 | 77,688 / 118,026 / 137,448 |
| NewReno / NewReno | 0.991190 / 0.999885 / 0.998563 | 134,890 / 80,676 / 80,676 |
| Experimental BBR / CUBIC | 0.582141 / 0.537422 / 0.568215 | 887,471 / 839,473 / 626,909 |
| Experimental BBR / BBR | 0.999097 / 0.999706 / 0.997978 | 1,315,890 / 942,801 / 1,332,046 |

Exact fractions, common intervals, boundary ages, queue samples and slopes are
retained. The 0.5 indices above include a failed handshake and cannot describe
steady-state fairness. BBR/CUBIC complete BBR client p95 values span
187.187593–195.229051 ms; BBR/BBR completes all six flows with client p95
283.634985–320.799160 ms. All competing queues record zero backlog after
cleanup. Finite sampled maxima neither bound continuous peaks nor establish
unbounded queue growth or long-lived TCP/WAN fairness. Instrumentation can
perturb execution; the independent uninstrumented gate remains authoritative.

## Independent original gate

[Run 38585868810](https://github.com/ryancinsight/Moirai/actions/runs/38585868810)
executes all original 108 CUBIC pairs without protocol capture: 105 complete
exact-byte reports, the same three 60-second maximum loss failures and zero
UDP-error deltas. Its merge checkout `3a59abd35557a878c1e535cfcc3fdadbff2d48f1`
has tree `fc4f3f017fbb7e7a21ff382bb5c8c4f114c412ea`, identical to provider
`733afc9f`. The performance gate fails. Native Linux/Windows, the full workspace
and ADR checks pass at that source. A separate fuzz-lock failure omits the new
regular tracing dependency edge; offline resolution adds that edge without
changing any package identity, and the repaired lock resolves under `--locked`.

## Retained unsuccessful experiments

- `d946459c` / [run 38583505199](https://github.com/ryancinsight/Moirai/actions/runs/38583505199):
  all 108 pairs per policy and all 30 competing experiments run. Stock complete
  counts are 105/105/108, but all 768 child roles lack protocol observations:
  sudo filters the controller's opt-in variable. The independent stock-workload
  audit passes; trace acceptance fails. Explicit namespace-local forwarding of
  the single public flag and authenticated preflight correct collection only.
- `62876270` / [run 38584037658](https://github.com/ryancinsight/Moirai/actions/runs/38584037658):
  a redundant run of the same invalid trace path is canceled after diagnosis.
  Matrix shards retain 72/75/76 records; all 30 competing records are retained.
  Incomplete/canceled inputs never supply missing acceptance cases.
- Superseded original-gate runs 38583505613 and 38584038224 retain their
  available partial inputs. Neither replaces the complete independent
  38585868810 acceptance matrix.
- The initial local ready-poll experiment records no timer-ready polls across
  81 byte-complete pairs. Pinned Quinn checks monotonic time before polling its
  timer and normally handles expiry directly. That observation path is rejected
  and removed; actual shared-driver dispatch observations replace it.

`loopback-trace.json.gz` retains 81 real authenticated local diagnostic pairs
and 503 shared-driver wakes, with binary and measured-source hashes. All 162
peer traces reach authenticated exchange/completion/drain. This is pre-commit
macOS diagnostic verification, not shaped-network or promotion evidence.
The integrated uninstrumented local CUBIC probe also verifies all 27 cases.
The preceding [October 13 studies](../2026-10-13/README.md) remain unchanged.

## Reproduction and decision

`archives.json` hashes every retained compressed/decompressed input and audit.
Run the Python contract tests, then independently recompute:

```sh
python3 -m unittest discover -s scripts/tests -p 'test_quic*.py'
python3 scripts/quic_trace_audit.py moirai-quic/measurements/recovery/2026-10-14/733afc9f-{matrix,fairness}-*.json.gz --output /tmp/quic-trace-audit.json
python3 scripts/quic_recovery_audit.py --original-gate moirai-quic/measurements/recovery/2026-10-14/733afc9f-default-matrix.json.gz --output /tmp/quic-original-audit.json
```

No default change is accepted. NewReno does not close the workload limit.
Experimental BBR completes it but retains mixed-policy fairness, queue and
window-growth concerns; no numerical regression tolerance is invented. Next,
correlate per-space handshake ACK/CRYPTO state with PTO selection, and obtain
established-session flight, flow-credit and pacing timelines before selecting
and measuring a focused recovery candidate. All original requirements remain
binding, and protected provider/Metis promotion remains blocked.

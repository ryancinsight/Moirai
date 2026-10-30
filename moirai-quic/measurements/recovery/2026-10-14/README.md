# Recovery trace investigation — 2026-10-14

MOI-QUIC-RECOVERY-001 remains open. CUBIC remains the production default;
the original 108-case promotion gate remains enforced. No recovery correction
or production controller replacement is justified by this study. Metis receives
no provider-pin or transport-policy change.

## Security-patched complete study

Source `140d60cf43e152b816ccd8d4ff5ac12a0e53dfdf`, tree
`1c1f2746e701e27d2db342e59f43aa00c577076a`,
[run 38594187033](https://github.com/ryancinsight/Moirai/actions/runs/38594187033).
The shared dependency and standalone QUIC provider require the released
[Rustls 0.23.46 security patch](https://github.com/rustls/rustls/releases/tag/v%2F0.23.46)
after required CI reports RUSTSEC-2026-0346/0347/0348/0349 against 0.23.45.
Both locks change only that external package identity. TLS/ALPN/trust, early-data
policy, flow budgets and deadlines remain unchanged. The patched stack repeats
the complete study and independent original gate; preceding-stack cases never
fill this study's acceptance inventory.

All six shards contain a clean source tree, the same lock, platform, pinned
seeded-netem tool and actual original network settings. Every shard first
verifies both child traces in an extra authenticated original-domain preflight.
Those six preflight pairs do not contribute to acceptance counts; each later
case resets the original qdisc/seed.

[ADR 0080](../../../../docs/adr/0080-recovery-trace-evidence.md) specifies capture
and inference limits. All 384 matrix/competing pairs retain both peers' actual
protocol observations. Successful pairs compare every application-message byte
and acknowledge FIN; failures never become successful partial reports.
Five-second operation and 60-second process-pair deadlines, profiles,
aggregation, queue limits, 128 messages and stream/payload domains remain unchanged.

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
deadline. Last client observations count 1357/1383/1290 validated echoes for
CUBIC and 903/1021/815 for NewReno, out of 2048. These are progress observations,
not complete-exchange rates or latencies. BBR's three complete loss cases have
median application goodput 73.659754 Mbit/s, exchange duration 14.582386865 s,
and client RTT p95 187.939092 ms.

For the same 16-stream/65,560-byte workload, complete-pair medians are:

| Profile | Policy | Application Mbit/s | Client RTT p95, ms |
|---|---|---:|---:|
| 10 ms, 100 Mbit/s | CUBIC | 80.358963 | 162.616712 |
| 10 ms, 100 Mbit/s | NewReno | 77.767034 | 157.311774 |
| 10 ms, 100 Mbit/s | Experimental BBR | 77.307941 | 151.088025 |
| 50 ms, 100 Mbit/s | CUBIC | 36.668019 | 322.274876 |
| 50 ms, 100 Mbit/s | NewReno | 33.562570 | 324.569838 |
| 50 ms, 100 Mbit/s | Experimental BBR | 37.566929 | 313.341907 |

All controlled-loss matrix and competing UDP-error deltas are zero. One
successful unshaped NewReno case (16 streams, 65,560 bytes, repetition 2, client)
records InErrors +3 and RcvbufErrors +3: the counters overlap and describe three
receive-buffer drops, not six. This separate unshaped observation does not
establish socket exhaustion in the controlled-loss failures.

## Established-session observations

Both peers of every failed maximum loss case reach the authenticated exchange
phase. None records an Initial- or Handshake-space PTO. Per-peer counter ranges
at the last pre-drain outcome or last sample across all three repetitions are
below; they count events rather than blocked durations. PTO flight bytes describe
only PTO instants.

| Policy | Congestion blocks | Pacing blocks | Stream-data-blocked notifications | Data-space PTOs | Largest driver lateness, ns |
|---|---:|---:|---:|---:|---:|
| CUBIC | 21,341–24,911 | 10,459–11,888 | 1,116–1,241 | 4–8 | 1,242,382 |
| NewReno | 13,262–16,769 | 4,740–7,076 | 700–897 | 22–34 | 3,369,652 |

Per-role maximum PTO flight sizes range from 36,351–65,340 bytes under CUBIC
and 47,916–66,792 under NewReno. Recovery, pacing and flow-credit pressure
coexist; these counts do not identify their respective wall-time costs or prove
a complete cause. Shared-driver lateness is sampled at dispatch, before trace
handling and subsequent protocol scheduling. It cannot exclude delayed protocol
execution or incorrect timer selection. No larger flow-control window, socket
queue or deadline is adopted.

Experimental BBR's largest sampled congestion window in this original loss
workload is 10,313,912 bytes at the client in repetition 1. The preceding
733afc9f study records 133,608,526 bytes. A congestion window is an allowed
flight limit, not allocated memory or actual in-flight bytes. The pinned startup
calculation compares dimensionless `cwnd_gain` with byte-valued `target_window`
before increasing the window by acknowledged bytes. While that branch repeats,
w(n) = w(0) + sum(acked bytes). Upstream
[PR 2798](https://github.com/quinn-rs/quinn/pull/2798) addresses startup window
growth and admission of application-limited bandwidth samples; it remains open
and unmerged when checked on October 14. This study does not capture BBR's
internal mode/target/filter state and does not attribute all growth or unfairness
to that guard. No unreviewed upstream patch is adopted.

The pinned CUBIC implementation already includes the
[fast-convergence correction](https://github.com/quinn-rs/quinn/pull/2641).
Reapplying that fix does not supply a new candidate.

## Shared queues and early handshake failures

All 30 experiments / 60 pairs run: 39 pairs complete, 15 reach the process-pair
deadline, and six fail early at the server accept/handshake deadline. Each early
failure has no server established event, one client Handshake-space PTO followed
by five or six Data-space PTOs, and no validated client echo. The servers record
five shared-driver wakes each; their largest observed dispatch lateness is
85,146 ns. This distinguishes an uncompleted server handshake from the
established-session throughput failures. Pending CRYPTO bytes, per-space ACKs,
packet ordering and complete flight state remain unobserved.

Every loss-profile BBR/CUBIC experiment completes BBR while CUBIC reaches the
pair deadline. Sampled Jain indices, rounded to six decimals, and largest
observed per-direction backlogs are:

| Pairing | Repetition 0 / 1 / 2 sampled Jain index | Largest backlog per repetition, bytes |
|---|---|---|
| CUBIC / CUBIC | 0.500000 / 0.500000 / 0.998899 | 47,808 / 94,857 / 83,664 |
| NewReno / CUBIC | 0.500000 / 0.500000 / 0.500000 | 270,414 / 534,852 / 53,784 |
| NewReno / NewReno | 0.990764 / 0.500000 / 0.984099 | 70,218 / 79,798 / 89,640 |
| Experimental BBR / CUBIC | 0.581728 / 0.572192 / 0.557124 | 770,105 / 549,799 / 555,349 |
| Experimental BBR / BBR | 0.998046 / 0.999630 / 0.997704 | 1,216,112 / 1,122,243 / 1,100,461 |

Exact fractions, common intervals, boundary ages, queue samples and slopes are
retained. The 0.5 indices above include a failed handshake and cannot describe
steady-state fairness. BBR/CUBIC complete BBR client p95 values span
195.612340–204.914569 ms; BBR/BBR completes all six flows with client p95
302.820976–321.513621 ms. All competing queues record zero backlog after
cleanup. Finite sampled maxima neither bound continuous peaks nor establish
unbounded queue growth or long-lived TCP/WAN fairness. Instrumentation can
perturb execution; the independent uninstrumented gate remains authoritative.

## Independent original gate and required checks

[Run 38594188048](https://github.com/ryancinsight/Moirai/actions/runs/38594188048)
executes all original 108 CUBIC pairs without protocol capture: 105 complete
exact-byte reports, the same three 60-second maximum loss failures and zero
UDP-error deltas. Its merge checkout `80e0a51f2e921baa351ca3fe93972d9eeb2e32d4`
has tree `1c1f2746e701e27d2db342e59f43aa00c577076a`, identical to provider
140d60cf. The performance gate and aggregate verification result fail.
Every other applicable required job passes: workspace, Rust 1.95 floor, Loom,
fuzz, ADR index, supply-chain, Linux/Windows native QUIC, lockfile integrity
and informational SemVer. The archived required-check snapshot records actual
job conclusions; conditionally skipped jobs are not reported as passes.

Local patched validation passes 23 QUIC/trace tests, strict transport diagnostics,
223 patch-level QUIC API checks and 1,341 workspace tests (nine existing ignored
probes). One workspace process-output leak diagnostic in the unchanged executor
retention test does not recur in its focused rerun. This is a test-process output
observation, not evidence of heap leakage. Configured fuzz-target checks and both
canonical locked resolutions pass.

## Retained preceding and unsuccessful experiments

- `733afc9f` / [run 38585867253](https://github.com/ryancinsight/Moirai/actions/runs/38585867253):
  complete pre-security-patch trace study and 30 competing experiments. Policy
  counts remain 105/105/108; competing pairs complete 39/60, with 18 pair
  deadlines and three early server handshakes. Mixed BBR/CUBIC Jain indices
  are 0.582141/0.537422/0.568215; largest BBR/BBR sampled backlog is 1,332,046
  bytes. All UDP-error deltas are zero. The independent original
  [run 38585868810](https://github.com/ryancinsight/Moirai/actions/runs/38585868810)
  has the same source tree and reproduces 105/108. Required CI then exposes the
  Rustls advisories and the fuzz lock's missing regular tracing edge. Both are
  corrected before the fresh 140d60cf study; the previous raw inputs and audits
  remain separate. The [preceding report](https://github.com/ryancinsight/Moirai/blob/140d60cf43e152b816ccd8d4ff5ac12a0e53dfdf/moirai-quic/measurements/recovery/2026-10-14/README.md)
  preserves its complete calculations and inference limits.
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
  available partial inputs. Neither replaces a complete independent matrix.
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

## State-observation increment — ADR 0082

The next diagnostic build adds only immutable observer calls and private helpers
to the exact checksum-verified quinn-proto 0.11.19 source in a separate workspace.
The canonical registry source, workspace locks and original CUBIC job remain
unchanged. Record ACK/CRYPTO metadata, actual per-space flight, stream/connection
credit and timer/PTO state. Retain every handshake state/event and per-second
first/last/extrema for established driver observations; these do not prove
continuous blocked durations. No plaintext or secrets are recorded.

`state-native-rejected-pto.json.gz` retains 81 real byte-complete local pairs
from the first observer. Its off-timer call of the PTO selector can report an
idle candidate and violate the selector's debug precondition. Those selected-space
claims fail audit and do not support a recovery change. The corrected observer
preserves the selector's flight/timer preconditions; `state-native.json.gz`
retains all 81 local policy/shape/repetition pairs with exact bytes, cross-space
flight sums and cross-peer Initial/Handshake CRYPTO generation/consumption.
Both captures retain binary, patch and measured-source hashes; neither supplies
shaped-network acceptance. The full Linux observer study runs at 03dcc4a3 in
[run 38612818404](https://github.com/ryancinsight/Moirai/actions/runs/38612818404);
the unchanged original gate runs separately in
[run 38612819578](https://github.com/ryancinsight/Moirai/actions/runs/38612819578).
All six shards complete and pass independent state/numerical auditing. The source
is `03dcc4a31d7f793e4cb1cb7d96ff74f47033946c`, tree
`48c5d032d581b2a2af78686da7eade7444537947`. CUBIC/NewReno again complete
105/108 each (1,844,797,440 bytes per direction); experimental BBR completes
108/108 (2,247,598,080). The same three maximum loss cases fail under CUBIC and
NewReno. All matrix/competing UDP-error deltas are zero. All 384 process pairs
retain actual engine state and verified child preflights; no earlier case fills
the inventory. The instrumented BBR loss medians are 70.367697 Mbit/s,
15.264604061 s exchange and 187.650529 ms client p95. These diagnostic timings
do not replace pristine-engine acceptance.

The independent original gate runs all 108 uninstrumented CUBIC pairs: 105
complete exact-byte reports, the same three 60-second loss failures and zero
UDP-error deltas. Its merge checkout `631753d691d2fdd1b0b2f2f4ec32746d8c26b3d4`
has the identical tree above. Linux/Windows native, workspace, Rust 1.95 floor,
Loom, fuzz, supply-chain, lockfile, ADR and informational SemVer jobs pass.
The original performance job fails; aggregate Verification result remains queued
in the archived required-check snapshot. This is not a full-check success.

For the three maximum loss repetitions, observed per-role timeline extrema and
last-observed connection credit are:

| Policy / role | Flight range, bytes | Congestion-window range, bytes | Final observed connection-credit range, bytes |
|---|---:|---:|---:|
| CUBIC / client | 0–101,640 | 12,000–120,506 | 630,718–881,290 |
| CUBIC / server | 0–540,868 | 5,880–544,476 | 560,200–703,434 |
| NewReno / client | 0–84,216 | 10,551–88,671 | 521,161–1,045,341 |
| NewReno / server | 0–88,572 | 3,000–287,793 | 204,268–941,563 |

These ranges include startup and retain actual driver-boundary values. Interval
extrema are not continuous bounds; positive final credit does not exclude earlier
credit stalls (each policy reaches zero credit in some observations). Pacing,
credit and congestion events still do not establish blocked wall-time fractions.
The established-session throughput cause remains incomplete.

All 30 competing experiments run: 39/60 pairs complete, six fail at server
handshake/accept and 15 reach the pair deadline. Mixed BBR/CUBIC loss Jain values
are 0.567320/0.583655/0.572130, with sampled queue peaks
633,275/671,276/520,820 bytes. BBR completes and CUBIC times out in all three.
BBR/BBR peaks reach 1,243,444 bytes; completed client p95 spans
303.302522–331.717687 ms. Every competing queue reports zero backlog after
cleanup. These remain finite sampled observations without an invented regression
tolerance, steady-state fairness conclusion or continuous queue guarantee.

The completed competing shards independently validate all 60 actual peer pairs.
All six early failures retain 634 queued client Handshake CRYPTO bytes with zero
Handshake flight and Data PTO selection. The server consumes 2246/2880 client
Handshake bytes and remains in TLS handshake. The focused real UDP experiment
retains 27 local maximum-workload cases per engine: baseline 17/27 complete,
experimental correction 27/27 complete. Each case drops one of the first three
client datagrams containing Handshake packets; all other datagrams pass unchanged.
Every baseline failure exhibits queued Handshake CRYPTO blocked by the shared
window while PTO selects Data. Packet batching varies, so this is a matched
fault rule, not an identical encrypted packet tape or throughput comparison.

The [candidate patch](state/handshake-candidate.patch) makes queued CRYPTO eligible
for PTO and distinguishes confirmation from local TLS completion. It passes all
23 native QUIC contract tests in an isolated build. It remains experimental:
the canonical dependency is unchanged, its complete Linux fairness/latency/queue
matrix is unmeasured, and established-session throughput is a separate limit.
See ADR 0082 for the causal argument and standards references. The exact measured
patch bytes remain in `state/handshake-measured-patch.json.gz`; the displayed
unified diff uses normalized context, and both produce byte-identical engine
source under independent application checks. The additional
[state archive ledger](state/archives.json) retains both repeated experiments,
preceding nine-case experiments and the failed result-file recorder. No failed
or candidate case supplies baseline acceptance.

## Reproduction and decision

`archives.json` hashes every retained compressed/decompressed input and audit.
Run the Python contract tests, then independently recompute:

```sh
python3 -m unittest discover -s scripts/tests -p 'test_quic*.py'
python3 scripts/quic_trace_audit.py moirai-quic/measurements/recovery/2026-10-14/140d60cf-{matrix,fairness}-*.json.gz --probe-executable quic_load --output /tmp/quic-trace-audit.json
python3 scripts/quic_recovery_audit.py --original-gate moirai-quic/measurements/recovery/2026-10-14/140d60cf-default-matrix.json.gz --output /tmp/quic-original-audit.json
python3 scripts/quic_trace_audit.py moirai-quic/measurements/recovery/2026-10-14/state/03dcc4a3-{matrix,fairness}-*.json.gz --probe-executable quic_load --output /tmp/quic-state-audit.json
python3 scripts/quic_recovery_audit.py --original-gate moirai-quic/measurements/recovery/2026-10-14/state/03dcc4a3-original-matrix.json.gz --output /tmp/quic-state-original-audit.json
```

To reproduce the supplementary local fault experiment, choose a fresh task-owned
output directory and build the observer as in the workflow. Preserve its baseline
binary before applying the candidate to that copied engine only. The manifest
records the baseline observation patch; the candidate is identified separately:

```sh
cargo fetch --locked
python3 scripts/quic_observation/prepare.py --output /tmp/moirai-quic-state-repro
CARGO_TARGET_DIR=/tmp/moirai-quic-state-repro/target cargo --config /tmp/moirai-quic-state-repro/config.toml build --manifest-path /tmp/moirai-quic-state-repro/workspace/Cargo.toml --locked --release -p moirai-quic --example quic_load
cp /tmp/moirai-quic-state-repro/target/release/examples/quic_load /tmp/moirai-quic-state-repro/baseline
python3 scripts/quic_observation/handshake.py --binary /tmp/moirai-quic-state-repro/baseline --engine-provenance /tmp/moirai-quic-state-repro/provenance.json --output /tmp/moirai-quic-state-repro/baseline-pairs.json
git -C /tmp/moirai-quic-state-repro/engine apply "$PWD/moirai-quic/measurements/recovery/2026-10-14/state/handshake-candidate.patch"
CARGO_TARGET_DIR=/tmp/moirai-quic-state-repro/target cargo --config /tmp/moirai-quic-state-repro/config.toml build --manifest-path /tmp/moirai-quic-state-repro/workspace/Cargo.toml --locked --release -p moirai-quic --example quic_load
python3 scripts/quic_observation/handshake.py --binary /tmp/moirai-quic-state-repro/target/release/examples/quic_load --engine-provenance /tmp/moirai-quic-state-repro/provenance.json --candidate-patch moirai-quic/measurements/recovery/2026-10-14/state/handshake-candidate.patch --output /tmp/moirai-quic-state-repro/candidate-pairs.json
```

The relay uses loopback and one explicit datagram fault. These commands never
configure the canonical provider or replace full shaped-network acceptance.
Keep command-recorder outputs separate from result JSON paths.

No default change is accepted. NewReno does not close the workload limit.
Experimental BBR completes it but retains mixed-policy fairness, queue and
window-growth concerns; no numerical regression tolerance is invented. Next,
correlate per-space handshake ACK/CRYPTO state with PTO selection, and obtain
established-session flight, flow-credit and pacing timelines before selecting
and measuring a focused recovery candidate. All original requirements remain
binding, and protected provider/Metis promotion remains blocked.

# Moirai gap audit

This register contains unresolved risks with no active implementation. A risk
moves to [backlog.md](backlog.md) when its re-open trigger fires.

<a id="MOI-GAP-STRUCTURE-001"></a>
## MOI-GAP-STRUCTURE-001 — Historical module-size findings are stale
- risk: The 2024 assessment reported 18 modules above 400 lines, but later splits make that count unfit for current planning.
- evidence: The retired historical board carried ISSUE-003 without a current-revision census or a DoR-scoped target.
- re-open trigger: A fresh structural scan finds a file above 500 lines containing more than one operation family.

<a id="ISSUE-226"></a>
## ISSUE-226 — Lazy priority-plane allocation remains unproved
- risk: Lazy construction could remove the last retained queue-plane allocation but requires safe late stealer publication.
- evidence: The delivered per-plane capacity policy reduced measured retained bytes from 1,572,864 to 540,672 at 24 workers; the remaining proposal lacks its required Loom model.
- re-open trigger: A measured consumer retention budget justifies the remaining saving and funds the stealer-publication model plus paired benchmark.

<a id="ISSUE-132"></a>
## ISSUE-132 — Rayon compatibility is intentionally bounded
- risk: A future consumer may require indexed cardinality, collect/unzip targets, interleave/step, logical-output blocks, numeric or fallible terminals, positions/windows, serial-inner flattening, mutation, intersperse, equality-checked zip, partition mapping, or sorting outside the audited subset.
- evidence: Existing transforms, reducers, slicing, sorting, mutation, and fallible terminals have value and comparison coverage; full Rayon API parity is not claimed.
- re-open trigger: A named consumer requires a missing surface that can retain a dedicated Moirai boundary and same-run comparison.

<a id="MOI-GAP-HTTP2-001"></a>
## MOI-GAP-HTTP2-001 — HTTP/2 has no current consumer
- risk: A future consumer may require multiplexed HTTP/2 behavior beyond the bounded HTTP/1.1 transport.
- evidence: The runtime roadmap makes HTTP/2 conditional while select ergonomics, IPv6, and graceful shutdown remain active work in `MOI-TOKIO-PARITY-STAGE-B1`.
- re-open trigger: A named consumer supplies an HTTP/2 acceptance workflow and resource-bound contract.

<a id="MOI-GAP-GPU-COSCHEDULING-001"></a>
## MOI-GAP-GPU-COSCHEDULING-001 — Device queue co-scheduling is provider-dependent
- risk: Moirai exposes accelerator route and occupancy metadata but does not own persistent device queues or hardware scheduling.
- evidence: ADR 0041 (revised by Moirai PR 501) keeps Hephaestus out of Moirai's dependency graph; stream/queue consumption and persistent kernels remain provider work.
- re-open trigger: Hephaestus exposes a stable queue/persistent-kernel contract and a consumer supplies hardware acceptance evidence.

<a id="MOI-AUDIT-PM-007"></a>
## MOI-AUDIT-PM-007 — PM fact ownership needs a structural audit
- risk: The earlier dead-checkout audit could not determine mechanically whether each live PM fact has one owner.
- evidence: The finding was held for manual review without an item scope or sampled duplication evidence.
- re-open trigger: A fresh board/ADR/doc scan identifies a duplicated live fact and names both authoritative candidates.

<a id="MOI-ADAPTIVE-THRESHOLD-PREMISE-2026-08-31"></a>
## MOI-ADAPTIVE-THRESHOLD-PREMISE-2026-08-31 — Adaptive dispatch lacks a body-cost model
- risk: One element-count threshold cannot select correctly across cheap and compute-heavy bodies; changing it can trade one measured regime for an unmeasured consumer regression.
- evidence: Retained measurements place the one-multiply crossover at 4,096–8,192 elements, `sqrt` plus `ln_1p` at 512–1,024, and chained fused operations below 512.
- re-open trigger: A consumer supplies a representative body-cost distribution or a body-cost-aware policy is specified without per-element dispatch.

<a id="MOI-GAP-ROADMAP-001"></a>
## MOI-GAP-ROADMAP-001 — Historical readiness goals lack acceptance contracts
- risk: API stabilization, production readiness, platform expansion, extended examples, tooling integration, and continuous performance work are aspirations rather than executable items.
- evidence: The retired checklists named these topics without scope, dependencies, value oracles, or concrete next steps.
- re-open trigger: A named consumer or measured defect supplies a DoR-complete acceptance workflow for one topic.

<a id="MOI-GAP-CONSUS-TRANSPORT-001"></a>
## MOI-GAP-CONSUS-TRANSPORT-001 — Object-storage adoption remains consumer-owned
- risk: Consus has not recorded a current comparative MinIO/toxiproxy result or the eventual default flip away from its legacy Tokio/rusoto path in this repository.
- evidence: Atlas ADR 0045 owns the cross-repository decision; Moirai already supplies the store-agnostic TLS/HTTP transport and must not own S3 protocol work.
- re-open trigger: The Consus owner requests a Moirai transport correction exposed by its paired benchmark or contract tests.

<a id="MOI-GAP-SLAB-GENERATION-32BIT-001"></a>
## MOI-GAP-SLAB-GENERATION-32BIT-001 — Slab free-list generation is 16 bits on 32-bit targets
- risk: On wasm32-with-atomics, armv7, and i686 the packed free-list head keeps a 16-bit generation, so 65,536 successful CASes inside one thread's load-to-CAS window can reinstall an occupied slot as the free-list head (ABA); 64-bit targets need 2^32.
- evidence: Read of `moirai-core/src/pool/slab.rs` packing and the `insert`/`remove` CAS sites; the interleaving is derived, not reproduced.
- re-open trigger: A 32-bit target becomes a supported deployment for `SlabAllocator`, or a pause-hook test on `i686-unknown-linux-gnu` reproduces the reinstall; the remedy is a `u32` index plus `u32` generation in an `AtomicU64`, as `LockFreeStack` does.

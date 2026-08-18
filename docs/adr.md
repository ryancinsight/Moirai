# Architecture Decision Record (ADR)

Moirai's decision records live one-per-file under [`docs/adr/`](adr/); see
[`docs/adr/README.md`](adr/README.md) for the generated index. This file is no
longer a record store — it retains only the ADR-015 pointer below, because
in-tree comments still cite that number.

## ADR-015: Native HTTP/S3 Transport Stack (Tokio-Free Object Storage)

**Relocated 2026-08-18 to the meta-repo: `docs/adr/0045-native-http-s3-transport-stack.md` (atlas ADR-0045).**

This decision is a cross-repo contract — moirai ships store-agnostic TLS/HTTP
transport (`moirai-tls`, `moirai-http`); consus builds the S3 protocol on top of
it. Governance places a cross-repo contract in the meta-repo, so the record
moved there and its status was corrected from Proposed to **Accepted**, which is
what the landed code already reflects. The meta-repo record carries a dated
revision note with the per-phase (P0-P5) delivery state as built.

This section is a pointer, not a second current record. Do not re-expand it;
edit atlas ADR-0045 instead. In-tree comments still citing "ADR-015" refer to
that record.

Implementation checklist: `docs/adr-015-checklist.md` (self-reported; its header
contradicts its own unchecked boxes - see the meta-repo record).

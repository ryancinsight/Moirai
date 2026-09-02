# ADR 0033: Provider-default MSRV alignment

Status: Accepted

## Context

Moirai follows the merged Mnemosyne provider default branch. Mnemosyne 0.5 and
Mnemosyne Core 0.2 declare Rust 1.95 after their Eunomia default-source
convergence. Moirai 0.3.1 still declared Rust 1.75 and constrained the older
Mnemosyne packages, which made its public compatibility metadata false.

## Decision

Moirai advances every workspace package from 0.3.1 to 0.4.0, raises the
workspace MSRV to Rust 1.95, and accepts Mnemosyne 0.5/Core 0.2 through their
default Git sources. The lockfile is the reproducibility pin; no revision
quarantine, local patch, or compatibility branch remains.

## Consequences

Consumers must update their Rust toolchain and any exact Moirai version
requirements. Scheduler algorithms and public API shapes do not change. The
release is breaking solely because the supported compiler range changes.

## Verification

Rust 1.95 compiles the focused GPU consumer and Rust 1.94 rejects the declared
contract. Warning-denied Clippy, configured Nextest, doctests, rustdoc, and the
provider duplicate scan remain required before merge.

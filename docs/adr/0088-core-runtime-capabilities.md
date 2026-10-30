# ADR 0088: Core runtime capabilities without the standard library

Status: Accepted

## Context

The core and utility packages declare a no_std mode, but core imports unavailable
standard-library services. Its alternate clock increments a counter, topology
returns a fixed processor count, and its task handle retains only an identifier.
These cannot establish elapsed time, hardware capability or completion. The
facade's no-std feature does not disable its native runtime dependencies.

The 0.12.0 release must either implement or remove these false capabilities.
Existing native consumers require the standard-library task and executor APIs;
they do not supply an operating-system provider for a bare target.

## Decision

Make the existing std feature the boundary for operations that require clocks,
threads, automatic hardware discovery and task completion. Compile the actual
standard-library implementations when enabled. Exclude those operation families
and their exports when disabled; delete fabricated alternate implementations.
Retain input-sensitive core/alloc vocabulary and algorithms that require none of
those capabilities. Allocation uses alloc/core rather than std re-exports.

Core's std feature forwards the utility std feature. Bare core must not pull it
in through an unconditional dependency, nor pull the native topology provider.
The explicit no-std marker features add no capability and are removed. Select
bare core and utilities through default-features = false. The facade remains a
native std runtime even with its optional defaults disabled.

Verify the actual x86_64-unknown-none target on pinned Rust 1.97.0, as well as
bare host builds. Run strict target Clippy, pure core/alloc tests and warning-free
Rustdoc. Inspect the resolved dependency graph for accidental std/native-provider
activation. Preserve default/std Clippy and behavior tests. These establish a
compiled no_std boundary; they do not establish a functioning bare-metal runtime,
a clock implementation, or execution on target hardware.

This public feature/API cleanup belongs to the existing 0.12.0 compatibility
boundary. The migration guide states the available subset and native-facade
requirement. [PR 650](https://github.com/ryancinsight/Moirai/pull/650) preserves
the change with the lifecycle release.

## Alternatives

Adding alloc imports alone leaves false time and completion behavior intact.
Adding a spin dependency does not supply clocks, threads or result publication.
Keeping an ignored no-std facade marker misrepresents its dependency graph.
Inventing a platform-provider trait without a present implementation expands the
contract without establishing a supported capability.

## Consequences

Bare consumers use only capabilities that the selected target can provide.
Native callers keep the real task-handle and executor surface through std.
Targets without the required allocator or atomic widths need their own explicit
support evidence; the checked target does not imply all embedded targets work.

## Overturning evidence

A real bare-target provider with measured clocks, topology and completion may
justify extending the corresponding role. A std dependency in the bare graph,
a fabricated result, or a changed native contract invalidates this boundary.

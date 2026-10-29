# Core no_std migration for 0.12

Use `default-features = false` on `moirai-core` or `moirai-utils` for their
core/alloc subset. The redundant `no-std` marker is removed. Bare core does not
provide OS threads, clocks, automatic topology discovery or task-result handles.
Those APIs require the existing `std` feature; a retained task identifier does
not substitute for completion.

`moirai-runtime` remains a native standard-library runtime when optional defaults
are disabled. Remove the facade's `no-std` marker from dependency declarations;
that marker never removed its standard-library dependencies. Bare applications
must use the supported core/utility subset rather than the facade.

The checked bare target is `x86_64-unknown-none` on Rust 1.97.0. This establishes
compilation and dependency separation, not target-hardware execution or support
for every embedded atomic width. [ADR 0088](../adr/0088-core-runtime-capabilities.md)
defines the boundary and evidence limits.

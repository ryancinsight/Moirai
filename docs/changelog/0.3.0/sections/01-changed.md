### Changed

- `moirai-core`: split the hybrid and MPMC channel implementations into focused
  endpoint and state modules. `HybridChannel<T>` is now a zero-sized endpoint
  factory; sender and receiver halves retain the only live shared state.
- Upgrade the Themis topology provider to 0.10.0 and consume its published
  optional cache-level contract without a workspace-local override.
- Upgrade Mnemosyne to the release that owns the same Themis topology identity.
- Pin Themis to the exact audited provider revision so downstream integrators
  resolve one source identity instead of compiling the same commit twice.
- Pin Mnemosyne to the revision that owns that same Themis identity.
- `moirai-executor`: migrated the Melinoe scheduler bridge to the validated
  `ParallelExecutor` capability. Its unsafe construction site documents the
  exact-once index, blocking-completion, and context-lifetime proof; raw executor
  function pointers no longer cross a safe registration boundary.
- The Mnemosyne facade contract now targets 0.3.0, removing the obsolete 0.2
  facade and its duplicate backend type identity from integrated consumers.


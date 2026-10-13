### Changed

- Release the workspace as 0.6.1 against mnemosyne-memory 0.8.0 and
  mnemosyne-memory-core 0.3.0. `moirai-executor` does not compile against
  the published 0.2 core (`periodic_defragmentation_sweep` cannot infer its
  pool type), so the workspace requirements move to the 0.8 and 0.3 lines. No
  `moirai-core` or `moirai-pal` public item changes (cargo-semver-checks
  against 0.6.0 reports no required update).


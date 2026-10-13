## [0.5.0] - 2026-08-11

### Changed

- Re-release the workspace as version 0.5.0 against mnemosyne-memory 0.7.0 so
  consumers use the same Mnemosyne/Eunomia provider graph as the rest of
  Atlas. The package identity and Rust library name remain unchanged.

### Breaking

- The optional Mnemosyne integration now requires mnemosyne-memory 0.7.0;
  consumers must update their dependency graph to the 0.7 provider line.


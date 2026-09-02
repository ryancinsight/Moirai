# ADR 0002: WASM-First Async Architecture

Status: Accepted

**Date**: 2024-12-19
**Context**: WebAssembly Support Strategy

### Decision

Moirai's async architecture shall be designed with WebAssembly as a first-class target, ensuring full functionality in WASM environments without platform-specific dependencies.

### Context

Current async runtimes have limited or no WebAssembly support:
- **Tokio**: Minimal WASM compatibility, lacks I/O reactor
- **Rayon**: No WASM support for parallel execution
- **Platform Dependencies**: Most runtimes require OS-specific thread management

### Implementation Strategy

**WASM Compatibility Requirements**:
- No dependency on OS threads (use web workers or cooperative scheduling)
- Platform-agnostic I/O abstraction layer
- JavaScript interop for browser environments
- Node.js compatibility for server-side WASM

**Architecture Patterns**:
- Pluggable executor backends (native threads vs web workers)
- Async I/O via platform abstraction layer (PAL)
- Timer implementation using platform-appropriate mechanisms

### Consequences

**Positive**: Universal deployment, browser compatibility, server-side WASM support
**Negative**: Additional abstraction complexity, platform-specific testing requirements

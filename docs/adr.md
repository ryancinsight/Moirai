# Architecture Decision Record (ADR)

## ADR-001: Moirai as Complete Alternative to Tokio/Rayon

**Date**: 2024-12-19  
**Status**: Accepted  
**Context**: Moirai Concurrency Library Architecture  

### Decision

Moirai shall be implemented as a complete, standalone alternative to existing concurrency libraries (tokio, rayon, openmp, tbb) with native WebAssembly support, without runtime dependencies on the libraries it aims to replace.

### Context

The Rust ecosystem currently requires developers to combine multiple libraries for comprehensive concurrency:
- **Tokio**: Async runtime for I/O-bound operations
- **Rayon**: Data parallelism for CPU-bound operations  
- **OpenMP/TBB**: Traditional parallel computing patterns
- **Separate WASM solutions**: Limited cross-platform async support

This fragmentation leads to:
- Complex integration patterns
- Performance overhead from library boundaries
- Inconsistent APIs across paradigms
- Limited WebAssembly compatibility

### Decision Rationale

**Core Principle**: Moirai provides unified concurrency primitives that eliminate the need for external runtime dependencies, particularly tokio and rayon.

**Implementation Strategy**:
1. **Native Async Runtime**: Custom executor without tokio dependencies
2. **Unified API**: Single interface for async, parallel, and hybrid execution
3. **Zero-Cost Abstractions**: Compile-time optimizations that match or exceed alternatives
4. **WebAssembly First**: Native WASM support without platform-specific limitations

### Implementation Details

#### Allowed Tokio/Rayon Usage
- **Benchmarks Only**: Performance comparison testing (`benchmarks/` directory)
- **Examples Only**: Comparison demonstrations (`examples/moirai_vs_tokio_rayon_comparison.rs`)
- **Development Dependencies**: Testing infrastructure only

#### Prohibited Tokio/Rayon Usage
- **Runtime Dependencies**: No tokio/rayon in core library Cargo.toml dependencies
- **Implementation Dependencies**: Core modules must not import tokio/rayon for functionality
- **API Exposure**: Public APIs must not expose tokio/rayon types

#### Alternative Implementations Required
- **File I/O**: Native async file operations using platform syscalls
- **Network I/O**: Direct socket programming with epoll/kqueue/iocp integration  
- **Timer Systems**: Custom timer wheels and deadline management
- **Task Scheduling**: Work-stealing schedulers with custom executors

### Consequences

**Positive**:
- **Zero External Runtime Dependencies**: Simplified deployment and reduced attack surface
- **Unified Programming Model**: Single API for all concurrency patterns
- **WebAssembly Compatibility**: Full async support in WASM environments
- **Performance Control**: Direct optimization without library boundary overhead
- **Predictable Behavior**: No hidden runtime complexity or thread pool conflicts

**Negative**:
- **Development Complexity**: Requires implementing low-level async primitives
- **Platform Abstraction**: Must handle OS-specific async I/O mechanisms
- **Maintenance Burden**: Responsible for async runtime quality and performance
- **Ecosystem Integration**: May require adapters for tokio-based libraries

### Compliance Verification

**Build-Time Checks**:
```bash
# Verify no tokio runtime dependencies
cargo tree | grep -v "benchmarks\|examples" | grep tokio && echo "VIOLATION"

# Verify core modules clean
find moirai-* -name "Cargo.toml" -exec grep -l "tokio" {} \; | grep -v benchmarks
```

**Code Review Requirements**:
- All `use tokio::` imports must be in benchmarks or examples
- All `#[tokio::test]` must be replaced with native test harness
- All async function implementations must use Moirai primitives

### Related Decisions

- **ADR-002**: WASM-First Async Architecture
- **ADR-003**: Zero-Copy Communication Primitives  
- **ADR-004**: Hybrid Execution Model Design

### Implementation Status

- [x] **Phase 1**: Remove tokio dependencies from `moirai-async`
- [x] **Phase 2**: Implement native file I/O operations
- [x] **Phase 3**: Implement native network I/O operations
- [x] **Phase 4**: Replace tokio test infrastructure
- [ ] **Phase 5**: Validate performance parity with benchmarks

---

## ADR-002: WASM-First Async Architecture

**Date**: 2024-12-19  
**Status**: Accepted  
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

---

## ADR-003: Zero-Copy Communication Primitives

**Date**: 2024-12-19  
**Status**: Accepted

### Decision

All inter-task communication shall prioritize zero-copy operations through shared memory, memory-mapped regions, and ownership transfer rather than serialization.

### Implementation

- Lock-free queues with ownership transfer
- Memory-mapped channels for large data
- Copy-on-write semantics for shared state
- NUMA-aware memory allocation

---

## ADR-004: Hybrid Execution Model

**Date**: 2024-12-19  
**Status**: Accepted

### Decision

Moirai provides a unified API that automatically selects optimal execution strategy (async, parallel, or hybrid) based on workload characteristics and system resources.

### Implementation

- Adaptive task scheduler with workload detection
- Automatic async/parallel task routing
- Resource-aware execution planning
- Single API surface for all execution patterns

---

This ADR establishes the foundational architectural principles that guide all implementation decisions in the Moirai concurrency library, ensuring consistency with the project's vision of being a complete alternative to existing fragmented concurrency solutions.
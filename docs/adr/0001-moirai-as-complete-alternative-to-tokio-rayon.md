# ADR-001: Moirai as Complete Alternative to Tokio/Rayon

Status: Accepted

**Date**: 2024-12-19
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
- **Examples Only**: Comparison demonstrations (`moirai/examples/moirai_vs_tokio_rayon_comparison.rs`)
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

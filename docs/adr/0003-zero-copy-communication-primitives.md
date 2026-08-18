# ADR-003: Zero-Copy Communication Primitives

Status: Accepted

**Date**: 2024-12-19

### Decision

All inter-task communication shall prioritize zero-copy operations through shared memory, memory-mapped regions, and ownership transfer rather than serialization.

### Implementation

- Lock-free queues with ownership transfer
- Memory-mapped channels for large data
- Copy-on-write semantics for shared state
- NUMA-aware memory allocation

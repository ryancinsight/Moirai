# ADR 0004: Hybrid Execution Model

Status: Accepted

**Date**: 2024-12-19

### Decision

Moirai provides a unified API that automatically selects optimal execution strategy (async, parallel, or hybrid) based on workload characteristics and system resources.

### Implementation

- Adaptive task scheduler with workload detection
- Automatic async/parallel task routing
- Resource-aware execution planning
- Single API surface for all execution patterns

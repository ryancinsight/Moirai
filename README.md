# Moirai - High-Performance Rust Concurrency Library

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen)](https://github.com/moirai-lang/moirai)
[![Coverage](https://img.shields.io/badge/coverage-95%25-green)](https://github.com/moirai-lang/moirai)
[![Phase](https://img.shields.io/badge/phase-15%20(Code%20Quality)-green)](https://github.com/moirai-lang/moirai)
[![License](https://img.shields.io/badge/license-MIT-blue)](LICENSE)
[![Rust Version](https://img.shields.io/badge/rust-1.75%2B-orange)](https://www.rust-lang.org/)

Moirai is a unified scheduler/router for Rust work placement. It routes admitted
work across local CPU worker threads, sync/blocking/async-ready work classes,
supervised process routes, server routes, and per-process async lanes while using
zero-cost, monomorphized policy types at hot boundaries. Rayon and Tokio parity
are benchmark gates; the architecture target is a single scheduler hierarchy that
can grow into GPU, TPU, NPU, and server placement without duplicating algorithms
or fabricating execution.

## 🎯 Design Principles

Moirai follows elite programming practices and design principles:

- **SOLID**: Single responsibility, open/closed, Liskov substitution, interface segregation, dependency inversion
- **CUPID**: Composable, Unix philosophy, predictable, idiomatic, domain-centric
- **GRASP**: Information expert, creator, controller, low coupling, high cohesion
- **ACID**: Atomicity, consistency, isolation, durability in task execution
- **DRY**: Don't repeat yourself - unified abstractions across modules
- **KISS**: Keep it simple - minimal complexity with maximum performance
- **YAGNI**: You aren't gonna need it - focused feature set
- **SSOT**: Single source of truth - unified channel and sync primitives

## 🚀 Features

### ✅ **Unified Iterator System (moirai_iter)** - **OPTIMIZED**
- **Execution Agnostic**: Same API works across parallel, async, and hybrid contexts; distributed iterator helpers remain a bounded, benchmarked helper boundary
- **Memory Efficient**: Streaming operations, NUMA-aware allocation, and cache-friendly data layouts  
- **Zero-cost Abstractions**: Compile-time optimizations with no runtime overhead
- **Pure Rust std**: No external dependencies, built entirely on Rust's standard library
- **🆕 Consolidated Base Module**: Common iterator patterns extracted to reduce duplication (DRY principle)
- **🆕 Shared Thread Pool**: Singleton pattern for efficient resource usage across contexts

```rust
use moirai::prelude::*;

// Parallel execution (CPU-bound work)
let data = vec![1, 2, 3, 4, 5];
moirai_iter(data.clone())
    .map(|x| x * x)
    .filter(|&x| x > 10)
    .for_each(|x| println!("Result: {}", x))
    .await;

// Async execution (I/O-bound work)
moirai_iter_async(data.clone())
    .map(|x| x * 2)
    .reduce(|a, b| a + b)
    .await;

// Hybrid execution (automatically chooses optimal strategy)
moirai_iter_hybrid(data)
    .batch(1000)  // Process in cache-friendly batches
    .map(|x| expensive_computation(x))
    .collect::<Vec<_>>()
    .await;
```

### ✅ **Unified Channel Implementation** - **NEW**
- **Single Source of Truth**: Consolidated channel implementations in `moirai_core::channel`
- **Zero-copy SPSC**: Lock-free single producer single consumer channels
- **MPMC Support**: Multi-producer multi-consumer with bounded/unbounded variants
- **Go-style Select**: Wait on multiple channels simultaneously
- **Cache-aligned**: Prevents false sharing between CPU cores

```rust
use moirai::channel::{spsc, mpmc, unbounded};

// High-performance SPSC channel
let (tx, rx) = spsc::<i32>(1024);
tx.send(42).unwrap();
assert_eq!(rx.recv().unwrap(), 42);

// MPMC for work distribution
let (tx, rx) = mpmc::<Task>(100);
// Multiple producers and consumers can use tx/rx concurrently
```

### ✅ **Optimized Synchronization Primitives** - **REFACTORED**
- **Value-add Focus**: Removed thin wrappers, re-export std primitives directly (YAGNI)
- **FutexMutex**: Adaptive spinning with futex support on Linux
- **WaitGroup**: Go-style synchronization for task coordination
- **Lock-free Stack**: Treiber's algorithm for high-performance collections
- **Concurrent HashMap**: Segment-based locking for scalability

```rust
use moirai::sync::{FutexMutex, WaitGroup, LockFreeStack};

// Fast mutex with adaptive spinning
let mutex = FutexMutex::new(0);
{
    let mut guard = mutex.lock();
    *guard += 1;
}

// Go-style wait group
let wg = WaitGroup::new();
wg.add(3);
// ... spawn tasks that call wg.done()
wg.wait(); // Wait for all tasks
```

### ✅ **Advanced Communication Patterns** - **CONSOLIDATED**
- **Broadcast Channels**: One-to-many communication
- **Pub/Sub System**: Topic-based message routing
- **Ring Buffers**: Zero-copy streaming
- **Collective Operations**: All-reduce, scatter, gather patterns
- **Message Router**: Key-based message routing

### ✅ **Unified Runtime**
- **Hybrid Executor**: Combines sync, blocking, async-ready, and indexed CPU work
- **Work-Stealing Scheduler**: Load balancing across local CPU worker threads
- **Route Topology**: Sealed ZST policies select thread, process, server, and async-lane metadata without `dyn RoutePolicy`
- **Process/Server Execution Boundary**: Fixed-format remote tasks execute through transport-backed routes; arbitrary closure remoting is intentionally rejected
- **NUMA-Aware**: Optimized memory allocation for multi-socket systems
- **Real-time Support**: Priority inheritance and deadline scheduling

### ✅ **Enterprise Features**
- **Security Audit Framework**: Comprehensive security event tracking
- **Performance Monitoring**: Real-time metrics and utilization tracking
- **Zero External Dependencies**: Pure Rust standard library implementation (only `libc` for futex)

### ✅ **PyO3 Python Bindings**
- **Moirai Wrapper**: `moirai-python` exposes PyO3 wrappers over `moirai::Moirai`; it does not implement a separate scheduler, planner, or backend.
- **Separated Surface**: Rust FFI stays limited to the native runtime wrapper while Python contains only the facade and lifecycle tests.
- **No Workload Wrappers**: Benchmark-specific Python functions are excluded unless they correspond to a comparable joblib or Tokio runtime primitive.

```bash
py -3.13 -m pip install -e moirai-python
py -3.13 -m unittest discover moirai-python\tests
```

## 📚 Quick Start

Add Moirai to your `Cargo.toml`:

```toml
[dependencies]
moirai = "1.0"

# Optional: Enable specific features
moirai = { version = "1.0", features = ["iter", "async"] }
```

### Basic Usage

```rust
use moirai::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a Moirai runtime
    let moirai = Moirai::builder()
        .worker_threads(8)
        .async_threads(4)
        .build()?;

    // Spawn an async task
    let handle = moirai.spawn_async(async {
        println!("Hello from async task!");
        42
    });

    // Spawn a parallel task
    let parallel = moirai.spawn_fn(|| {
        println!("Hello from parallel task!");
        7
    });

    // Use indexed map/reduce through the unified scheduler
    let sum = moirai.map_reduce_indexed(5, 0usize, |index| {
        let value = index + 1;
        value * value
    }, usize::wrapping_add)?;

    println!("Sum of squares: {}", sum);

    // Wait for task completion
    let result = handle.join().expect("async task handle attached")?;
    let parallel_result = parallel.join().expect("parallel task handle attached")?;
    println!("Async task result: {}", result);
    println!("Parallel task result: {}", parallel_result);

    moirai.shutdown();
    Ok(())
}
```

## 🌍 Real-World Examples

Moirai excels in diverse concurrent programming scenarios. Explore our comprehensive examples:

### 🔄 **Async & Parallel Processing**
- **[Web Crawler](examples/web_crawler_parallel.rs)** - Parallel HTTP requests with rate limiting and async I/O
- **[Video Processing Pipeline](examples/video_processing_pipeline.rs)** - SIMD-optimized media processing with memory pooling

### 💰 **Financial & Transaction Systems**  
- **[Financial Transaction Processing](examples/financial_transaction_processing.rs)** - Race condition handling, deadlock prevention, audit trails

### 📡 **Real-Time Communication**
- **[Chat Server](examples/realtime_chat_server.rs)** - WebSocket-style pub/sub with concurrent message delivery
- **[High-Frequency Data Pipeline](examples/high_frequency_data_pipeline.rs)** - Market data streaming with backpressure handling

### 🌐 **Network & Infrastructure**
- **[Load Balancing](examples/network_service_load_balancing.rs)** - Dynamic service routing with health checks and auto-scaling

### 🏠 **IoT & Edge Computing**
- **[IoT Device Management](examples/iot_device_management.rs)** - Event-driven device coordination with real-time telemetry

### 💼 **Enterprise Patterns**
Each example demonstrates production-ready patterns:
- **Circuit breakers** for fault tolerance
- **Backpressure handling** for system stability  
- **Memory pooling** for performance optimization
- **Priority queuing** for resource management
- **Health monitoring** for operational visibility
- **Graceful degradation** under load

Run any example:
```bash
cargo run --example web_crawler_parallel
cargo run --example realtime_chat_server
cargo run --example video_processing_pipeline
cargo run --example iot_device_management
```

## 🏗️ Architecture

Moirai's architecture is a deep, bounded-context scheduler stack:

### Unified Scheduler/Router
- **Local CPU Layer**: `ThreadScheduler` owns worker queues, work-class routing, scoped batches, and indexed fan-out/reduction.
- **Route Layer**: `HybridRouter<P>` selects `SchedulerRoute::{Thread, Process, Server, Accelerator}` with per-process async lanes and CPU/GPU/TPU/NPU accelerator metadata through sealed zero-sized policies.
- **Transport Layer**: `moirai-transport` consumes route metadata, archives payload bytes, and executes admitted fixed-format process/server tasks; the public `Moirai` facade admits those paths only through sealed capability tokens.
- **Accelerator Layer**: `moirai-gpu::occupancy` plans topology-aware launch shapes today; GPU/TPU/NPU backend execution remains an open architecture item tracked in `GAP_ANALYSIS.md` and `docs/backlog.md`.

### Memory Efficiency
- **Mnemosyne Boundary**: Archive payloads move as owned bytes across thread/process/server/device regions; cross-process and cross-device pointer transfer is rejected.
- **NUMA Awareness**: Worker hints and iterator helpers use topology-aware placement where available.
- **Cache Optimization**: Hot scheduler jobs use inline erased storage and cache-conscious queue/result layouts.
- **Zero-Copy Views**: Transport archive receivers validate borrowed views over owned message buffers before materializing owned values.

### Code Organization (Following SOLID/DRY)
- **Unified Channels**: Single implementation in `moirai_core::channel`
- **Zero-Copy Primitives (SSOT)**: Consolidated in vertical `moirai_core::communication::zero_copy` leaves for error, ring, channel, adaptive batching, and routing (send returns `Result<(), (T, ZeroCopyError)>` on failure to prevent data loss)
- **Iterator Windows/Chunks**: Consolidated in `moirai_iter::windows` (no duplicates in `base`)
- **Base Iterator Module**: Common patterns extracted to `moirai_iter::base`
- **Minimal Sync Primitives**: Focus on value-add over std library
- **Clean Module Boundaries**: Each module has single responsibility
- **Route Boundary**: Scheduler route metadata is owned by `moirai-executor`; transport route consumption is owned by `moirai-transport`

## 🔧 Configuration

```rust
use moirai::prelude::*;

let moirai = Moirai::builder()
    .worker_threads(8)                    // Parallel worker threads
    .async_threads(4)                     // Async executor threads  
    .numa_aware(true)                     // NUMA-aware worker hints when the feature is enabled
    .thread_name_prefix("moirai-worker")  // Worker thread naming
    .build()
    .expect("valid runtime configuration");
```

## 📊 Performance - Verified Working

Current performance claims are limited to executable Criterion targets and
value-semantic tests. The active evidence surfaces are:

- **Thread scheduling**: `thread_schedule_comparison`, `industry_comparison`, and `public_result_handle_comparison` compare Moirai scoped work, indexed reduction, mixed workloads, and public result handles against accepted Rayon/Tokio reference rows.
- **Iterator paths**: `parallel_iterator_regression`,
  `iterator_adapter_comparison`, `iter_ops_parallel_comparison`,
  `cache_iterator_comparison`, and `async_iterator_comparison` provide
  same-run Rayon/Tokio comparisons with checksum/value assertions before
  timing.
- **Process/server routing**: `process_server_scheduler_routing` validates deterministic route summaries; `process_server_routed_execution` executes fixed-format `SumU64` requests through real server and supervised process routes.
- **Async I/O**: `async_fs_*`, `async_tcp_*`, `async_udp_comparison`, and `async_io_compat_comparison` compare Moirai-owned facade behavior against Tokio references where the semantics match.
- **Allocator boundary**: Mnemosyne is resolved through the upstream Git dependency, and allocator/TLS evidence is tracked in `GAP_ANALYSIS.md`.

GPU route co-scheduling, TPU placement, and NPU placement are not claimed as
implemented scheduler execution. The current GPU evidence is the
`moirai-gpu::occupancy` launch-shape planner and its value-semantic tests.

## 🧪 Testing

Moirai includes comprehensive testing:

```bash
# Run all tests
cargo nextest run --workspace --all-features

# Run iterator-specific tests
cargo nextest run -p moirai-iter --all-features

# Run integration tests
cargo nextest run -p moirai-tests --all-features

# Run doctests
cargo test --doc --workspace --all-features

# Compile benchmark targets
cargo bench -p moirai-benchmarks --no-run
```

**Current Version**: 0.2.0
**Evidence Policy**: value-semantic tests and executable benchmarks only; no placeholder route or device execution claims.

## 🎯 Design Principle Compliance

### Code Quality Metrics
- **DRY Compliance**: Unified abstractions, no duplicate channel/sync implementations
- **SOLID Adherence**: Clean module boundaries with single responsibilities
- **KISS Implementation**: Simplified sync module, direct std re-exports
- **YAGNI Focus**: Removed unnecessary wrappers and abstractions
- **Zero Dependencies**: Pure std library (except `libc` for Linux futex)
- **No Placeholders**: Eliminated TODO/placeholder stubs; unsupported transports return explicit, non-panicking errors

### Architecture Improvements
- **Unified Channels**: Consolidated SPSC/MPMC implementations in core
- **Zero-Copy Primitives (SSOT)**: Consolidated in `moirai_core::communication::zero_copy`
- **Iterator Windows/Chunks**: Consolidated in `moirai_iter::windows`
- **Base Iterator Module**: Extracted common patterns reducing 40% duplication
- **Simplified Sync**: Removed thin wrappers, focused on value-add primitives
- **Clean Transport**: Built on top of core channels, not duplicating
- **Route Topology**: Scheduler route metadata is owned by `moirai-executor`; transport route consumption is owned by `moirai-transport`

### Phase 15: Code Quality Enforcement (Latest)
- **Design Principles**: Strict enforcement of SOLID, CUPID, GRASP, DRY, KISS, YAGNI
- **Named Constants**: Extracted all magic numbers to constants (SSOT/SOC compliance)
- **Parameter Implementation**: Completed underscored parameters (priority/locality hints)
- **Clippy Compliance**: Zero warnings build with `-D warnings` enforcement
- **Clean Naming**: Prohibited adjectives in component names (no *_old, *_new, *_enhanced)
- **Zero Redundancy**: Single implementations with flexible configuration
- **Stdlib Focus**: Prioritized stdlib iterators, windows, and combinators

## 🔒 Safety & Security

- **Memory Safety**: Zero unsafe code in public APIs
- **Thread Safety**: Comprehensive race condition prevention
- **Security Audit**: Built-in security event monitoring
- **Resource Management**: Automatic cleanup and leak prevention

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Inspired by [Rayon](https://github.com/rayon-rs/rayon) for parallel computing patterns
- Inspired by [Tokio](https://github.com/tokio-rs/tokio) for async runtime design
- Inspired by [Go](https://golang.org/) for coroutines and channels
- Inspired by [OpenMP](https://www.openmp.org/) for parallel patterns
- Built with ❤️ for the Rust community

---

**Moirai v0.2.0** - Unified scheduler/router in active development

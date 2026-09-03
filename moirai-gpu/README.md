# moirai-gpu

[![crates.io](https://img.shields.io/crates/v/moirai-gpu.svg)](https://crates.io/crates/moirai-gpu)
[![docs.rs](https://docs.rs/moirai-gpu/badge.svg)](https://docs.rs/moirai-gpu)

GPU integration for the [Moirai](https://github.com/ryancinsight/Moirai)
concurrency library.

Two layers, with different maturity:

- **Launch planning** (`occupancy`, always available, no backend required):
  `plan_launch`, `resident_blocks`, and `plan_persistent_launch` turn a
  mnemosyne `KernelResourceBudget` and a themis `GpuTopology` into a
  `LaunchShape`. The planner never reshapes a kernel's declared block size,
  because register and shared-memory budgets are stated per that width.
- **Compute execution** (`wgpu-backend` feature, on by default): `GpuDevice`,
  `GpuBuffer` and buffer pooling, `ComputeShader` / `ComputeKernel`,
  `ComputePipeline`, and `GpuTask` over wgpu. Disable default features to keep
  only the planner and drop the wgpu dependency.

GPU route co-scheduling through the Moirai scheduler is not implemented; the
current scheduler-level evidence is the launch-shape planner and its
value-semantic tests.

```toml
[dependencies]
moirai-gpu = "0.5"
```

```rust
use moirai_gpu::{KernelResourceBudget, LaunchShape, plan_launch};

// 64 registers/thread, 16 KiB shared memory, 256 threads per block.
let budget = KernelResourceBudget::new(64, 16 * 1024, 256).unwrap();

// Cover 1000 work items with one thread each.
let shape: LaunchShape = plan_launch(budget, 1000);
assert_eq!(shape.threads_per_block, 256);
assert_eq!(shape.grid_blocks, 4); // ceil(1000 / 256)
```

Full documentation: <https://docs.rs/moirai-gpu>

## License

Licensed under either of [Apache-2.0](../LICENSE-APACHE) or
[MIT](../LICENSE-MIT) at your option.

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use moirai_gpu::{plan_launch, KernelResourceBudget};
use std::hint::black_box;

const WORKGROUP_WIDTH: u32 = 256;

fn gpu_launch_planning_benchmark(c: &mut Criterion) {
    let budget = KernelResourceBudget::new(0, 0, WORKGROUP_WIDTH)
        .expect("invariant: benchmark workgroup width is non-zero");
    let mut group = c.benchmark_group("gpu_launch_planning");

    for work_items in [1_u64, 1_024, 65_536, 1_048_576] {
        group.bench_with_input(
            BenchmarkId::from_parameter(work_items),
            &work_items,
            |benchmark, &work_items| {
                benchmark.iter(|| black_box(plan_launch(budget, black_box(work_items))));
            },
        );
    }

    group.finish();
}

criterion_group!(benches, gpu_launch_planning_benchmark);
criterion_main!(benches);

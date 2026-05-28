//! Owned distributed iterator comparison against Rayon.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use moirai_iter::distributed::{
    DistributedContext, GpuConfig, LatencyProfile, NodeCapability, NodeConfig,
};
use rayon::prelude::*;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Duration;
use tokio::runtime::{Builder, Runtime};

const SAMPLE_SIZE: usize = 10;
const WARM_UP_MILLIS: u64 = 100;
const MEASUREMENT_MILLIS: u64 = 300;
const WORK_ITEMS: usize = 512;

fn source_data() -> Vec<u64> {
    (0..WORK_ITEMS as u64)
        .map(|value| value.wrapping_mul(19).wrapping_add(7))
        .collect()
}

fn node_config(port: u16, cpu_cores: usize) -> NodeConfig {
    NodeConfig {
        address: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), port),
        cpu_cores,
        memory_gb: 8,
        gpu_config: None::<GpuConfig>,
        latency_profile: LatencyProfile {
            average_latency_ms: 1.0,
            bandwidth_mbps: 1000.0,
            reliability_score: 0.99,
        },
        capabilities: vec![NodeCapability::HighCompute],
    }
}

fn distributed_context() -> DistributedContext {
    let mut context = DistributedContext::new();
    context.add_node(node_config(9101, 1));
    context.add_node(node_config(9102, 3));
    context
}

fn moirai_distributed_context_map(
    runtime: &Runtime,
    context: &DistributedContext,
    data: Vec<u64>,
) -> u64 {
    runtime
        .block_on(async {
            context
                .execute_distributed_map(data, |value| value.wrapping_mul(7).wrapping_add(5))
                .await
        })
        .expect("distributed context map must complete")
        .into_iter()
        .sum()
}

fn rayon_owned_map(data: Vec<u64>) -> u64 {
    data.into_par_iter()
        .map(|value| value.wrapping_mul(7).wrapping_add(5))
        .sum()
}

fn distributed_context_comparison(c: &mut Criterion) {
    let data = source_data();
    let context = distributed_context();
    let runtime = Builder::new_current_thread()
        .build()
        .expect("tokio runtime must build for distributed benchmark");

    assert_eq!(
        moirai_distributed_context_map(&runtime, &context, data.clone()),
        rayon_owned_map(data.clone())
    );

    let mut group = c.benchmark_group("distributed_context_owned_map");
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_millis(WARM_UP_MILLIS));
    group.measurement_time(Duration::from_millis(MEASUREMENT_MILLIS));
    group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter(|| {
            black_box(moirai_distributed_context_map(
                &runtime,
                &context,
                black_box(input.clone()),
            ))
        })
    });
    group.bench_with_input(BenchmarkId::new("rayon", WORK_ITEMS), &data, |b, input| {
        b.iter(|| black_box(rayon_owned_map(black_box(input.clone()))))
    });
    group.finish();
}

criterion_group!(benches, distributed_context_comparison);
criterion_main!(benches);

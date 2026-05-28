//! Owned multi-system iterator comparison against Rayon.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use moirai_iter::distributed::{
    GpuConfig, LatencyProfile as NodeLatencyProfile, NodeCapability, NodeConfig,
};
use moirai_iter::multi_system::{
    BandwidthProfile, CpuClusterConfig, GpuClusterConfig, GpuFramework, GpuInterconnect,
    InterconnectConfig, LatencyProfile as SystemLatencyProfile, MemoryHierarchy,
    MultiSystemContext, NetworkTopology, NumaTopology, StorageTier, SystemConfig,
    WorkloadSpecialization,
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
        .map(|value| value.wrapping_mul(23).wrapping_add(17))
        .collect()
}

fn system_config(port: u16, cpu_cores: usize) -> SystemConfig {
    SystemConfig {
        node: NodeConfig {
            address: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), port),
            cpu_cores,
            memory_gb: 16,
            gpu_config: None::<GpuConfig>,
            latency_profile: NodeLatencyProfile {
                average_latency_ms: 1.0,
                bandwidth_mbps: 1000.0,
                reliability_score: 0.99,
            },
            capabilities: vec![NodeCapability::HighCompute],
        },
        gpu_cluster: Some(GpuClusterConfig {
            node_count: 1,
            gpus_per_node: 1,
            total_gpu_memory_gb: 16,
            gpu_interconnect: GpuInterconnect::PCIe,
            frameworks: vec![GpuFramework::WGPU],
        }),
        cpu_cluster: CpuClusterConfig {
            total_cores: cpu_cores,
            numa_topology: NumaTopology {
                numa_nodes: 1,
                cores_per_numa_node: cpu_cores,
                memory_per_numa_node_gb: 16,
                interconnect_bandwidth_gbps: 64.0,
            },
            memory_hierarchy: MemoryHierarchy {
                l1_cache_kb: 32,
                l2_cache_kb: 512,
                l3_cache_kb: 32768,
                memory_bandwidth_gbps: 128.0,
                storage_tier: StorageTier {
                    nvme_capacity_gb: 1024,
                    ssd_capacity_gb: 2048,
                    hdd_capacity_gb: 0,
                    network_storage_gb: 0,
                },
            },
        },
        interconnect: InterconnectConfig {
            topology: NetworkTopology::Mesh,
            bandwidth_profile: BandwidthProfile {
                peak_bandwidth_gbps: 100.0,
                sustained_bandwidth_gbps: 80.0,
                burst_duration_ms: 10.0,
            },
            latency_profile: SystemLatencyProfile {
                min_latency_us: 1.0,
                avg_latency_us: 2.0,
                max_latency_us: 5.0,
                jitter_us: 0.1,
            },
        },
        specializations: vec![WorkloadSpecialization::DataAnalytics],
    }
}

fn multi_system_context() -> MultiSystemContext {
    let mut context = MultiSystemContext::new();
    context.add_system(system_config(9201, 2));
    context.add_system(system_config(9202, 2));
    context
}

fn moirai_multi_system_context_map(
    runtime: &Runtime,
    context: &MultiSystemContext,
    data: Vec<u64>,
) -> u64 {
    runtime
        .block_on(async {
            context
                .execute_heterogeneous_compute(
                    data,
                    |value| value.wrapping_mul(11).wrapping_add(13),
                    |value| value.wrapping_mul(11).wrapping_add(13),
                )
                .await
        })
        .expect("multi-system context map must complete")
        .into_iter()
        .sum()
}

fn rayon_owned_map(data: Vec<u64>) -> u64 {
    data.into_par_iter()
        .map(|value| value.wrapping_mul(11).wrapping_add(13))
        .sum()
}

fn multi_system_context_comparison(c: &mut Criterion) {
    let data = source_data();
    let context = multi_system_context();
    let runtime = Builder::new_current_thread()
        .build()
        .expect("tokio runtime must build for multi-system benchmark");

    assert_eq!(
        moirai_multi_system_context_map(&runtime, &context, data.clone()),
        rayon_owned_map(data.clone())
    );

    let mut group = c.benchmark_group("multi_system_context_owned_map");
    group.sample_size(SAMPLE_SIZE);
    group.warm_up_time(Duration::from_millis(WARM_UP_MILLIS));
    group.measurement_time(Duration::from_millis(MEASUREMENT_MILLIS));
    group.bench_with_input(BenchmarkId::new("moirai", WORK_ITEMS), &data, |b, input| {
        b.iter(|| {
            black_box(moirai_multi_system_context_map(
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

criterion_group!(benches, multi_system_context_comparison);
criterion_main!(benches);

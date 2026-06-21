use super::*;
use crate::distributed::{NodeCapability, NodeConfig};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};

struct NonClone(u64);

fn test_system(port: u16) -> SystemConfig {
    SystemConfig {
        node: NodeConfig {
            address: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), port),
            cpu_cores: 64,
            memory_gb: 256,
            gpu_config: None,
            latency_profile: crate::distributed::LatencyProfile {
                average_latency_ms: 0.1,
                bandwidth_mbps: 10000.0,
                reliability_score: 0.999,
            },
            capabilities: vec![NodeCapability::HighCompute, NodeCapability::HighMemory],
        },
        gpu_cluster: Some(GpuClusterConfig {
            node_count: 4,
            gpus_per_node: 8,
            total_gpu_memory_gb: 256,
            gpu_interconnect: GpuInterconnect::NVLink,
            frameworks: vec![GpuFramework::CUDA, GpuFramework::WGPU],
        }),
        cpu_cluster: CpuClusterConfig {
            total_cores: 256,
            numa_topology: NumaTopology {
                numa_nodes: 8,
                cores_per_numa_node: 32,
                memory_per_numa_node_gb: 32,
                interconnect_bandwidth_gbps: 100.0,
            },
            memory_hierarchy: MemoryHierarchy {
                l1_cache_kb: 32,
                l2_cache_kb: 512,
                l3_cache_kb: 32768,
                memory_bandwidth_gbps: 1000.0,
                storage_tier: StorageTier {
                    nvme_capacity_gb: 4000,
                    ssd_capacity_gb: 16000,
                    hdd_capacity_gb: 64000,
                    network_storage_gb: 1000000,
                },
            },
        },
        interconnect: InterconnectConfig {
            topology: NetworkTopology::Mesh,
            bandwidth_profile: BandwidthProfile {
                peak_bandwidth_gbps: 1600.0,
                sustained_bandwidth_gbps: 800.0,
                burst_duration_ms: 10.0,
            },
            latency_profile: LatencyProfile {
                min_latency_us: 0.5,
                avg_latency_us: 1.0,
                max_latency_us: 5.0,
                jitter_us: 0.1,
            },
        },
        specializations: vec![
            WorkloadSpecialization::MachineLearning,
            WorkloadSpecialization::ScientificComputing,
        ],
    }
}

#[tokio::test]
async fn test_multi_system_context() {
    let context = MultiSystemContext::new();
    assert_eq!(context.systems.len(), 0);
}

#[tokio::test]
async fn test_heterogeneous_system_config() {
    let mut context = MultiSystemContext::new();
    context.add_system(test_system(8080));
    assert_eq!(context.systems.len(), 1);
}

#[tokio::test]
async fn test_multi_system_iterator() {
    let context = MultiSystemContext::new();
    let data = vec![1, 2, 3, 4, 5];

    let multi_iter = MultiSystemIterator::new(data, context);
    let result = multi_iter.map_heterogeneous(|x| x * 2).await;

    assert_eq!(
        result
            .expect("multi-system map should complete")
            .collect()
            .await,
        vec![2, 4, 6, 8, 10]
    );
}

#[tokio::test]
async fn non_clone_multi_system_partition_moves_items_by_key() {
    let mut context = MultiSystemContext::new();
    context.add_system(test_system(8081));
    context.add_system(test_system(8082));

    let partitions = context
        .partition_data((0..6).map(NonClone).collect(), |item: &NonClone| {
            item.0 as usize
        })
        .await;

    let mut observed = Vec::new();
    for partition in partitions {
        observed.push(
            partition
                .collect()
                .await
                .into_iter()
                .map(|item| item.0)
                .collect::<Vec<_>>(),
        );
    }

    assert_eq!(observed, vec![vec![0, 2, 4], vec![1, 3, 5]]);
}

#[tokio::test]
async fn non_clone_multi_system_heterogeneous_map_consumes_items() {
    let mut context = MultiSystemContext::new();
    context.add_system(test_system(8083));

    let result = context
        .execute_heterogeneous_compute(
            (0..5).map(NonClone).collect(),
            |item| item.0 * 3,
            |item| item.0 * 3,
        )
        .await
        .expect("multi-system heterogeneous map should complete");

    assert_eq!(result, vec![0, 3, 6, 9, 12]);
}

#[tokio::test]
async fn non_clone_multi_system_iterator_distribution_preserves_values() {
    let mut context = MultiSystemContext::new();
    context.add_system(test_system(8084));
    context.add_system(test_system(8085));

    let partitions = MultiSystemIterator::new((0..6).map(NonClone).collect(), context)
        .distribute_across_systems(|item: &NonClone| item.0 as usize)
        .await;

    let mut observed = Vec::new();
    for partition in partitions {
        observed.push(
            partition
                .collect()
                .await
                .into_iter()
                .map(|item| item.0)
                .collect::<Vec<_>>(),
        );
    }

    assert_eq!(observed, vec![vec![0, 2, 4], vec![1, 3, 5]]);
}

#[test]
fn split_owned_by_ratio_consumes_non_clone_values() {
    let (left, right) = iter::split_owned_by_ratio((0..5).map(NonClone).collect(), 0.6);

    let left_values = left.into_iter().map(|item| item.0).collect::<Vec<_>>();
    let right_values = right.into_iter().map(|item| item.0).collect::<Vec<_>>();

    assert_eq!(left_values, vec![0, 1, 2]);
    assert_eq!(right_values, vec![3, 4]);
}

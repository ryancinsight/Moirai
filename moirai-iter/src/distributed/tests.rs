use super::*;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::time::Duration;

#[derive(Debug, PartialEq)]
struct NonClone(u64);

fn test_node(port: u16, cpu_cores: usize) -> NodeConfig {
    NodeConfig {
        address: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), port),
        cpu_cores,
        memory_gb: 8,
        gpu_config: None,
        latency_profile: LatencyProfile {
            average_latency_ms: 1.0,
            bandwidth_mbps: 1000.0,
            reliability_score: 0.99,
        },
        capabilities: vec![NodeCapability::HighCompute],
    }
}

fn measured_node(
    port: u16,
    cpu_cores: usize,
    average_latency_ms: f64,
    bandwidth_mbps: f64,
) -> NodeConfig {
    let mut node = test_node(port, cpu_cores);
    node.latency_profile = LatencyProfile {
        average_latency_ms,
        bandwidth_mbps,
        reliability_score: 1.0,
    };
    node
}

#[tokio::test]
async fn test_distributed_context_creation() {
    let context = DistributedContext::new();
    assert_eq!(context.nodes.len(), 0);
}

#[tokio::test]
async fn test_node_configuration() {
    let mut context = DistributedContext::new();

    let node = NodeConfig {
        address: SocketAddr::new(IpAddr::V4(Ipv4Addr::new(127, 0, 0, 1)), 8080),
        cpu_cores: 8,
        memory_gb: 16,
        gpu_config: Some(GpuConfig {
            device_count: 1,
            memory_per_device_gb: 8,
            compute_capability: "8.0".to_string(),
            specializations: vec![GpuSpecialization::MachineLearning],
        }),
        latency_profile: LatencyProfile {
            average_latency_ms: 1.0,
            bandwidth_mbps: 1000.0,
            reliability_score: 0.99,
        },
        capabilities: vec![NodeCapability::HighCompute, NodeCapability::LowLatency],
    };

    context.add_node(node);
    assert_eq!(context.nodes.len(), 1);
}

#[tokio::test]
async fn test_data_partitioning() {
    let mut context = DistributedContext::new();

    // Add test nodes
    for i in 0..3 {
        context.add_node(test_node(8080 + i, 4));
    }

    let data = (0..100).collect::<Vec<_>>();
    let partitions = context.partition_data(data, |x| *x % 3).await;

    assert_eq!(partitions.len(), 3);
}

#[tokio::test]
async fn non_clone_distributed_partition_moves_items_by_key() {
    let mut context = DistributedContext::new();
    context.add_node(test_node(9001, 4));
    context.add_node(test_node(9002, 4));

    let data = (0..6).map(NonClone).collect::<Vec<_>>();
    let partitions = context.partition_data(data, |item| item.0 as usize).await;
    let mut partition_values = Vec::new();
    for partition in partitions {
        partition_values.push(
            partition
                .collect()
                .await
                .into_iter()
                .map(|item| item.0)
                .collect::<Vec<_>>(),
        );
    }

    assert_eq!(partition_values, vec![vec![0, 2, 4], vec![1, 3, 5]]);
}

#[tokio::test]
async fn non_clone_distributed_map_consumes_items() {
    let mut context = DistributedContext::new();
    context.add_node(test_node(9011, 1));
    context.add_node(test_node(9012, 3));

    let mapped = context
        .execute_distributed_map((0..5).map(NonClone).collect(), |item| {
            item.0.wrapping_mul(3)
        })
        .await
        .expect("distributed map should consume non-clone items");

    assert_eq!(mapped, vec![0, 3, 6, 9, 12]);
}

#[tokio::test]
async fn non_clone_distributed_reduce_consumes_items() {
    let context = DistributedContext::new();

    let reduced = context
        .execute_distributed_reduce((1..5).map(NonClone).collect(), |left, right| {
            NonClone(left.0 + right.0)
        })
        .await
        .expect("distributed reduce should consume non-clone items");

    assert_eq!(reduced, Some(NonClone(10)));
}

#[tokio::test]
async fn test_distributed_iterator() {
    let context = DistributedContext::new();
    let data = vec![1, 2, 3, 4, 5];

    let dist_iter = DistributedIterator::new(data, context);
    let result = dist_iter
        .map(|x| x * 2)
        .await
        .expect("distributed iterator map should complete")
        .collect()
        .await;

    assert_eq!(result, vec![2, 4, 6, 8, 10]);
}

#[test]
fn distributed_stats_zero_tasks_have_zero_estimate() {
    let context = DistributedContext::new();
    let iterator = DistributedIterator::<u64>::new(Vec::new(), context);

    let stats = iterator.execution_stats();

    assert_eq!(stats.total_nodes, 0);
    assert_eq!(stats.total_tasks, 0);
    assert_eq!(stats.estimated_completion_time, Duration::ZERO);
}

#[test]
fn distributed_stats_local_estimate_scales_with_tasks() {
    let context = DistributedContext::new();
    let iterator = DistributedIterator::new((0..8_u64).collect(), context);

    let stats = iterator.execution_stats();

    assert_eq!(stats.total_nodes, 0);
    assert_eq!(stats.total_tasks, 8);
    assert_eq!(stats.estimated_completion_time, Duration::from_micros(800));
}

#[test]
fn distributed_stats_estimate_uses_node_capacity_latency_and_bandwidth() {
    let mut context = DistributedContext::new();
    context.add_node(measured_node(9101, 4, 2.0, 128.0));
    context.add_node(measured_node(9102, 4, 2.0, 128.0));
    let iterator = DistributedIterator::new((0..16_u64).collect(), context);

    let stats = iterator.execution_stats();

    assert_eq!(stats.total_nodes, 2);
    assert_eq!(stats.total_tasks, 16);
    assert_eq!(
        stats.estimated_completion_time,
        Duration::from_nanos(2_232_000)
    );
}

#[test]
fn distributed_stats_estimate_saturates_extreme_node_metrics() {
    let mut context = DistributedContext::new();
    let mut node = measured_node(9101, 1, f64::MAX, 0.0);
    node.latency_profile.reliability_score = f64::NAN;
    context.add_node(node);
    let iterator = DistributedIterator::new(vec![1_u64], context);

    let stats = iterator.execution_stats();

    assert_eq!(stats.total_nodes, 1);
    assert_eq!(stats.total_tasks, 1);
    assert_eq!(stats.estimated_completion_time, Duration::MAX);
}

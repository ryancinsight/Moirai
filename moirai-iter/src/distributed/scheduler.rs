use super::balancer::LoadBalancer;
use super::config::{NodeCapability, NodeConfig};
use super::iter::{partition_owned_by_key, partition_owned_by_sizes, uniform_partition_sizes};
use super::DistributedError;
use std::collections::HashMap;
use std::time::Duration;

/// Distributed task representation
#[derive(Debug)]
pub struct DistributedTask {
    pub task_id: u64,
    pub node_preference: Option<NodeCapability>,
    pub estimated_duration: Duration,
    pub memory_requirement: usize,
    pub network_dependency: bool,
}

/// Distributed scheduler for optimal task placement
pub struct DistributedScheduler {
    task_history: HashMap<u64, TaskPerformance>,
    load_balancer: LoadBalancer,
}

impl DistributedScheduler {
    pub(super) fn new() -> Self {
        Self {
            task_history: HashMap::new(),
            load_balancer: LoadBalancer::new(),
        }
    }

    pub(super) async fn partition_data_intelligently<T, F>(
        &self,
        data: Vec<T>,
        nodes: &[NodeConfig],
        _partition_func: F,
    ) -> Vec<Vec<T>>
    where
        T: Send + 'static,
        F: Fn(&T) -> usize + Send + Sync + 'static,
    {
        partition_owned_by_key(data, nodes.len(), _partition_func)
    }

    pub(super) async fn map_data_intelligently<T, F, R>(
        &self,
        data: Vec<T>,
        nodes: &[NodeConfig],
        map_func: F,
    ) -> Vec<R>
    where
        T: Send + 'static,
        F: Fn(T) -> R + Send + Sync + 'static,
        R: Send + 'static,
    {
        let item_count = data.len();
        if nodes.is_empty() {
            return data.into_iter().map(map_func).collect();
        }

        let partition_sizes = self.calculate_optimal_partitions(nodes, item_count);
        let partitions = partition_owned_by_sizes(data, &partition_sizes);
        let mut results = Vec::with_capacity(item_count);

        for partition in partitions {
            results.extend(partition.into_iter().map(&map_func));
        }

        results
    }

    pub(super) fn calculate_optimal_partitions(
        &self,
        nodes: &[NodeConfig],
        total_items: usize,
    ) -> Vec<usize> {
        if nodes.is_empty() {
            return Vec::new();
        }
        if total_items == 0 {
            return (0..nodes.len()).map(|_| 0).collect();
        }

        let total_compute_power: usize = nodes.iter().map(|n| n.cpu_cores).sum();
        if total_compute_power == 0 {
            return uniform_partition_sizes(total_items, nodes.len());
        }

        let mut assigned = 0;
        let mut sizes = Vec::with_capacity(nodes.len());
        for (index, node) in nodes.iter().enumerate() {
            let size = if index + 1 == nodes.len() {
                total_items - assigned
            } else {
                let size = total_items * node.cpu_cores / total_compute_power;
                assigned += size;
                size
            };
            sizes.push(size);
        }
        sizes
    }

    pub(super) async fn create_map_tasks<T, F, R>(
        &self,
        data: Vec<T>,
        _map_func: F,
    ) -> Vec<DistributedTask>
    where
        T: Send + 'static,
        F: Fn(T) -> R + Send + Sync + 'static,
        R: Send + 'static,
    {
        // Create tasks with estimated performance characteristics
        data.into_iter()
            .enumerate()
            .map(|(idx, _item)| {
                DistributedTask {
                    task_id: idx as u64,
                    node_preference: None,
                    estimated_duration: Duration::from_millis(10), // Estimated from history
                    memory_requirement: 1024,                      // Estimated memory usage
                    network_dependency: false,
                }
            })
            .collect()
    }

    pub(super) async fn tree_reduce<T, F>(
        &self,
        data: Vec<T>,
        reduce_func: F,
        _nodes: &[NodeConfig],
    ) -> Result<Option<T>, DistributedError>
    where
        T: Send + 'static,
        F: Fn(T, T) -> T + Send + Sync + 'static,
    {
        if data.is_empty() {
            return Ok(None);
        }

        let result = data.into_iter().reduce(reduce_func);
        Ok(result)
    }

    pub(super) fn assign_tasks_to_nodes<'a>(
        &self,
        tasks: &'a [DistributedTask],
        nodes: &[NodeConfig],
    ) -> HashMap<usize, Vec<&'a DistributedTask>> {
        let mut assignments = HashMap::new();

        // Round-robin assignment for simplicity
        // Real implementation would use sophisticated load balancing
        if nodes.is_empty() {
            return assignments;
        }
        for (idx, task) in tasks.iter().enumerate() {
            let node_idx = idx % nodes.len();
            assignments
                .entry(node_idx)
                .or_insert_with(Vec::new)
                .push(task);
        }

        assignments
    }
}

/// Task performance tracking
#[derive(Debug)]
pub struct TaskPerformance {
    pub execution_time: Duration,
    pub memory_used: usize,
    pub network_bytes: u64,
    pub success_rate: f64,
}

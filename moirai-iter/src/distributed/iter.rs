use std::time::Duration;
use super::context::DistributedContext;
use super::config::NodeConfig;
use super::DistributedError;

const LOCAL_TASK_ESTIMATE_SECS: f64 = 0.000_1;
const ESTIMATED_TASK_BYTES: f64 = 64.0;

/// Distributed processing iterator
pub struct DistributedIterator<T> {
    pub(super) data: Vec<T>,
    pub(super) context: DistributedContext,
}

impl<T: Send + 'static> DistributedIterator<T> {
    pub fn new(data: Vec<T>, context: DistributedContext) -> Self {
        Self { data, context }
    }

    /// Map operation distributed across nodes
    pub async fn map<F, R>(self, map_func: F) -> Result<DistributedIterator<R>, DistributedError>
    where
        F: Fn(T) -> R + Send + Sync + 'static,
        R: Send + 'static,
    {
        let results = self
            .context
            .execute_distributed_map(self.data, map_func)
            .await?;
        Ok(DistributedIterator::new(results, self.context))
    }

    /// Reduce operation with tree-reduce optimization
    pub async fn reduce<F>(self, reduce_func: F) -> Result<Option<T>, DistributedError>
    where
        F: Fn(T, T) -> T + Send + Sync + 'static,
    {
        self.context
            .execute_distributed_reduce(self.data, reduce_func)
            .await
    }

    /// Collect results from all nodes
    pub async fn collect(self) -> Vec<T> {
        self.data
    }

    /// Gather statistics about distributed execution
    pub fn execution_stats(&self) -> DistributedStats {
        DistributedStats {
            total_nodes: self.context.nodes.len(),
            total_tasks: self.data.len(),
            estimated_completion_time: estimate_completion_time(
                &self.context.nodes,
                self.data.len(),
            ),
        }
    }
}

pub(super) fn partition_owned_by_key<T, F>(
    data: Vec<T>,
    partition_count: usize,
    partition_func: F,
) -> Vec<Vec<T>>
where
    F: Fn(&T) -> usize,
{
    if partition_count == 0 {
        return vec![data];
    }

    let mut partitions: Vec<Vec<T>> = (0..partition_count).map(|_| Vec::new()).collect();
    for item in data {
        let partition_index = partition_func(&item) % partition_count;
        partitions[partition_index].push(item);
    }
    partitions
}

pub(super) fn partition_owned_by_sizes<T>(data: Vec<T>, partition_sizes: &[usize]) -> Vec<Vec<T>> {
    if partition_sizes.is_empty() {
        return vec![data];
    }

    let mut remaining = data.len();
    let mut iter = data.into_iter();
    let mut partitions = Vec::with_capacity(partition_sizes.len());

    for size in partition_sizes {
        let take = (*size).min(remaining);
        let partition: Vec<T> = iter.by_ref().take(take).collect();
        remaining -= partition.len();
        partitions.push(partition);
    }

    if remaining > 0 {
        if let Some(last) = partitions.last_mut() {
            last.extend(iter);
        } else {
            partitions.push(iter.collect());
        }
    }

    partitions
}

pub(super) fn uniform_partition_sizes(total_items: usize, partition_count: usize) -> Vec<usize> {
    if partition_count == 0 {
        return Vec::new();
    }

    let base = total_items / partition_count;
    let remainder = total_items % partition_count;
    (0..partition_count)
        .map(|index| base + usize::from(index < remainder))
        .collect()
}

pub(super) fn estimate_completion_time(nodes: &[NodeConfig], task_count: usize) -> Duration {
    if task_count == 0 {
        return Duration::ZERO;
    }

    if nodes.is_empty() {
        return duration_from_secs_saturating(task_count as f64 * LOCAL_TASK_ESTIMATE_SECS);
    }

    let effective_parallelism = nodes
        .iter()
        .map(|node| {
            let reliability =
                finite_clamped(node.latency_profile.reliability_score, 0.01, 1.0, 1.0);
            node.cpu_cores.max(1) as f64 * reliability
        })
        .sum::<f64>()
        .max(1.0);
    let compute_waves = (task_count as f64 / effective_parallelism).ceil();
    let compute_seconds = compute_waves * LOCAL_TASK_ESTIMATE_SECS;

    let latency_seconds = nodes
        .iter()
        .map(|node| finite_non_negative(node.latency_profile.average_latency_ms, 0.0))
        .sum::<f64>()
        / nodes.len() as f64
        / 1_000.0;

    let aggregate_bandwidth_mbps = nodes
        .iter()
        .map(|node| finite_non_negative(node.latency_profile.bandwidth_mbps, 0.0))
        .sum::<f64>();
    let network_seconds = if aggregate_bandwidth_mbps > 0.0 {
        task_count as f64 * ESTIMATED_TASK_BYTES * 8.0 / (aggregate_bandwidth_mbps * 1_000_000.0)
    } else {
        0.0
    };

    duration_from_secs_saturating(compute_seconds + latency_seconds + network_seconds)
}

pub(super) fn duration_from_secs_saturating(seconds: f64) -> Duration {
    if !seconds.is_finite() {
        return Duration::MAX;
    }

    if seconds <= 0.0 {
        return Duration::ZERO;
    }

    let max_seconds = Duration::MAX.as_secs() as f64;
    if seconds >= max_seconds {
        Duration::MAX
    } else {
        Duration::from_secs_f64(seconds)
    }
}

pub(super) fn finite_non_negative(value: f64, fallback: f64) -> f64 {
    if value.is_finite() && value >= 0.0 {
        value
    } else {
        fallback
    }
}

pub(super) fn finite_clamped(value: f64, min: f64, max: f64, fallback: f64) -> f64 {
    if value.is_finite() {
        value.clamp(min, max)
    } else {
        fallback
    }
}

/// Statistics for distributed execution
#[derive(Debug)]
pub struct DistributedStats {
    pub total_nodes: usize,
    pub total_tasks: usize,
    pub estimated_completion_time: Duration,
}

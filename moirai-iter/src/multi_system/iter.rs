use super::context::MultiSystemContext;
use super::MultiSystemError;
use std::time::Duration;

/// Multi-system iterator for coordinated processing
pub struct MultiSystemIterator<T> {
    pub(super) data: Vec<T>,
    pub(super) context: MultiSystemContext,
}

impl<T: Send + 'static> MultiSystemIterator<T> {
    pub fn new(data: Vec<T>, context: MultiSystemContext) -> Self {
        Self { data, context }
    }

    /// Execute heterogeneous compute across CPU and GPU systems
    pub async fn map_heterogeneous<F, R>(
        self,
        func: F,
    ) -> Result<MultiSystemIterator<R>, MultiSystemError>
    where
        F: Fn(T) -> R + Send + Sync + 'static,
        R: Send + 'static,
    {
        let results = self
            .context
            .execute_shared_heterogeneous_compute(self.data, func)
            .await?;
        Ok(MultiSystemIterator::new(results, self.context))
    }

    /// Partition and distribute across multiple systems
    pub async fn distribute_across_systems<F>(
        self,
        partition_func: F,
    ) -> Vec<MultiSystemIterator<T>>
    where
        F: Fn(&T) -> usize + Send + Sync + 'static,
    {
        let partitions = self.context.partition_data(self.data, partition_func).await;

        let mut iterators = Vec::with_capacity(partitions.len());
        for partition in partitions {
            iterators.push(MultiSystemIterator::new(
                partition.collect().await,
                self.context.clone(),
            ));
        }
        iterators
    }

    /// Collect results from all systems
    pub async fn collect(self) -> Vec<T> {
        self.data
    }

    /// Get execution statistics across all systems
    pub fn system_stats(&self) -> MultiSystemStats {
        MultiSystemStats {
            total_systems: self.context.systems.len(),
            total_cpu_cores: self
                .context
                .systems
                .iter()
                .map(|s| s.cpu_cluster.total_cores)
                .sum(),
            total_gpu_devices: self
                .context
                .systems
                .iter()
                .filter_map(|s| s.gpu_cluster.as_ref())
                .map(|g| g.node_count * g.gpus_per_node)
                .sum(),
            estimated_completion_time: Duration::from_secs(15),
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

pub(super) fn split_owned_by_ratio<T>(data: Vec<T>, ratio: f64) -> (Vec<T>, Vec<T>) {
    let split_point = ((data.len() as f64) * ratio.clamp(0.0, 1.0)) as usize;
    let mut left = Vec::with_capacity(split_point);
    let mut right = Vec::with_capacity(data.len().saturating_sub(split_point));

    for (index, item) in data.into_iter().enumerate() {
        if index < split_point {
            left.push(item);
        } else {
            right.push(item);
        }
    }

    (left, right)
}

pub(super) fn map_owned_compute<T, F, R>(data: Vec<T>, func: &F) -> Vec<R>
where
    F: Fn(T) -> R,
{
    data.into_iter().map(func).collect()
}

/// Statistics for multi-system execution
#[derive(Debug)]
pub struct MultiSystemStats {
    pub total_systems: usize,
    pub total_cpu_cores: usize,
    pub total_gpu_devices: usize,
    pub estimated_completion_time: Duration,
}

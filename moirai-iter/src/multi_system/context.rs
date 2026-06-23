use super::allocation::ComputeAllocation;
use super::config::SystemConfig;
use super::iter::{map_owned_compute, split_owned_by_ratio};
use super::optimizer::TopologyOptimizer;
use super::profile::{
    ComputeIntensity, DataProfile, GpuSuitabilityScore, MemoryAccessPattern, ParallelizabilityScore,
};
use super::resource::ResourceManager;
use super::scheduler::UnifiedScheduler;
use super::MultiSystemError;
use crate::MoiraiIterator;
use std::sync::{Arc, Mutex};

/// Multi-system execution context
#[derive(Clone)]
pub struct MultiSystemContext {
    pub(super) systems: Arc<Vec<SystemConfig>>,
    pub(super) unified_scheduler: Arc<UnifiedScheduler>,
    pub(super) resource_manager: Arc<ResourceManager>,
    pub(super) topology_optimizer: Arc<Mutex<TopologyOptimizer>>,
}

impl MultiSystemContext {
    /// Create a new multi-system context
    pub fn new() -> Self {
        Self {
            systems: Arc::new(Vec::new()),
            unified_scheduler: Arc::new(UnifiedScheduler::new()),
            resource_manager: Arc::new(ResourceManager::new()),
            topology_optimizer: Arc::new(Mutex::new(TopologyOptimizer::new())),
        }
    }

    /// Add a system to the multi-system cluster
    pub fn add_system(&mut self, system: SystemConfig) {
        Arc::get_mut(&mut self.systems).unwrap().push(system);
        self.topology_optimizer
            .lock()
            .unwrap()
            .update_topology(&self.systems);
    }

    /// Partition data across multiple systems with intelligent placement
    pub async fn partition_data<T, F>(
        &self,
        data: Vec<T>,
        partition_func: F,
    ) -> Vec<MoiraiIterator<T>>
    where
        T: Send + 'static,
        F: Fn(&T) -> usize + Send + Sync + 'static,
    {
        let data_profile = self.analyze_data_characteristics(&data).await;

        let assignments = self
            .unified_scheduler
            .assign_data_to_systems(data, &data_profile, &self.systems, partition_func)
            .await;

        assignments
            .into_iter()
            .map(|partition| MoiraiIterator::multi_system(partition))
            .collect()
    }

    /// Execute coordinated compute across CPU and GPU clusters
    pub async fn execute_heterogeneous_compute<T, CpuF, GpuF, R>(
        &self,
        data: Vec<T>,
        cpu_func: CpuF,
        gpu_func: GpuF,
    ) -> Result<Vec<R>, MultiSystemError>
    where
        T: Send + 'static,
        CpuF: Fn(T) -> R + Send + Sync + 'static,
        GpuF: Fn(T) -> R + Send + Sync + 'static,
        R: Send + 'static,
    {
        // Determine CPU vs GPU allocation based on workload characteristics
        let allocation = self
            .resource_manager
            .determine_compute_allocation(&data)
            .await;

        // Execute on appropriate compute units
        let results = match allocation {
            ComputeAllocation::CpuOnly => self.execute_cpu_compute(data, cpu_func).await?,
            ComputeAllocation::GpuOnly => self.execute_gpu_compute(data, gpu_func).await?,
            ComputeAllocation::Hybrid { cpu_ratio } => {
                self.execute_hybrid_compute(data, cpu_func, gpu_func, cpu_ratio)
                    .await?
            }
        };

        Ok(results)
    }

    pub(super) async fn analyze_data_characteristics<T>(&self, data: &[T]) -> DataProfile {
        DataProfile {
            size: data.len(),
            estimated_compute_intensity: ComputeIntensity::Medium,
            memory_access_pattern: MemoryAccessPattern::Sequential,
            parallelizability: ParallelizabilityScore(0.8),
            gpu_suitability: GpuSuitabilityScore(0.6),
        }
    }

    pub(super) async fn execute_cpu_compute<T, F, R>(
        &self,
        data: Vec<T>,
        func: F,
    ) -> Result<Vec<R>, MultiSystemError>
    where
        T: Send + 'static,
        F: Fn(T) -> R + Send + Sync + 'static,
        R: Send + 'static,
    {
        self.unified_scheduler
            .execute_numa_aware_compute(data, func)
            .await
    }

    pub(super) async fn execute_gpu_compute<T, F, R>(
        &self,
        data: Vec<T>,
        func: F,
    ) -> Result<Vec<R>, MultiSystemError>
    where
        T: Send + 'static,
        F: Fn(T) -> R + Send + Sync + 'static,
        R: Send + 'static,
    {
        self.unified_scheduler
            .execute_gpu_cluster_compute(data, func)
            .await
    }

    pub(super) async fn execute_hybrid_compute<T, CpuF, GpuF, R>(
        &self,
        data: Vec<T>,
        cpu_func: CpuF,
        gpu_func: GpuF,
        cpu_ratio: f64,
    ) -> Result<Vec<R>, MultiSystemError>
    where
        T: Send + 'static,
        CpuF: Fn(T) -> R + Send + Sync + 'static,
        GpuF: Fn(T) -> R + Send + Sync + 'static,
        R: Send + 'static,
    {
        let (cpu_data, gpu_data) = split_owned_by_ratio(data, cpu_ratio);

        let (cpu_results, gpu_results) = futures::future::join(
            self.execute_cpu_compute(cpu_data, cpu_func),
            self.execute_gpu_compute(gpu_data, gpu_func),
        )
        .await;

        let mut combined_results = cpu_results?;
        combined_results.extend(gpu_results?);
        Ok(combined_results)
    }

    pub(super) async fn execute_shared_heterogeneous_compute<T, F, R>(
        &self,
        data: Vec<T>,
        func: F,
    ) -> Result<Vec<R>, MultiSystemError>
    where
        T: Send + 'static,
        F: Fn(T) -> R + Send + Sync + 'static,
        R: Send + 'static,
    {
        match self
            .resource_manager
            .determine_compute_allocation(&data)
            .await
        {
            ComputeAllocation::CpuOnly => self.execute_cpu_compute(data, func).await,
            ComputeAllocation::GpuOnly => self.execute_gpu_compute(data, func).await,
            ComputeAllocation::Hybrid { cpu_ratio } => {
                let (cpu_data, gpu_data) = split_owned_by_ratio(data, cpu_ratio);
                let mut results = map_owned_compute(cpu_data, &func);
                results.extend(map_owned_compute(gpu_data, &func));
                Ok(results)
            }
        }
    }
}

impl Default for MultiSystemContext {
    fn default() -> Self {
        Self::new()
    }
}

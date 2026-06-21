use super::config::SystemConfig;
use super::profile::DataProfile;
use super::balancer::MultiSystemLoadBalancer;
use super::MultiSystemError;
use super::iter::{partition_owned_by_key, map_owned_compute};

/// Unified scheduler for multi-system coordination
pub struct UnifiedScheduler {
    scheduling_strategy: SchedulingStrategy,
    load_balancer: MultiSystemLoadBalancer,
}

impl UnifiedScheduler {
    pub(super) fn new() -> Self {
        Self {
            scheduling_strategy: SchedulingStrategy::WorkStealingHybrid,
            load_balancer: MultiSystemLoadBalancer::new(),
        }
    }

    pub(super) async fn assign_data_to_systems<T, F>(
        &self,
        data: Vec<T>,
        _profile: &DataProfile,
        systems: &[SystemConfig],
        partition_func: F,
    ) -> Vec<Vec<T>>
    where
        F: Fn(&T) -> usize,
    {
        let system_count = systems.len();
        if system_count == 0 {
            return vec![data];
        }

        partition_owned_by_key(data, system_count, partition_func)
    }

    pub(super) async fn execute_numa_aware_compute<T, F, R>(
        &self,
        data: Vec<T>,
        func: F,
    ) -> Result<Vec<R>, MultiSystemError>
    where
        T: Send + 'static,
        F: Fn(T) -> R + Send + Sync + 'static,
        R: Send + 'static,
    {
        Ok(map_owned_compute(data, &func))
    }

    pub(super) async fn execute_gpu_cluster_compute<T, F, R>(
        &self,
        data: Vec<T>,
        func: F,
    ) -> Result<Vec<R>, MultiSystemError>
    where
        T: Send + 'static,
        F: Fn(T) -> R + Send + Sync + 'static,
        R: Send + 'static,
    {
        Ok(map_owned_compute(data, &func))
    }
}

/// Scheduling strategy for multi-system coordination
#[derive(Debug)]
pub enum SchedulingStrategy {
    WorkStealingHybrid,
    LocalityAware,
    LoadBalanced,
    LatencyOptimized,
    ThroughputOptimized,
}

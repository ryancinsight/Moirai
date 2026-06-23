use super::config::NodeConfig;
use super::failure::FailureHandler;
use super::scheduler::DistributedScheduler;
use super::DistributedError;
use crate::MoiraiIterator;
use std::net::SocketAddr;
use std::sync::Arc;

/// Distributed execution context for multi-machine processing
#[derive(Clone)]
pub struct DistributedContext {
    pub(super) nodes: Arc<Vec<NodeConfig>>,
    pub(super) coordinator_address: Option<SocketAddr>,
    pub(super) task_scheduler: Arc<DistributedScheduler>,
    pub(super) failure_handler: Arc<FailureHandler>,
}

impl DistributedContext {
    /// Create a new distributed context
    pub fn new() -> Self {
        Self {
            nodes: Arc::new(Vec::new()),
            coordinator_address: None,
            task_scheduler: Arc::new(DistributedScheduler::new()),
            failure_handler: Arc::new(FailureHandler::new()),
        }
    }

    /// Add a compute node to the distributed cluster
    pub fn add_node(&mut self, node: NodeConfig) {
        Arc::get_mut(&mut self.nodes).unwrap().push(node);
    }

    /// Set the coordinator node address
    pub fn set_coordinator(&mut self, address: SocketAddr) {
        self.coordinator_address = Some(address);
    }

    /// Partition data across multiple nodes based on node capabilities
    pub async fn partition_data<T, F>(
        &self,
        data: Vec<T>,
        partition_func: F,
    ) -> Vec<MoiraiIterator<T>>
    where
        T: Send + 'static,
        F: Fn(&T) -> usize + Send + Sync + 'static,
    {
        let node_count = self.nodes.len();
        if node_count == 0 {
            return vec![MoiraiIterator::distributed(data)];
        }

        // Intelligent partitioning based on node capabilities
        let partitions = self
            .task_scheduler
            .partition_data_intelligently(data, &self.nodes, partition_func)
            .await;

        partitions
            .into_iter()
            .map(|partition| MoiraiIterator::distributed(partition))
            .collect()
    }

    /// Execute a distributed map operation
    pub async fn execute_distributed_map<T, F, R>(
        &self,
        data: Vec<T>,
        map_func: F,
    ) -> Result<Vec<R>, DistributedError>
    where
        T: Send + 'static,
        F: Fn(T) -> R + Send + Sync + 'static,
        R: Send + 'static,
    {
        let results = self
            .task_scheduler
            .map_data_intelligently(data, &self.nodes, map_func)
            .await;
        Ok(results)
    }

    /// Execute a distributed reduce operation
    pub async fn execute_distributed_reduce<T, F>(
        &self,
        data: Vec<T>,
        reduce_func: F,
    ) -> Result<Option<T>, DistributedError>
    where
        T: Send + 'static,
        F: Fn(T, T) -> T + Send + Sync + 'static,
    {
        // Tree-reduce across nodes for optimal network usage
        let result = self
            .task_scheduler
            .tree_reduce(data, reduce_func, &self.nodes)
            .await?;
        Ok(result)
    }
}

impl Default for DistributedContext {
    fn default() -> Self {
        Self::new()
    }
}

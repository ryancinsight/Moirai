//! CPU topology adapter backed by Themis placement law.

use std::collections::HashMap;

/// NUMA-aware work stealing scheduler topology.
#[derive(Debug, Clone)]
pub struct CpuTopology {
    /// Number of NUMA nodes.
    pub numa_nodes: Vec<NumaNode>,
    /// Mapping from CPU core to NUMA node.
    pub core_to_node: HashMap<usize, usize>,
    /// Total number of logical cores.
    pub logical_cores: usize,
    /// Cache hierarchy information.
    pub cache_levels: Vec<CacheLevel>,
}

/// NUMA node information.
#[derive(Debug, Clone)]
pub struct NumaNode {
    /// Node ID.
    pub id: usize,
    /// CPU cores belonging to this node.
    pub cores: Vec<usize>,
    /// Distance to other NUMA nodes.
    pub distances: Vec<u32>,
}

/// Cache level information.
#[derive(Debug, Clone)]
pub struct CacheLevel {
    /// Cache level.
    pub level: u32,
    /// Cache size in bytes.
    pub size: usize,
    /// Cores sharing this cache.
    pub shared_cores: Vec<usize>,
}

impl CpuTopology {
    /// Detect the CPU topology from Themis.
    pub fn detect() -> Option<Self> {
        themis::CpuTopology::detect().map(Self::from_themis)
    }

    /// Create a single-node topology for systems without NUMA.
    pub fn single_node() -> Self {
        Self::from_themis(themis::CpuTopology::single_node(
            std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1),
        ))
    }

    /// Get the NUMA node for a given CPU core.
    pub fn core_to_numa_node(&self, core_id: usize) -> Option<usize> {
        self.core_to_node.get(&core_id).copied()
    }

    /// Get cores in the same NUMA node as the given core.
    pub fn cores_in_same_node(&self, core_id: usize) -> Vec<usize> {
        if let Some(node_id) = self.core_to_numa_node(core_id) {
            self.numa_nodes
                .get(node_id)
                .map(|node| node.cores.clone())
                .unwrap_or_default()
        } else {
            Vec::new()
        }
    }

    /// Get adjacent NUMA nodes sorted by distance.
    pub fn adjacent_nodes(&self, node_id: usize) -> Vec<usize> {
        if let Some(node) = self.numa_nodes.get(node_id) {
            let mut adjacent: Vec<_> = node
                .distances
                .iter()
                .enumerate()
                .filter(|(id, _)| *id != node_id)
                .map(|(id, &distance)| (id, distance))
                .collect();
            adjacent.sort_by_key(|&(_, distance)| distance);
            adjacent.into_iter().map(|(id, _)| id).collect()
        } else {
            Vec::new()
        }
    }

    /// Get distance between two NUMA nodes.
    pub fn distance(&self, from_node: usize, to_node: usize) -> u32 {
        if let Some(from) = self.numa_nodes.iter().find(|node| node.id == from_node) {
            if to_node < from.distances.len() {
                return from.distances[to_node];
            }
        }

        if from_node == to_node {
            10
        } else {
            20
        }
    }

    fn from_themis(topology: themis::CpuTopology) -> Self {
        let core_to_node = topology
            .processor_node_pairs()
            .map(|(processor, node)| (processor as usize, node.index()))
            .collect();

        let numa_nodes = topology
            .numa_nodes
            .into_iter()
            .map(|node| NumaNode {
                id: node.id.index(),
                cores: node
                    .processors
                    .into_iter()
                    .map(|processor| processor as usize)
                    .collect(),
                distances: node.distances,
            })
            .collect();

        let cache_levels = topology
            .cache_levels
            .into_iter()
            .map(|cache| CacheLevel {
                level: cache.level,
                size: cache.size_bytes,
                shared_cores: cache
                    .shared_processors
                    .into_iter()
                    .map(|processor| processor as usize)
                    .collect(),
            })
            .collect();

        Self {
            numa_nodes,
            core_to_node,
            logical_cores: topology.logical_processors,
            cache_levels,
        }
    }
}

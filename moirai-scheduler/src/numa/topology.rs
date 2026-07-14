//! CPU topology adapter backed by Themis placement law.

/// NUMA-aware work stealing scheduler topology.
#[derive(Debug, Clone)]
pub struct CpuTopology {
    /// Number of NUMA nodes.
    pub numa_nodes: Box<[NumaNode]>,
    /// Adjacent NUMA node order by compact node index.
    pub adjacent_nodes: Box<[Box<[usize]>]>,
    /// Mapping from CPU core to NUMA node.
    pub core_to_node: Box<[Option<usize>]>,
    /// Total number of logical cores.
    pub logical_cores: usize,
    /// Cache hierarchy information.
    pub cache_levels: Box<[CacheLevel]>,
}

/// NUMA node information.
#[derive(Debug, Clone)]
pub struct NumaNode {
    /// Node ID.
    pub id: usize,
    /// CPU cores belonging to this node.
    pub cores: Box<[usize]>,
    /// Distance to other NUMA nodes.
    pub distances: Box<[u32]>,
}

/// Cache level information.
#[derive(Debug, Clone)]
pub struct CacheLevel {
    /// Cache level.
    pub level: u32,
    /// Cache size in bytes.
    pub size: usize,
    /// Cores sharing this cache.
    pub shared_cores: Box<[usize]>,
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
        self.core_to_node.get(core_id).copied().flatten()
    }

    /// Get cores in the same NUMA node as the given core.
    pub fn cores_in_same_node(&self, core_id: usize) -> Vec<usize> {
        if let Some(node_id) = self.core_to_numa_node(core_id) {
            self.numa_nodes
                .get(node_id)
                .map(|node| node.cores.to_vec())
                .unwrap_or_default()
        } else {
            Vec::new()
        }
    }

    /// Get adjacent NUMA nodes sorted by distance.
    pub fn adjacent_nodes(&self, node_id: usize) -> Vec<usize> {
        self.adjacent_node_slice(node_id).to_vec()
    }

    /// Get adjacent NUMA nodes sorted by distance without allocation.
    pub fn adjacent_node_slice(&self, node_id: usize) -> &[usize] {
        self.adjacent_nodes.get(node_id).map_or(&[], |nodes| nodes)
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
        let core_to_node = build_core_to_node(topology.logical_processors(), &topology);

        let numa_nodes = topology
            .numa_nodes()
            .iter()
            .map(|node| NumaNode {
                id: node.id.index(),
                cores: node
                    .processors
                    .iter()
                    .map(|processor| *processor as usize)
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
                distances: node.distances.to_vec().into_boxed_slice(),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let adjacent_nodes = topology
            .numa_nodes()
            .iter()
            .map(|node| {
                topology
                    .adjacent_nodes(node.id)
                    .iter()
                    .map(|node_id| node_id.index())
                    .collect::<Vec<_>>()
                    .into_boxed_slice()
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        let cache_levels = topology
            .cache_levels()
            .unwrap_or(&[])
            .iter()
            .map(|cache| CacheLevel {
                level: cache.level,
                size: cache.size_bytes,
                shared_cores: cache
                    .shared_processors
                    .iter()
                    .map(|processor| *processor as usize)
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
            })
            .collect::<Vec<_>>()
            .into_boxed_slice();

        Self {
            numa_nodes,
            adjacent_nodes,
            core_to_node,
            logical_cores: topology.logical_processors(),
            cache_levels,
        }
    }
}

fn build_core_to_node(
    logical_cores: usize,
    topology: &themis::CpuTopology,
) -> Box<[Option<usize>]> {
    let max_core = topology
        .processor_node_pairs()
        .map(|(processor, _)| processor as usize)
        .max()
        .unwrap_or(0);
    let mut core_to_node = vec![None; logical_cores.max(max_core + 1).max(1)];
    for (processor, node) in topology.processor_node_pairs() {
        core_to_node[processor as usize] = Some(node.index());
    }
    core_to_node.into_boxed_slice()
}

#[cfg(test)]
mod tests {
    use super::CpuTopology;

    #[test]
    fn single_node_topology_covers_all_cores_under_one_node() {
        let topo = CpuTopology::single_node();
        // A non-NUMA system is modelled as exactly one node.
        assert_eq!(topo.numa_nodes.len(), 1);
        assert!(topo.logical_cores >= 1);
        // Core 0 resolves to the sole node, and that node owns every logical core.
        assert_eq!(topo.core_to_numa_node(0), Some(0));
        assert_eq!(topo.cores_in_same_node(0).len(), topo.logical_cores);
        // A single node has no other node to be adjacent to.
        assert!(topo.adjacent_nodes(0).is_empty());
    }
}

//! CPU topology detection: `CpuTopology`, `NumaNode`, `CacheLevel`.

use std::collections::HashMap;

/// NUMA-aware work stealing scheduler.
#[derive(Debug, Clone)]
pub struct CpuTopology {
    /// Number of NUMA nodes
    pub numa_nodes: Vec<NumaNode>,
    /// Mapping from CPU core to NUMA node
    pub core_to_node: HashMap<usize, usize>,
    /// Total number of logical cores
    pub logical_cores: usize,
    /// Cache hierarchy information
    pub cache_levels: Vec<CacheLevel>,
}

/// NUMA node information.
#[derive(Debug, Clone)]
pub struct NumaNode {
    /// Node ID
    pub id: usize,
    /// CPU cores belonging to this node
    pub cores: Vec<usize>,
    /// Distance to other NUMA nodes (lower = closer)
    pub distances: Vec<u32>,
}

/// Cache level information.
#[derive(Debug, Clone)]
pub struct CacheLevel {
    /// Cache level (1, 2, 3, etc.)
    pub level: u32,
    /// Cache size in bytes
    pub size: usize,
    /// Cores sharing this cache
    pub shared_cores: Vec<usize>,
}

impl CpuTopology {
    /// Detect the CPU topology from the system.
    pub fn detect() -> Option<Self> {
        #[cfg(target_os = "linux")]
        {
            // Try to read from /sys/devices/system/cpu/
            Self::detect_linux()
        }

        #[cfg(target_os = "windows")]
        {
            Self::detect_windows()
        }

        #[cfg(not(any(target_os = "linux", target_os = "windows")))]
        {
            // Fallback: assume single NUMA node with all cores
            Some(Self::single_node())
        }
    }

    #[cfg(target_os = "linux")]
    fn detect_linux() -> Option<Self> {
        use std::fs;

        // Read number of NUMA nodes
        let nodes_path = "/sys/devices/system/node/";
        let node_count = fs::read_dir(nodes_path)
            .ok()?
            .filter_map(|entry| entry.ok())
            .filter(|entry| {
                entry
                    .file_name()
                    .to_str()
                    .map(|s| s.starts_with("node"))
                    .unwrap_or(false)
            })
            .count();

        if node_count == 0 {
            return Some(Self::single_node());
        }

        let mut numa_nodes = Vec::new();
        let mut core_to_node = HashMap::new();

        // Read NUMA node information
        for node_id in 0..node_count {
            let cpulist_path = format!("{}/node{}/cpulist", nodes_path, node_id);
            if let Ok(cpulist) = fs::read_to_string(&cpulist_path) {
                let cores = Self::parse_cpu_list(&cpulist);
                for &core in &cores {
                    core_to_node.insert(core, node_id);
                }

                let distance_path = format!("{}/node{}/distance", nodes_path, node_id);
                let distances = if let Ok(distance_str) = fs::read_to_string(&distance_path) {
                    distance_str
                        .split_whitespace()
                        .filter_map(|s| s.parse().ok())
                        .collect()
                } else {
                    vec![10; node_count] // Default distances
                };

                numa_nodes.push(NumaNode {
                    id: node_id,
                    cores,
                    distances,
                });
            }
        }

        // Detect logical cores
        let logical_cores = num_cpus::get();

        // Basic cache detection (simplified)
        let cache_levels = vec![
            CacheLevel {
                level: 1,
                size: 32 * 1024, // 32KB L1
                shared_cores: vec![],
            },
            CacheLevel {
                level: 2,
                size: 256 * 1024, // 256KB L2
                shared_cores: vec![],
            },
            CacheLevel {
                level: 3,
                size: 8 * 1024 * 1024, // 8MB L3
                shared_cores: (0..logical_cores).collect(),
            },
        ];

        Some(CpuTopology {
            numa_nodes,
            core_to_node,
            logical_cores,
            cache_levels,
        })
    }

    #[cfg(target_os = "linux")]
    fn parse_cpu_list(cpulist: &str) -> Vec<usize> {
        let mut cores = Vec::new();
        for part in cpulist.trim().split(',') {
            if let Some(dash_pos) = part.find('-') {
                let (start, end) = part.split_at(dash_pos);
                if let (Ok(start), Ok(end)) = (start.parse::<usize>(), end[1..].parse::<usize>()) {
                    cores.extend(start..=end);
                }
            } else if let Ok(core) = part.parse::<usize>() {
                cores.push(core);
            }
        }
        cores
    }

    #[cfg(target_os = "windows")]
    fn detect_windows() -> Option<Self> {
        extern "system" {
            fn GetNumaHighestNodeNumber(highest_node_number: *mut u32) -> i32;
            fn GetNumaNodeProcessorMask(node: u8, processor_mask: *mut u64) -> i32;
        }

        let mut highest_node = 0u32;
        if unsafe { GetNumaHighestNodeNumber(&mut highest_node) } == 0 {
            return Some(Self::single_node());
        }

        let node_count = (highest_node + 1) as usize;
        let mut numa_nodes = Vec::new();
        let mut core_to_node = HashMap::new();
        let mut logical_cores = 0;

        for node_id in 0..node_count {
            let mut mask = 0u64;
            if unsafe { GetNumaNodeProcessorMask(node_id as u8, &mut mask) } != 0 && mask != 0 {
                let mut cores = Vec::new();
                for core in 0..64 {
                    if (mask & (1 << core)) != 0 {
                        cores.push(core as usize);
                        core_to_node.insert(core as usize, node_id);
                        logical_cores = logical_cores.max(core as usize + 1);
                    }
                }

                numa_nodes.push(NumaNode {
                    id: node_id,
                    cores,
                    distances: vec![10; node_count],
                });
            }
        }

        if numa_nodes.is_empty() {
            return Some(Self::single_node());
        }

        // Fill distances
        for i in 0..numa_nodes.len() {
            for j in 0..numa_nodes.len() {
                numa_nodes[i].distances[j] = if i == j { 10 } else { 20 };
            }
        }

        let cache_levels = vec![
            CacheLevel {
                level: 1,
                size: 32 * 1024,
                shared_cores: vec![],
            },
            CacheLevel {
                level: 2,
                size: 256 * 1024,
                shared_cores: vec![],
            },
            CacheLevel {
                level: 3,
                size: 8 * 1024 * 1024,
                shared_cores: (0..logical_cores).collect(),
            },
        ];

        Some(CpuTopology {
            numa_nodes,
            core_to_node,
            logical_cores,
            cache_levels,
        })
    }

    /// Create a single-node topology for systems without NUMA.
    pub fn single_node() -> Self {
        let logical_cores = num_cpus::get();
        let cores: Vec<usize> = (0..logical_cores).collect();
        let mut core_to_node = HashMap::new();

        for &core in &cores {
            core_to_node.insert(core, 0);
        }

        let numa_nodes = vec![NumaNode {
            id: 0,
            cores,
            distances: vec![10], // Distance to self
        }];

        let cache_levels = vec![
            CacheLevel {
                level: 1,
                size: 32 * 1024,
                shared_cores: vec![],
            },
            CacheLevel {
                level: 2,
                size: 256 * 1024,
                shared_cores: vec![],
            },
            CacheLevel {
                level: 3,
                size: 8 * 1024 * 1024,
                shared_cores: (0..logical_cores).collect(),
            },
        ];

        Self {
            numa_nodes,
            core_to_node,
            logical_cores,
            cache_levels,
        }
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

    /// Get adjacent NUMA nodes (sorted by distance).
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
        if let Some(from) = self.numa_nodes.iter().find(|n| n.id == from_node) {
            if to_node < from.distances.len() {
                return from.distances[to_node];
            }
        }
        // Default distance if not found
        if from_node == to_node {
            10
        } else {
            20
        }
    }
}

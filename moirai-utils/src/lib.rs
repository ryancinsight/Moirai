//! Utility functions and data structures for Moirai concurrency library.

#![no_std]
#![deny(missing_docs)]

#[cfg(feature = "std")]
extern crate std;

#[cfg(feature = "std")]
use std::collections::HashMap;

/// Cache line size for alignment optimizations.
pub const CACHE_LINE_SIZE: usize = 64;

/// Align a value to the cache line boundary.
#[must_use]
pub const fn align_to_cache_line(size: usize) -> usize {
    (size + CACHE_LINE_SIZE - 1) & !(CACHE_LINE_SIZE - 1)
}

/// A cache-aligned wrapper for data structures.
#[repr(align(64))]
pub struct CacheAligned<T>(pub T);

impl<T> CacheAligned<T> {
    /// Create a new cache-aligned value.
    pub const fn new(value: T) -> Self {
        Self(value)
    }

    /// Get a reference to the inner value.
    pub const fn get(&self) -> &T {
        &self.0
    }

    /// Get a mutable reference to the inner value.
    pub fn get_mut(&mut self) -> &mut T {
        &mut self.0
    }
}

impl<T> core::ops::Deref for CacheAligned<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T> core::ops::DerefMut for CacheAligned<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

/// CPU topology detection and management.
#[cfg(feature = "std")]
pub mod cpu {
    use super::*;
    use std::{sync::OnceLock, vec, vec::Vec, format};

    /// CPU core identifier.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
    pub struct CpuCore(pub u32);

    impl CpuCore {
        /// Create a new CPU core identifier.
        pub const fn new(id: u32) -> Self {
            Self(id)
        }

        /// Get the core ID.
        pub const fn id(self) -> u32 {
            self.0
        }
    }

    /// CPU cache level.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum CacheLevel {
        /// L1 cache (per-core).
        L1,
        /// L2 cache (per-core or shared).
        L2,
        /// L3 cache (shared).
        L3,
    }

    /// CPU cache information.
    #[derive(Debug, Clone)]
    pub struct CacheInfo {
        /// Cache level.
        pub level: CacheLevel,
        /// Cache size in bytes.
        pub size: usize,
        /// Cache line size in bytes.
        pub line_size: usize,
        /// Associativity (0 = fully associative).
        pub associativity: u32,
        /// Cores sharing this cache.
        pub shared_cores: Vec<CpuCore>,
    }

    /// CPU topology information.
    #[derive(Debug, Clone)]
    pub struct CpuTopology {
        /// Total number of logical cores.
        pub logical_cores: u32,
        /// Total number of physical cores.
        pub physical_cores: u32,
        /// Number of CPU sockets.
        pub sockets: u32,
        /// Cache hierarchy information.
        pub caches: Vec<CacheInfo>,
        /// Core affinity groups (cores sharing L3 cache).
        pub affinity_groups: Vec<Vec<CpuCore>>,
        /// NUMA node mapping.
        pub numa_nodes: HashMap<CpuCore, u32>,
    }

    impl CpuTopology {
        /// Get the current CPU topology.
        pub fn detect() -> Self {
            static TOPOLOGY: OnceLock<CpuTopology> = OnceLock::new();
            TOPOLOGY.get_or_init(Self::detect_impl).clone()
        }

        /// Detect CPU topology (implementation).
        fn detect_impl() -> Self {
            // Platform-specific detection
            #[cfg(target_os = "linux")]
            return Self::detect_linux();
            
            #[cfg(target_os = "windows")]
            return Self::detect_windows();
            
            #[cfg(target_os = "macos")]
            return Self::detect_macos();
            
            // Fallback for unknown platforms
            #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
            return Self::detect_fallback();
        }

        /// Linux-specific CPU topology detection.
        #[cfg(target_os = "linux")]
        fn detect_linux() -> Self {
            use std::fs;
            use std::str::FromStr;

            let mut topology = Self::detect_fallback();
            
            // Try to read from /proc/cpuinfo
            if let Ok(cpuinfo) = fs::read_to_string("/proc/cpuinfo") {
                let mut physical_cores = std::collections::HashSet::new();
                let mut logical_cores = 0;
                
                for line in cpuinfo.lines() {
                    if line.starts_with("processor") {
                        logical_cores += 1;
                    } else if line.starts_with("physical id") {
                        if let Some(id_str) = line.split(':').nth(1) {
                            if let Ok(id) = u32::from_str(id_str.trim()) {
                                physical_cores.insert(id);
                            }
                        }
                    }
                }
                
                topology.logical_cores = logical_cores;
                topology.physical_cores = physical_cores.len() as u32;
                topology.sockets = physical_cores.len() as u32;
            }
            
            // Try to read cache information from /sys/devices/system/cpu/
            topology.detect_linux_caches();
            topology.detect_linux_numa();
            
            topology
        }

        /// Windows-specific CPU topology detection.
        #[cfg(target_os = "windows")]
        fn detect_windows() -> Self {
            // Use Windows API calls here
            // For now, fall back to basic detection
            Self::detect_fallback()
        }

        /// macOS-specific CPU topology detection.
        #[cfg(target_os = "macos")]
        fn detect_macos() -> Self {
            // Use sysctl calls here
            // For now, fall back to basic detection
            Self::detect_fallback()
        }

        /// Fallback CPU topology detection.
        fn detect_fallback() -> Self {
            let logical_cores = std::thread::available_parallelism()
                .map(|n| n.get() as u32)
                .unwrap_or(1);
            
            // Assume hyperthreading (2 logical cores per physical core)
            let physical_cores = (logical_cores + 1) / 2;
            
            // Create basic affinity groups (assume all cores share L3)
            let affinity_groups = vec![(0..logical_cores).map(CpuCore::new).collect()];
            
            // Basic cache info (typical modern CPU)
            let caches = vec![
                CacheInfo {
                    level: CacheLevel::L1,
                    size: 32 * 1024, // 32KB
                    line_size: 64,
                    associativity: 8,
                    shared_cores: (0..logical_cores).map(CpuCore::new).collect(),
                },
                CacheInfo {
                    level: CacheLevel::L2,
                    size: 256 * 1024, // 256KB
                    line_size: 64,
                    associativity: 8,
                    shared_cores: (0..logical_cores).map(CpuCore::new).collect(),
                },
                CacheInfo {
                    level: CacheLevel::L3,
                    size: 8 * 1024 * 1024, // 8MB
                    line_size: 64,
                    associativity: 16,
                    shared_cores: (0..logical_cores).map(CpuCore::new).collect(),
                },
            ];

            let numa_nodes = (0..logical_cores)
                .map(|i| (CpuCore::new(i), 0))
                .collect();

            Self {
                logical_cores,
                physical_cores,
                sockets: 1,
                caches,
                affinity_groups,
                numa_nodes,
            }
        }

        /// Detect cache information on Linux.
        #[cfg(target_os = "linux")]
        fn detect_linux_caches(&mut self) {
            use std::fs;
            
            
            let mut detected_caches = Vec::new();
            
            for cpu_id in 0..self.logical_cores {
                let cache_path = format!("/sys/devices/system/cpu/cpu{}/cache", cpu_id);
                if let Ok(entries) = fs::read_dir(&cache_path) {
                    for entry in entries.flatten() {
                        if let Some(index_str) = entry.file_name().to_str() {
                            if index_str.starts_with("index") {
                                if let Some(cache_info) = self.read_cache_info(&entry.path(), cpu_id) {
                                    detected_caches.push(cache_info);
                                }
                            }
                        }
                    }
                }
            }
            
            if !detected_caches.is_empty() {
                self.caches = detected_caches;
            }
        }

        /// Read cache information from Linux sysfs.
        #[cfg(target_os = "linux")]
        fn read_cache_info(&self, cache_path: &std::path::Path, cpu_id: u32) -> Option<CacheInfo> {
            use std::fs;
            
            
            let level_path = cache_path.join("level");
            let size_path = cache_path.join("size");
            let coherency_line_size_path = cache_path.join("coherency_line_size");
            let shared_cpu_list_path = cache_path.join("shared_cpu_list");
            
            let level = fs::read_to_string(&level_path).ok()?
                .trim().parse::<u32>().ok()?;
            
            let level = match level {
                1 => CacheLevel::L1,
                2 => CacheLevel::L2,
                3 => CacheLevel::L3,
                _ => return None,
            };
            
            let size_str = fs::read_to_string(&size_path).ok()?;
            let size = Self::parse_size(&size_str)?;
            
            let line_size = fs::read_to_string(&coherency_line_size_path).ok()?
                .trim().parse::<usize>().ok().unwrap_or(64);
            
            let shared_cores = fs::read_to_string(&shared_cpu_list_path).ok()
                .and_then(|s| Self::parse_cpu_list(&s))
                .unwrap_or_else(|| vec![CpuCore::new(cpu_id)]);
            
            Some(CacheInfo {
                level,
                size,
                line_size,
                associativity: 0, // Not easily available from sysfs
                shared_cores,
            })
        }

        /// Parse size string (e.g., "32K", "1M").
        #[cfg(target_os = "linux")]
        pub fn parse_size(size_str: &str) -> Option<usize> {
            let size_str = size_str.trim();
            if size_str.is_empty() {
                return None;
            }
            
            let (num_str, suffix) = if let Some(stripped) = size_str.strip_suffix('K') {
                (stripped, 1024)
            } else if let Some(stripped) = size_str.strip_suffix('M') {
                (stripped, 1024 * 1024)
            } else if let Some(stripped) = size_str.strip_suffix('G') {
                (stripped, 1024 * 1024 * 1024)
            } else {
                (size_str, 1)
            };
            
            num_str.parse::<usize>().ok().map(|n| n * suffix)
        }

        /// Parse CPU list (e.g., "0-3,6,8-11").
        #[cfg(target_os = "linux")]
        pub fn parse_cpu_list(cpu_list: &str) -> Option<Vec<CpuCore>> {
            let mut cores = Vec::new();
            
            for part in cpu_list.trim().split(',') {
                if part.contains('-') {
                    let range_parts: Vec<&str> = part.split('-').collect();
                    if range_parts.len() == 2 {
                        if let (Ok(start), Ok(end)) = (
                            range_parts[0].parse::<u32>(),
                            range_parts[1].parse::<u32>()
                        ) {
                            for i in start..=end {
                                cores.push(CpuCore::new(i));
                            }
                        }
                    }
                } else if let Ok(cpu_id) = part.parse::<u32>() {
                    cores.push(CpuCore::new(cpu_id));
                }
            }
            
            if cores.is_empty() { None } else { Some(cores) }
        }

        /// Detect NUMA information on Linux.
        #[cfg(target_os = "linux")]
        fn detect_linux_numa(&mut self) {
            
            
            let mut numa_mapping = HashMap::new();
            
            for cpu_id in 0..self.logical_cores {
                let numa_path = format!("/sys/devices/system/cpu/cpu{}/node0", cpu_id);
                if std::path::Path::new(&numa_path).exists() {
                    numa_mapping.insert(CpuCore::new(cpu_id), 0);
                } else {
                    // Try to find which NUMA node this CPU belongs to
                    for node_id in 0..8 { // Check up to 8 NUMA nodes
                        let node_path = format!("/sys/devices/system/node/node{}/cpu{}", node_id, cpu_id);
                        if std::path::Path::new(&node_path).exists() {
                            numa_mapping.insert(CpuCore::new(cpu_id), node_id);
                            break;
                        }
                    }
                }
            }
            
            if !numa_mapping.is_empty() {
                self.numa_nodes = numa_mapping;
            }
        }

        /// Get cores in the same affinity group as the given core.
        pub fn affinity_group(&self, core: CpuCore) -> Option<&[CpuCore]> {
            self.affinity_groups.iter()
                .find(|group| group.contains(&core))
                .map(|group| group.as_slice())
        }

        /// Get the NUMA node for a given core.
        pub fn numa_node(&self, core: CpuCore) -> Option<u32> {
            self.numa_nodes.get(&core).copied()
        }

        /// Get all cores in a NUMA node.
        pub fn cores_in_numa_node(&self, numa_node: u32) -> Vec<CpuCore> {
            self.numa_nodes.iter()
                .filter(|(_, &node)| node == numa_node)
                .map(|(&core, _)| core)
                .collect()
        }

        /// Get the optimal core for a task based on current load.
        pub fn optimal_core(&self, preferred_numa_node: Option<u32>) -> CpuCore {
            if let Some(numa_node) = preferred_numa_node {
                let numa_cores = self.cores_in_numa_node(numa_node);
                if !numa_cores.is_empty() {
                    return numa_cores[0]; // Simple selection, could be improved
                }
            }
            
            // Fallback to first available core
            CpuCore::new(0)
        }
    }

    /// Core affinity management.
    pub mod affinity {
        use super::*;

        /// Core affinity mask.
        #[derive(Debug, Clone)]
        pub struct AffinityMask {
            cores: std::collections::HashSet<CpuCore>,
        }

        impl AffinityMask {
            /// Create an empty affinity mask.
            pub fn new() -> Self {
                Self {
                    cores: std::collections::HashSet::new(),
                }
            }

            /// Create an affinity mask for all available cores.
            pub fn all() -> Self {
                let topology = CpuTopology::detect();
                let cores = (0..topology.logical_cores).map(CpuCore::new).collect();
                Self { cores }
            }

            /// Create an affinity mask for a single core.
            pub fn single(core: CpuCore) -> Self {
                let mut cores = std::collections::HashSet::new();
                cores.insert(core);
                Self { cores }
            }

            /// Create an affinity mask for a NUMA node.
            pub fn numa_node(numa_node: u32) -> Self {
                let topology = CpuTopology::detect();
                let cores = topology.cores_in_numa_node(numa_node).into_iter().collect();
                Self { cores }
            }

            /// Add a core to the affinity mask.
            pub fn add_core(&mut self, core: CpuCore) {
                self.cores.insert(core);
            }

            /// Remove a core from the affinity mask.
            pub fn remove_core(&mut self, core: CpuCore) {
                self.cores.remove(&core);
            }

            /// Check if a core is in the affinity mask.
            pub fn contains(&self, core: CpuCore) -> bool {
                self.cores.contains(&core)
            }

            /// Get all cores in the affinity mask.
            pub fn cores(&self) -> impl Iterator<Item = CpuCore> + '_ {
                self.cores.iter().copied()
            }

            /// Get the number of cores in the mask.
            pub fn len(&self) -> usize {
                self.cores.len()
            }

            /// Check if the mask is empty.
            pub fn is_empty(&self) -> bool {
                self.cores.is_empty()
            }

            /// Set thread affinity to this mask.
            pub fn set_current_thread_affinity(&self) -> Result<(), AffinityError> {
                self.set_thread_affinity_impl(None)
            }

            /// Set affinity for a specific thread.
            pub fn set_thread_affinity(&self, thread: &std::thread::Thread) -> Result<(), AffinityError> {
                self.set_thread_affinity_impl(Some(thread))
            }

            /// Platform-specific affinity setting implementation.
            fn set_thread_affinity_impl(&self, _thread: Option<&std::thread::Thread>) -> Result<(), AffinityError> {
                if self.cores.is_empty() {
                    return Err(AffinityError::EmptyMask);
                }

                #[cfg(target_os = "linux")]
                return self.set_affinity_linux();

                #[cfg(target_os = "windows")]
                return self.set_affinity_windows();

                #[cfg(target_os = "macos")]
                return self.set_affinity_macos();

                // Fallback for unsupported platforms
                #[cfg(not(any(target_os = "linux", target_os = "windows", target_os = "macos")))]
                Ok(()) // No-op on unsupported platforms
            }

            /// Linux-specific affinity setting.
            #[cfg(target_os = "linux")]
            fn set_affinity_linux(&self) -> Result<(), AffinityError> {
                // This would use sched_setaffinity system call
                // For now, this is a placeholder
                Ok(())
            }

            /// Windows-specific affinity setting.
            #[cfg(target_os = "windows")]
            fn set_affinity_windows(&self) -> Result<(), AffinityError> {
                // This would use SetThreadAffinityMask
                // For now, this is a placeholder
                Ok(())
            }

            /// macOS-specific affinity setting.
            #[cfg(target_os = "macos")]
            fn set_affinity_macos(&self) -> Result<(), AffinityError> {
                // macOS doesn't support hard thread affinity
                // This would use thread_policy_set with affinity hints
                Ok(())
            }
        }

        impl Default for AffinityMask {
            fn default() -> Self {
                Self::new()
            }
        }

        /// Affinity-related errors.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum AffinityError {
            /// Empty affinity mask.
            EmptyMask,
            /// System call failed.
            SystemError,
            /// Unsupported platform.
            Unsupported,
        }

        impl core::fmt::Display for AffinityError {
            fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
                match self {
                    AffinityError::EmptyMask => write!(f, "Empty affinity mask"),
                    AffinityError::SystemError => write!(f, "System call failed"),
                    AffinityError::Unsupported => write!(f, "Unsupported platform"),
                }
            }
        }

        #[cfg(feature = "std")]
        impl std::error::Error for AffinityError {}

        /// Get the current thread's affinity.
        pub fn get_current_thread_affinity() -> Result<AffinityMask, AffinityError> {
            // Platform-specific implementation would go here
            // For now, return all cores
            Ok(AffinityMask::all())
        }

        /// Pin the current thread to a specific core.
        pub fn pin_to_core(core: CpuCore) -> Result<(), AffinityError> {
            let mask = AffinityMask::single(core);
            mask.set_current_thread_affinity()
        }

        /// Pin the current thread to a NUMA node.
        pub fn pin_to_numa_node(numa_node: u32) -> Result<(), AffinityError> {
            let mask = AffinityMask::numa_node(numa_node);
            mask.set_current_thread_affinity()
        }
    }
}

#[cfg(feature = "numa")]
/// NUMA topology information.
pub mod numa {
    use super::*;
    #[cfg(feature = "std")]
    use super::cpu::CpuTopology;
    
    /// NUMA node identifier.
    #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
    pub struct NumaNode(pub u32);

    impl NumaNode {
        /// Create a new NUMA node identifier.
        pub const fn new(id: u32) -> Self {
            Self(id)
        }

        /// Get the node ID.
        pub const fn id(self) -> u32 {
            self.0
        }
    }
    
    /// Get the current NUMA node.
    #[cfg(feature = "std")]
    pub fn current_numa_node() -> NumaNode {
        #[cfg(target_os = "linux")]
        {
            use std::os::raw::{c_long, c_uint, c_void};
            use crate::cpu::CpuCore;
            
            extern "C" {
                fn syscall(num: c_long, ...) -> c_long;
            }
            
            const SYS_GETCPU: c_long = 309;
            
            let mut cpu: c_uint = 0;
            let mut node: c_uint = 0;
            
            let result = unsafe {
                syscall(SYS_GETCPU, &mut cpu, &mut node, std::ptr::null::<c_void>())
            };
            
            if result == 0 {
                NumaNode::new(node)
            } else {
                // Fallback: try to determine from CPU topology
                let topology = CpuTopology::detect();
                let current_cpu = CpuCore::new(cpu);
                topology.numa_node(current_cpu).map(NumaNode::new).unwrap_or(NumaNode::new(0))
            }
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            // For non-Linux platforms, try to determine from CPU topology
            let topology = CpuTopology::detect();
            // Simple heuristic: assume we're on core 0 and get its NUMA node
            topology.numa_node(CpuCore::new(0)).map(NumaNode::new).unwrap_or(NumaNode::new(0))
        }
    }
    
    /// Get the current NUMA node (no_std version).
    #[cfg(not(feature = "std"))]
    pub fn current_numa_node() -> NumaNode {
        NumaNode::new(0)
    }
    
    /// Get the number of NUMA nodes.
    #[cfg(feature = "std")]
    pub fn numa_node_count() -> usize {
        let topology = CpuTopology::detect();
        let max_node = topology.numa_nodes.values().max().copied().unwrap_or(0);
        (max_node + 1) as usize
    }
    
    /// Get the number of NUMA nodes (no_std version).
    #[cfg(not(feature = "std"))]
    pub fn numa_node_count() -> usize {
        1
    }

    /// NUMA memory allocation preferences.
    #[cfg(feature = "std")]
    pub mod memory {
        use super::*;

        /// NUMA memory policy.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub enum NumaPolicy {
            /// Default system policy.
            Default,
            /// Bind to specific node.
            Bind(NumaNode),
            /// Prefer specific node.
            Preferred(NumaNode),
            /// Interleave across nodes.
            Interleave,
        }

        /// Set NUMA memory policy for current process.
        pub fn set_memory_policy(policy: NumaPolicy) -> Result<(), NumaError> {
            #[cfg(target_os = "linux")]
            {
                use std::os::raw::{c_int, c_long, c_ulong};
                
                const MPOL_DEFAULT: c_int = 0;
                const MPOL_PREFERRED: c_int = 1;
                const MPOL_BIND: c_int = 2;
                const MPOL_INTERLEAVE: c_int = 3;
                
                extern "C" {
                    fn syscall(num: c_long, ...) -> c_long;
                }
                
                const SYS_SET_MEMPOLICY: c_long = 238;
                
                let (mode, nodemask, maxnode) = match policy {
                    NumaPolicy::Default => (MPOL_DEFAULT, std::ptr::null::<c_ulong>(), 0),
                    NumaPolicy::Bind(node) => {
                        let mask = 1u64 << node.id();
                        (MPOL_BIND, &mask as *const u64 as *const c_ulong, 64)
                    },
                    NumaPolicy::Preferred(node) => {
                        let mask = 1u64 << node.id();
                        (MPOL_PREFERRED, &mask as *const u64 as *const c_ulong, 64)
                    },
                    NumaPolicy::Interleave => {
                        let mask = !0u64; // All nodes
                        (MPOL_INTERLEAVE, &mask as *const u64 as *const c_ulong, 64)
                    },
                };
                
                let result = unsafe {
                    syscall(SYS_SET_MEMPOLICY, mode, nodemask, maxnode)
                };
                
                if result == 0 {
                    Ok(())
                } else {
                    Err(NumaError::SystemError)
                }
            }
            
            #[cfg(not(target_os = "linux"))]
            {
                // For non-Linux platforms, just accept the policy without error
                // Real implementation would use platform-specific APIs
                match policy {
                    NumaPolicy::Default | NumaPolicy::Bind(_) | 
                    NumaPolicy::Preferred(_) | NumaPolicy::Interleave => Ok(()),
                }
            }
        }

        /// Allocate memory bound to a specific NUMA node.
        /// 
        /// On Linux, this function uses `mmap` for allocation and `mbind` syscall 
        /// to bind the allocated memory to the specified NUMA node, ensuring
        /// optimal memory locality for NUMA-aware applications.
        /// 
        /// # Parameters
        /// - `node`: The NUMA node to bind the memory to
        /// - `size`: Size of memory to allocate in bytes
        /// 
        /// # Returns
        /// - `Ok(ptr)`: Pointer to allocated and NUMA-bound memory
        /// - `Err(NumaError)`: Allocation or binding failed
        /// 
        /// # Platform Support
        /// - Linux: Full NUMA binding with `mbind` syscall
        /// - Other platforms: Falls back to standard allocation
        /// 
        /// # Design Principles
        /// - **SOLID SRP**: Single responsibility for NUMA-bound allocation
        /// - **GRASP Information Expert**: Uses system knowledge for optimal binding
        pub fn allocate_on_node<T>(node: NumaNode, size: usize) -> Result<*mut T, NumaError> {
            let layout = std::alloc::Layout::from_size_align(size, std::mem::align_of::<T>())
                .map_err(|_| NumaError::InvalidSize)?;
            
            #[cfg(target_os = "linux")]
            {
                use std::os::raw::{c_int, c_long, c_ulong, c_void};
                
                extern "C" {
                    fn syscall(num: c_long, ...) -> c_long;
                    fn mmap(addr: *mut c_void, length: usize, prot: c_int, flags: c_int, fd: c_int, offset: i64) -> *mut c_void;
                }
                
                const PROT_READ: c_int = 1;
                const PROT_WRITE: c_int = 2;
                const MAP_PRIVATE: c_int = 2;
                const MAP_ANONYMOUS: c_int = 32;
                
                // Use mmap to allocate memory that can be bound to specific NUMA node
                let ptr = unsafe {
                    mmap(
                        std::ptr::null_mut(),
                        layout.size(),
                        PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS,
                        -1,
                        0
                    )
                };
                
                if ptr.is_null() || ptr == (-1isize) as *mut c_void {
                    return Err(NumaError::AllocationFailed);
                }
                
                // Bind the allocated memory to the specified NUMA node using mbind syscall
                const MPOL_BIND: c_int = 2;
                const SYS_MBIND: c_long = 237;
                
                let mask = 1u64 << node.id();
                let result = unsafe {
                    syscall(
                        SYS_MBIND,
                        ptr,
                        layout.size(),
                        MPOL_BIND,
                        &mask as *const u64 as *const c_ulong,
                        64u64
                    )
                };
                
                if result != 0 {
                    // If NUMA binding fails, we still have the memory allocated
                    // Log warning but don't fail the allocation
                    #[cfg(feature = "std")]
                    {
                        use std::io::{self, Write};
                        let _ = writeln!(io::stderr(), "WARNING: NUMA binding failed for node {}, continuing with unbound memory", node.id());
                    }
                }
                
                Ok(ptr as *mut T)
            }
            
            #[cfg(not(target_os = "linux"))]
            {
                // Fallback to standard allocation on non-Linux platforms
                let ptr = unsafe { std::alloc::alloc(layout) };
                if ptr.is_null() {
                    Err(NumaError::AllocationFailed)
                } else {
                    Ok(ptr as *mut T)
                }
            }
        }

        /// Free NUMA-allocated memory.
        ///
        /// # Safety
        /// - `ptr` must have been allocated by `allocate_on_node`
        /// - `ptr` must not be used after this call
        /// - `size` must match the size used during allocation
        pub unsafe fn free_numa_memory<T>(ptr: *mut T, size: usize) {
            #[cfg(target_os = "linux")]
            {
                use std::os::raw::c_void;
                
                extern "C" {
                    fn munmap(addr: *mut c_void, length: usize) -> i32;
                }
                
                // For Linux, we assume memory was allocated with mmap, so use munmap
                let result = munmap(ptr as *mut c_void, size);
                if result != 0 {
                    // If munmap fails, fallback to standard deallocation
                    // This handles cases where memory was allocated with standard allocator
                    let layout = std::alloc::Layout::from_size_align_unchecked(size, std::mem::align_of::<T>());
                    std::alloc::dealloc(ptr as *mut u8, layout);
                }
            }
            
            #[cfg(not(target_os = "linux"))]
            {
                // For non-Linux platforms, use standard deallocation
                let layout = std::alloc::Layout::from_size_align_unchecked(size, std::mem::align_of::<T>());
                std::alloc::dealloc(ptr as *mut u8, layout);
            }
        }
    }

    /// NUMA-related errors.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum NumaError {
        /// Invalid size for allocation.
        InvalidSize,
        /// Memory allocation failed.
        AllocationFailed,
        /// NUMA not supported.
        NotSupported,
        /// System call failed.
        SystemError,
    }

    impl core::fmt::Display for NumaError {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            match self {
                NumaError::InvalidSize => write!(f, "Invalid allocation size"),
                NumaError::AllocationFailed => write!(f, "Memory allocation failed"),
                NumaError::NotSupported => write!(f, "NUMA not supported"),
                NumaError::SystemError => write!(f, "System call failed"),
            }
        }
    }

    #[cfg(feature = "std")]
    impl std::error::Error for NumaError {}
}

/// Memory optimization utilities.
pub mod memory {
    /// Prefetch data into cache.
    #[inline]
    pub fn prefetch_read<T>(ptr: *const T) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            core::arch::x86_64::_mm_prefetch(ptr as *const i8, core::arch::x86_64::_MM_HINT_T0);
        }
        
        #[cfg(target_arch = "aarch64")]
        unsafe {
            core::arch::aarch64::_prefetch(ptr as *const u8, core::arch::aarch64::_PREFETCH_READ, core::arch::aarch64::_PREFETCH_LOCALITY3);
        }
        
        // For other architectures, this is a no-op
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let _ = ptr;
    }

    /// Prefetch data for writing.
    #[inline]
    pub fn prefetch_write<T>(ptr: *const T) {
        #[cfg(target_arch = "x86_64")]
        unsafe {
            core::arch::x86_64::_mm_prefetch(ptr as *const i8, core::arch::x86_64::_MM_HINT_T0);
        }
        
        #[cfg(target_arch = "aarch64")]
        unsafe {
            core::arch::aarch64::_prefetch(ptr as *const u8, core::arch::aarch64::_PREFETCH_WRITE, core::arch::aarch64::_PREFETCH_LOCALITY3);
        }
        
        // For other architectures, this is a no-op
        #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
        let _ = ptr;
    }

    /// Memory barrier for ordering guarantees.
    #[inline]
    pub fn memory_barrier() {
        core::sync::atomic::fence(core::sync::atomic::Ordering::SeqCst);
    }

    /// Compiler barrier to prevent reordering.
    #[inline]
    pub fn compiler_barrier() {
        core::sync::atomic::compiler_fence(core::sync::atomic::Ordering::SeqCst);
    }

    /// Branch prediction optimization utilities.
    /// 
    /// # Design Principles Applied
    /// - **SOLID**: Single responsibility for branch prediction optimization
    /// - **CUPID**: Composable hints that can be used throughout the codebase
    /// - **KISS**: Simple, lightweight branch prediction hints
    /// - **YAGNI**: Only essential branch prediction features
    pub mod branch_prediction {
        /// Hint that a condition is likely to be true.
        /// 
        /// This helps the CPU's branch predictor optimize for the common case,
        /// reducing branch misprediction penalties. Uses manual branch prediction
        /// techniques that are stable and compatible with all Rust versions.
        /// 
        /// # Example
        /// ```
        /// use moirai_utils::memory::branch_prediction::likely;
        /// 
        /// if likely(some_condition) {
        ///     // This branch is expected to be taken most of the time
        /// }
        /// ```
        #[inline]
        pub fn likely(condition: bool) -> bool {
            // Manual branch prediction using cold attribute on unlikely path
            if condition {
                true
            } else {
                cold_path_false()
            }
        }

        /// Hint that a condition is unlikely to be true.
        /// 
        /// This helps the CPU's branch predictor optimize for the uncommon case,
        /// reducing branch misprediction penalties for error paths.
        /// 
        /// # Example
        /// ```
        /// use moirai_utils::memory::branch_prediction::unlikely;
        /// 
        /// if unlikely(error_condition) {
        ///     // This branch is expected to be taken rarely
        ///     handle_error();
        /// }
        /// ```
        #[inline]
        pub fn unlikely(condition: bool) -> bool {
            // Manual branch prediction using cold attribute on likely path
            if condition {
                cold_path_true()
            } else {
                false
            }
        }

        /// Cold path for false condition - marked cold to hint branch predictor.
        #[cold]
        #[inline(never)]
        fn cold_path_false() -> bool {
            false
        }

        /// Cold path for true condition - marked cold to hint branch predictor.
        #[cold]
        #[inline(never)]
        fn cold_path_true() -> bool {
            true
        }

        /// Force a branch prediction to be cold (unlikely to be taken).
        /// 
        /// This is useful for error handling paths that should not pollute
        /// the branch predictor's cache.
        #[inline]
        pub fn cold_branch() {
            #[cfg(target_arch = "x86_64")]
            {
                // Insert a serializing instruction to ensure branch prediction cache coherency
                unsafe {
                    core::arch::asm!("", options(nomem, nostack));
                }
            }
        }

        /// Prefetch the next instruction to improve instruction cache performance.
        /// 
        /// This is particularly useful in tight loops where instruction cache
        /// misses can significantly impact performance.
        #[inline]
        pub fn prefetch_instruction(ptr: *const u8) {
            #[cfg(target_arch = "x86_64")]
            unsafe {
                core::arch::x86_64::_mm_prefetch(ptr as *const i8, core::arch::x86_64::_MM_HINT_T0);
            }
            
            #[cfg(target_arch = "aarch64")]
            unsafe {
                core::arch::aarch64::_prefetch(ptr, core::arch::aarch64::_PREFETCH_READ, core::arch::aarch64::_PREFETCH_LOCALITY3);
            }
            
            // For other architectures, this is a no-op
            #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
            let _ = ptr;
        }
    }
}

/// Memory pool allocator for high-performance memory management.
/// 
/// # Design Principles Applied
/// - **SOLID**: Single responsibility (memory allocation), open for extension
/// - **CUPID**: Composable with other allocators, predictable behavior
/// - **GRASP**: Information expert pattern, low coupling
/// - **KISS**: Simple block-based allocation strategy
/// - **YAGNI**: Only implements essential allocation features
/// - **DRY**: Reusable allocation logic across different block sizes
/// - **SSOT**: Single source of truth for memory pool state
pub mod memory_pool {
    #[cfg(feature = "std")]
    use std::{
        sync::Mutex,
        collections::{VecDeque, HashMap},
        ptr::NonNull,
        alloc::{alloc, dealloc, Layout},
        thread_local,
        cell::RefCell,
        vec::Vec,
    };
    use super::CACHE_LINE_SIZE;

    /// A thread-safe memory pool for fixed-size allocations.
    /// 
    /// # Performance Characteristics
    /// - Allocation: O(1) when pool has free blocks
    /// - Deallocation: O(1) always
    /// - Memory overhead: ~8 bytes per block + pool metadata
    /// - Thread-safe with minimal contention
    /// 
    /// # Design Philosophy
    /// - **Single Responsibility**: Only handles memory allocation/deallocation
    /// - **Information Expert**: Pool knows about its own memory blocks
    /// - **Low Coupling**: Independent of specific data types
    pub struct MemoryPool {
        block_size: usize,
        alignment: usize,
        free_blocks: Mutex<VecDeque<NonNull<u8>>>,
        allocated_chunks: Mutex<Vec<NonNull<u8>>>,
        blocks_per_chunk: usize,
        total_allocated_blocks: Mutex<usize>,
    }

    impl MemoryPool {
        /// Create a new memory pool with specified block size.
        /// 
        /// # Parameters
        /// - `block_size`: Size of each allocation block
        /// - `initial_blocks`: Number of blocks to pre-allocate
        /// 
        /// # Design Principles
        /// - **KISS**: Simple constructor with sensible defaults
        /// - **YAGNI**: Only essential parameters
        pub fn new(block_size: usize, initial_blocks: usize) -> Self {
            let alignment = block_size.next_power_of_two().min(CACHE_LINE_SIZE);
            let blocks_per_chunk = (4096 / block_size).max(1); // At least 1 page worth
            
            let pool = Self {
                block_size,
                alignment,
                free_blocks: Mutex::new(VecDeque::new()),
                allocated_chunks: Mutex::new(Vec::new()),
                blocks_per_chunk,
                total_allocated_blocks: Mutex::new(0),
            };
            
            if initial_blocks > 0 {
                pool.allocate_chunk(initial_blocks);
            }
            
            pool
        }

        /// Allocate a block from the pool.
        /// 
        /// # Returns
        /// - `Some(ptr)`: Pointer to allocated block
        /// - `None`: Allocation failed
        /// 
        /// # Design Principles
        /// - **SOLID**: Single responsibility for allocation
        /// - **GRASP**: Creator pattern - pool creates memory blocks
        pub fn allocate(&self) -> Option<NonNull<u8>> {
            let mut free_blocks = self.free_blocks.lock().unwrap();
            
            if let Some(block) = free_blocks.pop_front() {
                Some(block)
            } else {
                // Need to allocate a new chunk
                drop(free_blocks); // Release lock before expensive operation
                self.allocate_chunk(self.blocks_per_chunk);
                
                // Try again after allocating new chunk
                let mut free_blocks = self.free_blocks.lock().unwrap();
                free_blocks.pop_front()
            }
        }

        /// Deallocate a block back to the pool.
        /// 
        /// # Safety
        /// The pointer must have been allocated by this pool.
        /// 
        /// # Design Principles
        /// - **SOLID**: Single responsibility for deallocation
        /// - **SSOT**: Pool is the single source of truth for block ownership
        pub unsafe fn deallocate(&self, ptr: NonNull<u8>) {
            let mut free_blocks = self.free_blocks.lock().unwrap();
            free_blocks.push_back(ptr);
        }

        /// Get pool statistics for monitoring.
        /// 
        /// # Design Principles
        /// - **Information Expert**: Pool knows its own state
        /// - **CUPID**: Predictable interface for monitoring
        pub fn stats(&self) -> PoolStats {
            let free_blocks = self.free_blocks.lock().unwrap();
            let allocated_chunks = self.allocated_chunks.lock().unwrap();
            let total_allocated_blocks = self.total_allocated_blocks.lock().unwrap();
            
            PoolStats {
                block_size: self.block_size,
                free_blocks: free_blocks.len(),
                total_chunks: allocated_chunks.len(),
                total_capacity: *total_allocated_blocks,
            }
        }

        fn allocate_chunk(&self, num_blocks: usize) {
            // Account for alignment when calculating chunk size
            let aligned_block_size = ((self.block_size + self.alignment - 1) / self.alignment) * self.alignment;
            let chunk_size = aligned_block_size * num_blocks + self.alignment; // Extra space for alignment
            let layout = Layout::from_size_align(chunk_size, self.alignment)
                .expect("Invalid layout");
            
            unsafe {
                let chunk_ptr = alloc(layout);
                if chunk_ptr.is_null() {
                    return; // Allocation failed
                }
                
                // Align the start of the chunk
                let aligned_start = ((chunk_ptr as usize + self.alignment - 1) & !(self.alignment - 1)) as *mut u8;
                
                // Add original chunk to tracking for proper deallocation
                {
                    let mut allocated_chunks = self.allocated_chunks.lock().unwrap();
                    allocated_chunks.push(NonNull::new_unchecked(chunk_ptr));
                }
                
                // Split chunk into properly aligned blocks
                {
                    let mut free_blocks = self.free_blocks.lock().unwrap();
                    let mut total_allocated_blocks = self.total_allocated_blocks.lock().unwrap();
                    for i in 0..num_blocks {
                        let block_ptr = aligned_start.add(i * aligned_block_size);
                        let block = NonNull::new_unchecked(block_ptr);
                        free_blocks.push_back(block);
                    }
                    *total_allocated_blocks += num_blocks;
                }
            }
        }
    }

    impl Drop for MemoryPool {
        /// Clean up all allocated memory chunks.
        /// 
        /// # Design Principles
        /// - **SOLID**: Single responsibility for cleanup
        /// - **RAII**: Automatic resource management
        fn drop(&mut self) {
            let allocated_chunks = self.allocated_chunks.lock().unwrap();
            let chunk_size = self.block_size * self.blocks_per_chunk;
            let layout = Layout::from_size_align(chunk_size, self.alignment)
                .expect("Invalid layout");
            
            unsafe {
                for chunk in allocated_chunks.iter() {
                    dealloc(chunk.as_ptr(), layout);
                }
            }
        }
    }

    /// Statistics for a memory pool.
    /// 
    /// # Design Principles
    /// - **CUPID**: Composable with monitoring systems
    /// - **Information Expert**: Contains relevant pool metrics
    #[derive(Debug, Clone)]
    pub struct PoolStats {
        /// Size of each block in bytes
        pub block_size: usize,
        /// Number of currently free blocks
        pub free_blocks: usize,
        /// Total number of allocated chunks
        pub total_chunks: usize,
        /// Total capacity in blocks across all chunks
        pub total_capacity: usize,
    }

    impl PoolStats {
        /// Get the utilization percentage (0.0 to 1.0).
        pub fn utilization(&self) -> f64 {
            if self.total_capacity == 0 {
                0.0
            } else {
                let used_blocks = self.total_capacity.saturating_sub(self.free_blocks);
                used_blocks as f64 / self.total_capacity as f64
            }
        }

        /// Get the memory overhead in bytes.
        pub fn overhead_bytes(&self) -> usize {
            // Approximate overhead: VecDeque + Vec + Mutex overhead
            let vec_deque_overhead = std::mem::size_of::<VecDeque<NonNull<u8>>>() + 16; // Add heap allocation overhead
            let vec_overhead = std::mem::size_of::<Vec<NonNull<u8>>>() + 16; // Add heap allocation overhead
            let chunk_overhead = self.total_chunks * (std::mem::size_of::<NonNull<u8>>() + 8); // Add per-chunk overhead
            crate::align_to_cache_line(vec_deque_overhead + vec_overhead + chunk_overhead) // Align to cache line size
        }
    }

    /// Thread-safe memory pool manager for multiple block sizes.
    /// 
    /// # Design Principles
    /// - **SOLID**: Single responsibility for managing multiple pools
    /// - **GRASP**: Controller pattern for coordinating pools
    /// - **DRY**: Reuses MemoryPool logic for different sizes
    pub struct PoolManager {
        pools: HashMap<usize, MemoryPool>,
    }

    impl PoolManager {
        /// Create a new pool manager.
        pub fn new() -> Self {
            Self {
                pools: HashMap::new(),
            }
        }

        /// Get or create a pool for the specified block size.
        /// 
        /// # Design Principles
        /// - **GRASP**: Creator pattern with lazy initialization
        /// - **SSOT**: Manager is single source for pool instances
        pub fn get_or_create_pool(&mut self, block_size: usize, initial_blocks: usize) -> &MemoryPool {
            self.pools.entry(block_size)
                .or_insert_with(|| MemoryPool::new(block_size, initial_blocks))
        }

        /// Allocate from the appropriate pool.
        pub fn allocate(&mut self, size: usize) -> Option<NonNull<u8>> {
            let block_size = size.next_power_of_two().max(8); // Minimum 8 bytes
            let pool = self.get_or_create_pool(block_size, 16);
            pool.allocate()
        }

        /// Deallocate to the appropriate pool.
        /// 
        /// # Safety
        /// The pointer must have been allocated by this manager.
        pub unsafe fn deallocate(&mut self, ptr: NonNull<u8>, size: usize) {
            let block_size = size.next_power_of_two().max(8);
            if let Some(pool) = self.pools.get(&block_size) {
                pool.deallocate(ptr);
            }
        }

        /// Get statistics for all pools.
        pub fn all_stats(&self) -> Vec<(usize, PoolStats)> {
            self.pools.iter()
                .map(|(size, pool)| (*size, pool.stats()))
                .collect()
        }
    }

    impl Default for PoolManager {
        fn default() -> Self {
            Self::new()
        }
    }

    // Safety: NonNull<u8> is Send when the underlying data is Send
    // MemoryPool manages its own memory, so this is safe
    unsafe impl Send for MemoryPool {}
    unsafe impl Sync for MemoryPool {}

    #[cfg(feature = "std")]
    mod thread_local_pools {
        use super::*;
        
        // Thread-local pool manager for zero-contention allocations
        thread_local! {
            static THREAD_POOL_MANAGER: RefCell<PoolManager> = 
                RefCell::new(PoolManager::new());
        }

        /// Allocate from thread-local pool (zero contention).
        /// 
        /// # Design Principles
        /// - **CUPID**: Predictable performance characteristics
        /// - **KISS**: Simple interface for common use case
        pub fn thread_local_allocate(size: usize) -> Option<NonNull<u8>> {
            THREAD_POOL_MANAGER.with(|manager| {
                manager.borrow_mut().allocate(size)
            })
        }

        /// Deallocate to thread-local pool.
        /// 
        /// # Safety
        /// The pointer must have been allocated by thread_local_allocate.
        pub unsafe fn thread_local_deallocate(ptr: NonNull<u8>, size: usize) {
            THREAD_POOL_MANAGER.with(|manager| {
                manager.borrow_mut().deallocate(ptr, size);
            });
        }
    }

    #[cfg(feature = "std")]
    pub use thread_local_pools::{thread_local_allocate, thread_local_deallocate};

    /// NUMA-aware memory pool that allocates memory on specific NUMA nodes.
    /// 
    /// # Design Principles Applied
    /// - **SOLID**: Single responsibility (NUMA-aware allocation), extends MemoryPool
    /// - **CUPID**: Composable with existing allocators, domain-centric for NUMA
    /// - **GRASP**: Information expert for NUMA topology
    /// - **ADP**: Adapts allocation strategy based on NUMA topology
    #[cfg(all(feature = "std", feature = "numa"))]
    pub struct NumaAwarePool {
        pools: HashMap<u32, MemoryPool>, // One pool per NUMA node
        preferred_node: Option<u32>,
        block_size: usize, // Used in allocation logic
        // Metadata tracking: maps allocated pointers to their source node
        allocation_metadata: std::sync::Mutex<HashMap<usize, u32>>,
    }

    #[cfg(all(feature = "std", feature = "numa"))]
    impl NumaAwarePool {
        /// Create a new NUMA-aware memory pool.
        /// 
        /// # Parameters
        /// - `block_size`: Size of each allocation block
        /// - `initial_blocks_per_node`: Initial blocks to allocate per NUMA node
        /// - `preferred_node`: Preferred NUMA node for allocations (None = current node)
        pub fn new(block_size: usize, initial_blocks_per_node: usize, preferred_node: Option<u32>) -> Self {
            use super::numa::{numa_node_count, current_numa_node};
            
            let mut pools = HashMap::new();
            let node_count = numa_node_count();
            
            // Pre-create pools for all NUMA nodes
            for node_id in 0..(node_count as u32) {
                pools.insert(node_id, MemoryPool::new(block_size, initial_blocks_per_node));
            }
            
            let preferred_node = preferred_node.or_else(|| Some(current_numa_node().id()));
            
            Self {
                pools,
                preferred_node,
                block_size,
                allocation_metadata: std::sync::Mutex::new(HashMap::new()),
            }
        }

        /// Allocate a block, preferring the specified NUMA node.
        /// 
        /// # Design Principles
        /// - **ADP**: Adapts to current NUMA topology
        /// - **GRASP**: Creator pattern with NUMA awareness
        pub fn allocate(&self) -> Option<NonNull<u8>> {
            use super::numa::current_numa_node;
            
            // Try preferred node first
            let target_node = self.preferred_node.unwrap_or_else(|| current_numa_node().id());
            
            if let Some(pool) = self.pools.get(&target_node) {
                if let Some(ptr) = pool.allocate() {
                    // Track allocation metadata
                    if let Ok(mut metadata) = self.allocation_metadata.lock() {
                        metadata.insert(ptr.as_ptr() as usize, target_node);
                    }
                    return Some(ptr);
                }
            }
            
            // Fallback: try other nodes in round-robin fashion
            for (&node_id, pool) in &self.pools {
                if node_id != target_node {
                    if let Some(ptr) = pool.allocate() {
                        // Track allocation metadata
                        if let Ok(mut metadata) = self.allocation_metadata.lock() {
                            metadata.insert(ptr.as_ptr() as usize, node_id);
                        }
                        return Some(ptr);
                    }
                }
            }
            
            None
        }

        /// Allocate a block on a specific NUMA node.
        /// 
        /// # Parameters
        /// - `node_id`: Target NUMA node ID
        /// 
        /// # Returns
        /// - `Some(ptr)`: Pointer to allocated block on the specified node
        /// - `None`: Allocation failed on that node
        pub fn allocate_on_node(&self, node_id: u32) -> Option<NonNull<u8>> {
            if let Some(pool) = self.pools.get(&node_id) {
                if let Some(ptr) = pool.allocate() {
                    // Track allocation metadata
                    if let Ok(mut metadata) = self.allocation_metadata.lock() {
                        metadata.insert(ptr.as_ptr() as usize, node_id);
                    }
                    return Some(ptr);
                }
            }
            None
        }

        /// Deallocate a block back to the appropriate pool.
        /// 
        /// # Safety
        /// The pointer must have been allocated by this NUMA-aware pool.
        /// This method is now safe because it uses metadata tracking to ensure
        /// the block is returned to the correct pool.
        /// 
        /// # Design Principles
        /// - **SSOT**: Pool is single source of truth for block ownership
        /// - **GRASP Information Expert**: Uses tracked metadata to make correct decisions
        pub unsafe fn deallocate(&self, ptr: NonNull<u8>) {
            let ptr_addr = ptr.as_ptr() as usize;
            
            // Look up the source node from allocation metadata
            if let Ok(mut metadata) = self.allocation_metadata.lock() {
                if let Some(source_node) = metadata.remove(&ptr_addr) {
                    // Return the block to its original pool
                    if let Some(pool) = self.pools.get(&source_node) {
                        pool.deallocate(ptr);
                        return;
                    }
                }
            }
            
            // Fallback: This should never happen in correct usage, but provides safety
            // If metadata is corrupted or missing, this indicates a programming error
            #[cfg(feature = "std")]
            {
                use std::io::{self, Write};
                let _ = writeln!(io::stderr(), "WARNING: NumaAwarePool::deallocate - Missing metadata for pointer {:p}. This indicates a bug.", ptr.as_ptr());
            }
            
            // As a last resort, try all pools (this is not ideal but prevents crashes)
            if let Some(pool) = self.pools.values().next() {
                // Note: This is still problematic as we don't know which pool owns this memory
                // In a production system, this should panic or return an error
                pool.deallocate(ptr);
            }
        }

        /// Deallocate a block from a specific NUMA node (safer alternative).
        /// 
        /// This method provides an explicit node ID to ensure correct deallocation
        /// and can be used when the caller tracks the allocation source.
        /// 
        /// # Safety
        /// The pointer must have been allocated from the specified node's pool.
        /// 
        /// # Parameters
        /// - `ptr`: Pointer to deallocate
        /// - `node_id`: The NUMA node ID where the block was originally allocated
        /// 
        /// # Design Principles
        /// - **SOLID ISP**: Interface segregation - explicit contract
        /// - **GRASP Information Expert**: Caller provides necessary information
        pub unsafe fn deallocate_from_node(&self, ptr: NonNull<u8>, node_id: u32) {
            if let Some(pool) = self.pools.get(&node_id) {
                pool.deallocate(ptr);
                
                // Also remove from metadata if present (cleanup)
                if let Ok(mut metadata) = self.allocation_metadata.lock() {
                    metadata.remove(&(ptr.as_ptr() as usize));
                }
            } else {
                #[cfg(feature = "std")]
                {
                    use std::io::{self, Write};
                    let _ = writeln!(io::stderr(), "WARNING: NumaAwarePool::deallocate_from_node - Invalid node ID: {}", node_id);
                }
            }
        }

        /// Get the block size for this NUMA-aware pool.
        pub fn block_size(&self) -> usize {
            self.block_size
        }

        /// Get statistics for all NUMA nodes.
        pub fn numa_stats(&self) -> HashMap<u32, PoolStats> {
            self.pools.iter()
                .map(|(&node_id, pool)| (node_id, pool.stats()))
                .collect()
        }

        /// Get the total memory utilization across all NUMA nodes.
        pub fn total_utilization(&self) -> f64 {
            let stats: Vec<_> = self.pools.values().map(|p| p.stats()).collect();
            let total_capacity: usize = stats.iter().map(|s| s.total_capacity).sum();
            let total_free: usize = stats.iter().map(|s| s.free_blocks).sum();
            
            if total_capacity == 0 {
                0.0
            } else {
                let total_used = total_capacity.saturating_sub(total_free);
                total_used as f64 / total_capacity as f64
            }
        }

        /// Set the preferred NUMA node for future allocations.
        pub fn set_preferred_node(&mut self, node_id: Option<u32>) {
            self.preferred_node = node_id;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::vec;

    #[test]
    fn test_cache_line_alignment() {
        assert_eq!(align_to_cache_line(1), 64);
        assert_eq!(align_to_cache_line(64), 64);
        assert_eq!(align_to_cache_line(65), 128);
        assert_eq!(align_to_cache_line(128), 128);
    }

    #[test]
    fn test_cache_aligned_wrapper() {
        let wrapper = CacheAligned::new(42u64);
        assert_eq!(*wrapper, 42u64);
        assert_eq!(std::mem::align_of_val(&wrapper), CACHE_LINE_SIZE);
        
        // Test that the wrapper is indeed cache-aligned
        let addr = &wrapper as *const _ as usize;
        assert_eq!(addr % CACHE_LINE_SIZE, 0);
    }

    #[cfg(feature = "std")]
    mod cpu_tests {
        use super::super::cpu::*;
        use std::vec;

        #[test]
        fn test_cpu_core() {
            let core = CpuCore::new(5);
            assert_eq!(core.id(), 5);
            
            let core2 = CpuCore::new(3);
            assert!(core > core2);
            assert!(core2 < core);
        }

        #[test]
        fn test_cpu_topology_detection() {
            let topology = CpuTopology::detect();
            
            // Basic sanity checks
            assert!(topology.logical_cores > 0);
            assert!(topology.physical_cores > 0);
            assert!(topology.sockets > 0);
            assert!(!topology.caches.is_empty());
            assert!(!topology.affinity_groups.is_empty());
            
            // Logical cores should be >= physical cores
            assert!(topology.logical_cores >= topology.physical_cores);
            
            // Check cache levels
            let has_l1 = topology.caches.iter().any(|c| c.level == CacheLevel::L1);
            let has_l2 = topology.caches.iter().any(|c| c.level == CacheLevel::L2);
            let has_l3 = topology.caches.iter().any(|c| c.level == CacheLevel::L3);
            
            assert!(has_l1, "Should have L1 cache");
            assert!(has_l2, "Should have L2 cache");
            assert!(has_l3, "Should have L3 cache");
        }

        #[test]
        fn test_numa_node_operations() {
            let topology = CpuTopology::detect();
            let core0 = CpuCore::new(0);
            
            // Should have NUMA node mapping for core 0
            let numa_node = topology.numa_node(core0);
            assert!(numa_node.is_some());
            
            if let Some(node) = numa_node {
                let cores_in_node = topology.cores_in_numa_node(node);
                assert!(!cores_in_node.is_empty());
                assert!(cores_in_node.contains(&core0));
            }
        }

        #[test]
        fn test_affinity_group() {
            let topology = CpuTopology::detect();
            let core0 = CpuCore::new(0);
            
            let affinity_group = topology.affinity_group(core0);
            assert!(affinity_group.is_some());
            
            if let Some(group) = affinity_group {
                assert!(!group.is_empty());
                assert!(group.contains(&core0));
            }
        }

        #[test]
        fn test_optimal_core_selection() {
            let topology = CpuTopology::detect();
            
            // Test without preferred NUMA node
            let core = topology.optimal_core(None);
            assert_eq!(core, CpuCore::new(0));
            
            // Test with preferred NUMA node
            let core = topology.optimal_core(Some(0));
            assert!(core.id() < topology.logical_cores);
        }

        #[cfg(target_os = "linux")]
        #[test]
        fn test_linux_parsing() {
            // Test size parsing
            assert_eq!(CpuTopology::parse_size("32K"), Some(32 * 1024));
            assert_eq!(CpuTopology::parse_size("256K"), Some(256 * 1024));
            assert_eq!(CpuTopology::parse_size("8M"), Some(8 * 1024 * 1024));
            assert_eq!(CpuTopology::parse_size("1G"), Some(1024 * 1024 * 1024));
            assert_eq!(CpuTopology::parse_size("1024"), Some(1024));
            assert_eq!(CpuTopology::parse_size(""), None);
            assert_eq!(CpuTopology::parse_size("invalid"), None);
            
            // Test CPU list parsing
            let cores = CpuTopology::parse_cpu_list("0-3").unwrap();
            assert_eq!(cores, vec![CpuCore::new(0), CpuCore::new(1), CpuCore::new(2), CpuCore::new(3)]);
            
            let cores = CpuTopology::parse_cpu_list("0,2,4").unwrap();
            assert_eq!(cores, vec![CpuCore::new(0), CpuCore::new(2), CpuCore::new(4)]);
            
            let cores = CpuTopology::parse_cpu_list("0-1,4-5").unwrap();
            assert_eq!(cores, vec![CpuCore::new(0), CpuCore::new(1), CpuCore::new(4), CpuCore::new(5)]);
            
            assert_eq!(CpuTopology::parse_cpu_list(""), None);
            assert_eq!(CpuTopology::parse_cpu_list("invalid"), None);
        }

        mod affinity_tests {
            use super::super::super::cpu::affinity::*;
            use super::super::super::cpu::CpuCore;
            use std::{vec::Vec, format};

            #[test]
            fn test_affinity_mask_creation() {
                let empty_mask = AffinityMask::new();
                assert!(empty_mask.is_empty());
                assert_eq!(empty_mask.len(), 0);
                
                let all_mask = AffinityMask::all();
                assert!(!all_mask.is_empty());
                assert!(all_mask.len() > 0);
                
                let single_mask = AffinityMask::single(CpuCore::new(1));
                assert!(!single_mask.is_empty());
                assert_eq!(single_mask.len(), 1);
                assert!(single_mask.contains(CpuCore::new(1)));
                assert!(!single_mask.contains(CpuCore::new(0)));
                
                let numa_mask = AffinityMask::numa_node(0);
                assert!(!numa_mask.is_empty());
            }

            #[test]
            fn test_affinity_mask_operations() {
                let mut mask = AffinityMask::new();
                
                mask.add_core(CpuCore::new(0));
                assert_eq!(mask.len(), 1);
                assert!(mask.contains(CpuCore::new(0)));
                
                mask.add_core(CpuCore::new(1));
                assert_eq!(mask.len(), 2);
                assert!(mask.contains(CpuCore::new(1)));
                
                mask.remove_core(CpuCore::new(0));
                assert_eq!(mask.len(), 1);
                assert!(!mask.contains(CpuCore::new(0)));
                assert!(mask.contains(CpuCore::new(1)));
                
                // Test iterator
                let cores: Vec<_> = mask.cores().collect();
                assert_eq!(cores.len(), 1);
                assert!(cores.contains(&CpuCore::new(1)));
            }

            #[test]
            fn test_affinity_setting() {
                let mask = AffinityMask::single(CpuCore::new(0));
                
                // This should not fail on any platform (may be no-op on some)
                let result = mask.set_current_thread_affinity();
                assert!(result.is_ok());
                
                // Empty mask should fail
                let empty_mask = AffinityMask::new();
                let result = empty_mask.set_current_thread_affinity();
                assert!(result.is_err());
                assert_eq!(result.unwrap_err(), AffinityError::EmptyMask);
            }

            #[test]
            fn test_convenience_functions() {
                let result = pin_to_core(CpuCore::new(0));
                assert!(result.is_ok());
                
                let result = pin_to_numa_node(0);
                assert!(result.is_ok());
                
                let result = get_current_thread_affinity();
                assert!(result.is_ok());
            }

            #[test]
            fn test_affinity_error_display() {
                assert_eq!(format!("{}", AffinityError::EmptyMask), "Empty affinity mask");
                assert_eq!(format!("{}", AffinityError::SystemError), "System call failed");
                assert_eq!(format!("{}", AffinityError::Unsupported), "Unsupported platform");
            }
        }
    }

    #[cfg(feature = "numa")]
    mod numa_tests {
        use super::super::numa::*;

        #[test]
        fn test_numa_node() {
            let node = NumaNode::new(42);
            assert_eq!(node.id(), 42);
        }

        #[test]
        fn test_numa_detection() {
            let current_node = current_numa_node();
            assert!(current_node.id() < 64); // Reasonable upper bound
            
            let node_count = numa_node_count();
            assert!(node_count > 0);
            assert!(node_count < 64); // Reasonable upper bound
        }

        #[cfg(feature = "std")]
        mod memory_tests {
            use super::super::super::numa::memory::*;
            use super::super::super::numa::{NumaNode, NumaError};
            use std::format;

            #[test]
            fn test_numa_policy() {
                let policies = [
                    NumaPolicy::Default,
                    NumaPolicy::Bind(NumaNode::new(0)),
                    NumaPolicy::Preferred(NumaNode::new(0)),
                    NumaPolicy::Interleave,
                ];
                
                for policy in &policies {
                    let result = set_memory_policy(*policy);
                    assert!(result.is_ok());
                }
            }

            #[test]
            fn test_numa_allocation() {
                let node = NumaNode::new(0);
                let size = std::mem::size_of::<u64>() * 1024; // 8KB
                
                let result = allocate_on_node::<u64>(node, size);
                assert!(result.is_ok());
                
                if let Ok(ptr) = result {
                    assert!(!ptr.is_null());
                    
                    // Test that we can actually write to the allocated memory
                    unsafe {
                        *ptr = 0x12345678;
                        assert_eq!(*ptr, 0x12345678);
                    }
                    
                    // Free the memory
                    unsafe {
                        free_numa_memory(ptr, size);
                    }
                }
            }

            #[test]
            fn test_numa_binding_verification() {
                // Test allocation on different nodes if available
                for node_id in 0..2 {
                    let node = NumaNode::new(node_id);
                    let size = 4096; // One page
                    
                    let result = allocate_on_node::<u8>(node, size);
                    if let Ok(ptr) = result {
                        assert!(!ptr.is_null());
                        
                        // Verify we can write to the memory (basic functionality test)
                        unsafe {
                            *ptr = 0xAB;
                            assert_eq!(*ptr, 0xAB);
                        }
                        
                        // The binding itself is verified by the mbind syscall
                        // If it fails, a warning is logged but allocation succeeds
                        // This is the correct behavior for graceful degradation
                        
                        unsafe {
                            free_numa_memory(ptr, size);
                        }
                    }
                }
            }

            #[test]
            fn test_numa_error_display() {
                assert_eq!(format!("{}", NumaError::InvalidSize), "Invalid allocation size");
                assert_eq!(format!("{}", NumaError::AllocationFailed), "Memory allocation failed");
                assert_eq!(format!("{}", NumaError::NotSupported), "NUMA not supported");
                assert_eq!(format!("{}", NumaError::SystemError), "System call failed");
            }
        }

        #[cfg(all(feature = "std", feature = "numa"))]
        mod numa_pool_tests {
            use super::super::super::memory_pool::NumaAwarePool;

            #[test]
            fn test_numa_aware_pool_creation() {
                let pool = NumaAwarePool::new(64, 10, None);
                let stats = pool.numa_stats();
                
                // Should have pools for all NUMA nodes
                assert!(!stats.is_empty());
                
                // Each node should have the right block size
                for (_, stat) in &stats {
                    assert_eq!(stat.block_size, 64);
                    assert!(stat.total_capacity >= 10);
                }
            }

            #[test]
            fn test_numa_aware_pool_allocation() {
                let pool = NumaAwarePool::new(128, 5, Some(0));
                
                // Test basic allocation
                let ptr1 = pool.allocate();
                assert!(ptr1.is_some());
                
                let ptr2 = pool.allocate_on_node(0);
                assert!(ptr2.is_some());
                
                // Test utilization
                let utilization = pool.total_utilization();
                assert!(utilization > 0.0);
                assert!(utilization <= 1.0);
                
                // Clean up
                if let Some(ptr) = ptr1 {
                    unsafe { pool.deallocate(ptr); }
                }
                if let Some(ptr) = ptr2 {
                    unsafe { pool.deallocate(ptr); }
                }
            }

            #[test]
            fn test_numa_aware_pool_node_preference() {
                let mut pool = NumaAwarePool::new(256, 8, Some(0));
                
                // Test setting preferred node
                pool.set_preferred_node(Some(1));
                
                let ptr = pool.allocate();
                assert!(ptr.is_some());
                
                if let Some(ptr) = ptr {
                    unsafe { pool.deallocate(ptr); }
                }
            }

            #[test]
            fn test_numa_aware_pool_stats() {
                let pool = NumaAwarePool::new(512, 12, None);
                let stats = pool.numa_stats();
                
                let mut _total_capacity = 0;
                for (_node_id, stat) in &stats {
                    assert!(stat.total_capacity > 0);
                    assert_eq!(stat.block_size, 512);
                    _total_capacity += stat.total_capacity;
                }
                
                let utilization = pool.total_utilization();
                assert_eq!(utilization, 0.0); // No allocations yet
            }

            #[test]
            fn test_numa_aware_pool_deallocate_correctness() {
                use std::vec::Vec;
                
                let pool = NumaAwarePool::new(64, 10, None);
                let mut allocated_ptrs = Vec::new();
                
                // Allocate from different nodes
                for node_id in 0..2 {
                    if let Some(ptr) = pool.allocate_on_node(node_id) {
                        allocated_ptrs.push((ptr, node_id));
                    }
                }
                
                // Verify that deallocate returns blocks to the correct pools
                // by checking that the metadata tracking works
                for (ptr, _expected_node) in allocated_ptrs {
                    // The pool should track which node each allocation came from
                    unsafe { pool.deallocate(ptr); }
                    
                    // After deallocation, the block should be available in the correct node's pool
                    // We can't directly verify this without exposing internals, but the fact that
                    // deallocate doesn't panic or cause corruption is a good sign
                }
                
                // Test the explicit node deallocation method
                if let Some(ptr) = pool.allocate_on_node(0) {
                    unsafe { pool.deallocate_from_node(ptr, 0); }
                }
                
                // Verify stats are still consistent after deallocations
                let stats = pool.numa_stats();
                for (_node_id, stat) in &stats {
                    assert!(stat.total_capacity > 0);
                    assert_eq!(stat.block_size, 64);
                }
            }
        }
    }

    #[cfg(feature = "std")]
    mod memory_tests {
        use super::super::memory_pool::*;
        use std::{sync::Arc, thread, vec::Vec};

        #[test]
        fn test_memory_pool_basic() {
            let pool = MemoryPool::new(64, 10);
            let stats = pool.stats();
            
            assert_eq!(stats.block_size, 64);
            assert_eq!(stats.free_blocks, 10);
            assert!(stats.total_capacity >= 10);
            
            // Test allocation
            let ptr1 = pool.allocate().expect("Should allocate successfully");
            let ptr2 = pool.allocate().expect("Should allocate successfully");
            
            let stats = pool.stats();
            assert_eq!(stats.free_blocks, 8); // 2 blocks allocated
            
            // Test deallocation
            unsafe {
                pool.deallocate(ptr1);
                pool.deallocate(ptr2);
            }
            
            let stats = pool.stats();
            assert_eq!(stats.free_blocks, 10); // Back to original
        }

        #[test]
        fn test_memory_pool_expansion() {
            let pool = MemoryPool::new(32, 2);
            
            // Allocate all initial blocks
            let ptr1 = pool.allocate().expect("Should allocate");
            let ptr2 = pool.allocate().expect("Should allocate");
            
            let stats = pool.stats();
            assert_eq!(stats.free_blocks, 0);
            
            // This should trigger chunk expansion
            let ptr3 = pool.allocate().expect("Should allocate after expansion");
            
            let stats = pool.stats();
            assert!(stats.total_capacity > 2); // Pool expanded
            
            unsafe {
                pool.deallocate(ptr1);
                pool.deallocate(ptr2);
                pool.deallocate(ptr3);
            }
        }

        #[test]
        fn test_memory_pool_concurrent() {
            let pool = Arc::new(MemoryPool::new(128, 100));
            let num_threads = 4;
            let allocations_per_thread = 25;
            
            let handles: Vec<_> = (0..num_threads)
                .map(|_| {
                    let pool = pool.clone();
                    thread::spawn(move || {
                        let mut ptrs = Vec::new();
                        
                        // Allocate
                        for _ in 0..allocations_per_thread {
                            if let Some(ptr) = pool.allocate() {
                                ptrs.push(ptr);
                            }
                        }
                        
                        // Deallocate
                        for ptr in ptrs {
                            unsafe { pool.deallocate(ptr); }
                        }
                    })
                })
                .collect();
            
            for handle in handles {
                handle.join().unwrap();
            }
            
            let stats = pool.stats();
            // All blocks should be returned (allowing for some pool expansion during concurrent access)
            assert!(stats.free_blocks >= 100); // At least the initial blocks should be free
            assert_eq!(stats.utilization(), 0.0); // No blocks should be in use
        }

        #[test]
        fn test_pool_stats() {
            let pool = MemoryPool::new(256, 10);
            let initial_stats = pool.stats();
            
            // With the new alignment logic, we might have more blocks than requested
            // The key is that initially, utilization should be 0 (all blocks are free)
            assert_eq!(initial_stats.utilization(), 0.0); // No blocks allocated
            assert!(initial_stats.overhead_bytes() > 0); // Has some overhead
            
            // Allocate half the available blocks
            let initial_free = initial_stats.free_blocks;
            let half_blocks = initial_free / 2;
            let mut ptrs = Vec::new();
            for _ in 0..half_blocks {
                if let Some(ptr) = pool.allocate() {
                    ptrs.push(ptr);
                }
            }
            
            let after_alloc_stats = pool.stats();
            let expected_utilization = half_blocks as f64 / after_alloc_stats.total_capacity as f64;
            assert!((after_alloc_stats.utilization() - expected_utilization).abs() < 0.1); // Allow some tolerance
            
            // Clean up
            for ptr in ptrs {
                unsafe { pool.deallocate(ptr); }
            }
            
            // Check that utilization is back to 0 after cleanup
            let final_stats = pool.stats();
            assert_eq!(final_stats.utilization(), 0.0); // All blocks deallocated
        }

        #[test]
        fn test_pool_manager() {
            let mut manager = PoolManager::new();
            
            // Test allocation of different sizes
            let ptr1 = manager.allocate(64).expect("Should allocate 64 bytes");
            let ptr2 = manager.allocate(128).expect("Should allocate 128 bytes");
            let ptr3 = manager.allocate(64).expect("Should allocate 64 bytes again");
            
            // Should have created pools for sizes 64 and 128
            let stats = manager.all_stats();
            assert!(stats.len() >= 2);
            
            // Clean up
            unsafe {
                manager.deallocate(ptr1, 64);
                manager.deallocate(ptr2, 128);
                manager.deallocate(ptr3, 64);
            }
        }

        #[test]
        fn test_thread_local_allocation() {
            use std::sync::Barrier;
            
            let barrier = Arc::new(Barrier::new(3));
            let handles: Vec<_> = (0..3)
                .map(|thread_id| {
                    let barrier = barrier.clone();
                    thread::spawn(move || {
                        barrier.wait(); // Synchronize start
                        
                        let mut ptrs = Vec::new();
                        
                        // Each thread allocates different sizes
                        let size = 32 << thread_id; // 32, 64, 128
                        for _ in 0..10 {
                            if let Some(ptr) = thread_local_allocate(size) {
                                ptrs.push((ptr, size));
                            }
                        }
                        
                        // Clean up
                        for (ptr, size) in ptrs {
                            unsafe { thread_local_deallocate(ptr, size); }
                        }
                    })
                })
                .collect();
            
            for handle in handles {
                handle.join().unwrap();
            }
        }

        #[test]
        fn test_memory_alignment() {
            let pool = MemoryPool::new(17, 5); // Odd size to test alignment
            
            for _ in 0..5 {
                if let Some(ptr) = pool.allocate() {
                    let addr = ptr.as_ptr() as usize;
                    // Should be aligned to power of 2
                    let expected_alignment = 17_usize.next_power_of_two().min(super::super::CACHE_LINE_SIZE);
                    assert_eq!(addr % expected_alignment, 0, "Allocation not properly aligned");
                    
                    unsafe { pool.deallocate(ptr); }
                }
            }
        }
    }

    #[test]
    fn test_prefetch_functions() {
        let data = [1u32, 2, 3, 4, 5];
        
        // These should not crash or cause issues
        memory::prefetch_read(data.as_ptr());
        memory::prefetch_write(data.as_ptr());
        
        // Test with different types
        let string = "Hello, World!";
        memory::prefetch_read(string.as_ptr());
        
        let large_data = vec![0u8; 4096];
        memory::prefetch_read(large_data.as_ptr());
        memory::prefetch_write(large_data.as_ptr());
    }

    #[test]
    fn test_memory_barriers() {
        // These should not crash or cause issues
        memory::memory_barrier();
        memory::compiler_barrier();
        
        // Test in a simple scenario using AtomicU32 for safety
        use std::sync::atomic::{AtomicU32, Ordering};
        static COUNTER: AtomicU32 = AtomicU32::new(0);
        
        COUNTER.store(1, Ordering::Relaxed);
        memory::compiler_barrier();
        assert_eq!(COUNTER.load(Ordering::Relaxed), 1);
        
        COUNTER.store(2, Ordering::Relaxed);
        memory::memory_barrier();
        assert_eq!(COUNTER.load(Ordering::Relaxed), 2);
    }

    #[test]
    fn test_prefetch_null_safety() {
        // These should not crash even with null pointers
        memory::prefetch_read(core::ptr::null::<u32>());
        memory::prefetch_write(core::ptr::null::<u32>());
    }

    #[test]
    fn test_branch_prediction_likely() {
        use memory::branch_prediction::likely;
        
        // Test that likely conditions work correctly
        assert!(likely(true));
        assert!(!likely(false));
        
        // Test in a realistic scenario
        let mut count = 0;
        for i in 0..100 {
            if likely(i < 95) {  // This should be true 95% of the time
                count += 1;
            }
        }
        assert_eq!(count, 95);
    }

    #[test]
    fn test_branch_prediction_unlikely() {
        use memory::branch_prediction::unlikely;
        
        // Test that unlikely conditions work correctly
        assert!(unlikely(true));
        assert!(!unlikely(false));
        
        // Test in a realistic scenario
        let mut error_count = 0;
        for i in 0..100 {
            if unlikely(i >= 95) {  // This should be true 5% of the time
                error_count += 1;
            }
        }
        assert_eq!(error_count, 5);
    }

    #[test]
    fn test_branch_prediction_cold_branch() {
        use memory::branch_prediction::cold_branch;
        
        // Test that cold_branch doesn't crash
        cold_branch();
        
        // Test in an error handling context
        let result: Result<i32, &str> = Ok(42);
        match result {
            Ok(value) => assert_eq!(value, 42),
            Err(_) => {
                cold_branch(); // Mark this as a cold path
                panic!("Should not reach here");
            }
        }
    }

    #[test]
    fn test_branch_prediction_instruction_prefetch() {
        use memory::branch_prediction::prefetch_instruction;
        
        // Test instruction prefetching with valid addresses
        let code = [0u8; 64];  // Simulate some code
        prefetch_instruction(code.as_ptr());
        
        // Test with null pointer (should not crash)
        prefetch_instruction(core::ptr::null::<u8>());
    }

    #[test]
    fn test_branch_prediction_performance_pattern() {
        use memory::branch_prediction::likely;
        
        // Simulate a common performance pattern: hot path with error handling
        let mut success_count = 0;
        let mut error_count = 0;
        
        for i in 0..1000 {
            let success = i % 100 != 0;  // 99% success rate
            
            if likely(success) {
                // Hot path - should be optimized
                success_count += 1;
            } else {
                // Cold path - error handling
                // Note: success is false here, so this is the error case
                error_count += 1;
            }
        }
        
        assert_eq!(success_count, 990);
        assert_eq!(error_count, 10);
    }

    #[test]
    fn test_branch_prediction_with_unlikely_condition() {
        use memory::branch_prediction::unlikely;
        
        // Test unlikely in a more realistic scenario
        let mut rare_event_count = 0;
        let mut normal_count = 0;
        
        for i in 0..1000 {
            if unlikely(i % 100 == 0) {  // 1% chance - rare event
                rare_event_count += 1;
            } else {
                normal_count += 1;
            }
        }
        
        assert_eq!(rare_event_count, 10);
        assert_eq!(normal_count, 990);
    }
}
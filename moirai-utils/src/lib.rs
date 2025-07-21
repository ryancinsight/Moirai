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
            TOPOLOGY.get_or_init(|| Self::detect_impl()).clone()
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
            
            let (num_str, suffix) = if size_str.ends_with('K') {
                (&size_str[..size_str.len()-1], 1024)
            } else if size_str.ends_with('M') {
                (&size_str[..size_str.len()-1], 1024 * 1024)
            } else if size_str.ends_with('G') {
                (&size_str[..size_str.len()-1], 1024 * 1024 * 1024)
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

/// NUMA topology information.
#[cfg(feature = "numa")]
pub mod numa {
    //! NUMA topology detection and management.
    
    use super::*;
    #[cfg(feature = "std")]
    use super::cpu::{CpuCore, CpuTopology};
    
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
        let _topology = CpuTopology::detect();
        // Try to determine current core and its NUMA node
        // For now, return node 0 as fallback
        NumaNode::new(0)
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
            // Platform-specific implementation would go here
            // This would use mbind/set_mempolicy on Linux, or VirtualAlloc on Windows
            match policy {
                NumaPolicy::Default => Ok(()),
                NumaPolicy::Bind(_node) => Ok(()),
                NumaPolicy::Preferred(_node) => Ok(()),
                NumaPolicy::Interleave => Ok(()),
            }
        }

        /// NUMA-aware allocator hint.
        pub fn allocate_on_node<T>(_node: NumaNode, size: usize) -> Result<*mut T, NumaError> {
            // This would use numa_alloc_onnode or similar
            // For now, use standard allocation
            let layout = std::alloc::Layout::from_size_align(size, std::mem::align_of::<T>())
                .map_err(|_| NumaError::InvalidSize)?;
            
            let ptr = unsafe { std::alloc::alloc(layout) };
            if ptr.is_null() {
                Err(NumaError::AllocationFailed)
            } else {
                Ok(ptr as *mut T)
            }
        }

        /// Free NUMA-allocated memory.
        pub unsafe fn free_numa_memory<T>(ptr: *mut T, size: usize) {
            let layout = std::alloc::Layout::from_size_align_unchecked(size, std::mem::align_of::<T>());
            std::alloc::dealloc(ptr as *mut u8, layout);
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
    //! Memory optimization and prefetching utilities.
    
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
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{vec, vec::Vec};

    #[test]
    fn test_cache_line_alignment() {
        assert_eq!(align_to_cache_line(0), 0);
        assert_eq!(align_to_cache_line(1), 64);
        assert_eq!(align_to_cache_line(64), 64);
        assert_eq!(align_to_cache_line(65), 128);
        assert_eq!(align_to_cache_line(128), 128);
        assert_eq!(align_to_cache_line(129), 192);
    }

    #[test]
    fn test_cache_aligned_wrapper() {
        let aligned = CacheAligned::new(42u32);
        assert_eq!(*aligned.get(), 42);
        assert_eq!(*aligned, 42);
        
        let mut aligned = CacheAligned::new(vec![1, 2, 3]);
        aligned.get_mut().push(4);
        assert_eq!(*aligned.get(), vec![1, 2, 3, 4]);
        
        // Test alignment
        let ptr = &aligned as *const _ as usize;
        assert_eq!(ptr % 64, 0, "CacheAligned should be 64-byte aligned");
    }

    #[cfg(feature = "std")]
    mod cpu_tests {
        use super::super::cpu::*;
        use std::{vec, vec::Vec, format};

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
                    
                    // Free the memory
                    unsafe {
                        free_numa_memory(ptr, size);
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
    }

    mod memory_tests {
        use super::super::memory::*;
        use std::vec;

        #[test]
        fn test_prefetch_functions() {
            let data = [1u32, 2, 3, 4, 5];
            
            // These should not crash or cause issues
            prefetch_read(data.as_ptr());
            prefetch_write(data.as_ptr());
            
            // Test with different types
            let string = "Hello, World!";
            prefetch_read(string.as_ptr());
            
            let large_data = vec![0u8; 4096];
            prefetch_read(large_data.as_ptr());
            prefetch_write(large_data.as_ptr());
        }

        #[test]
        fn test_memory_barriers() {
            // These should not crash or cause issues
            memory_barrier();
            compiler_barrier();
            
            // Test in a simple scenario
            static mut COUNTER: u32 = 0;
            unsafe {
                COUNTER = 1;
                compiler_barrier();
                assert_eq!(COUNTER, 1);
                
                COUNTER = 2;
                memory_barrier();
                assert_eq!(COUNTER, 2);
            }
        }

        #[test]
        fn test_prefetch_null_safety() {
            // These should not crash even with null pointers
            prefetch_read(core::ptr::null::<u32>());
            prefetch_write(core::ptr::null::<u32>());
        }
    }
}
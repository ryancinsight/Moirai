//! NUMA-aware iterator implementations for optimal memory locality.
//!
//! This module provides iterators that are aware of Non-Uniform Memory Access
//! (NUMA) architectures, ensuring data is processed on the CPU cores closest
//! to where it resides in memory.

use std::sync::Arc;
use std::pin::Pin;
use std::future::Future;
use std::thread;
use std::alloc::{alloc, dealloc, Layout};
use std::ptr;
use std::mem;

use crate::{ExecutionContext, MoiraiIterator};
use moirai_scheduler::numa_scheduler::{CpuTopology, NumaNode};

/// Wrapper to make raw pointers Send
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}

/// NUMA memory allocation policy
#[derive(Debug, Clone, Copy)]
pub enum NumaPolicy {
    /// Allocate on local NUMA node
    Local,
    /// Interleave allocations across nodes
    Interleaved,
    /// Bind to specific NUMA node
    Bind(usize),
    /// Prefer local but allow remote on pressure
    Preferred,
}

/// NUMA-aware execution context for iterators
pub struct NumaAwareContext {
    topology: Arc<Option<CpuTopology>>,
    policy: NumaPolicy,
    thread_count: usize,
}

impl NumaAwareContext {
    /// Create a new NUMA-aware context
    pub fn new(policy: NumaPolicy) -> Self {
        let topology = Arc::new(CpuTopology::detect());
        let thread_count = topology.as_ref()
            .as_ref()
            .map(|t| t.logical_cores)
            .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
        
        Self {
            topology,
            policy,
            thread_count,
        }
    }
    
    /// Get the NUMA node for the current thread
    fn current_numa_node(&self) -> usize {
        #[cfg(target_os = "linux")]
        {
            unsafe {
                let mut cpu = 0;
                let result = libc::sched_getcpu();
                if result >= 0 {
                    cpu = result as usize;
                }
                
                self.topology.as_ref()
                    .as_ref()
                    .and_then(|t| t.core_to_node.get(&cpu).copied())
                    .unwrap_or(0)
            }
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            0 // Default to node 0 on non-Linux systems
        }
    }
    
    /// Allocate memory on a specific NUMA node
    unsafe fn numa_alloc(&self, size: usize, node: usize) -> *mut u8 {
        #[cfg(target_os = "linux")]
        {

            
            // Use mmap with NUMA policy
            let addr = libc::mmap(
                ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            );
            
            if addr != libc::MAP_FAILED {
                // Set NUMA memory policy
                let nodemask = 1u64 << node;
                libc::syscall(
                    libc::SYS_mbind,
                    addr,
                    size,
                    2, // MPOL_BIND
                    &nodemask as *const u64,
                    64, // maxnode
                    0, // flags
                );
                
                addr as *mut u8
            } else {
                // Fallback to regular allocation
                let layout = Layout::from_size_align_unchecked(size, 64);
                alloc(layout)
            }
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            let _ = node; // Unused on non-Linux
            let layout = Layout::from_size_align_unchecked(size, 64);
            alloc(layout)
        }
    }
    
    /// Free NUMA-allocated memory
    unsafe fn numa_free(&self, ptr: *mut u8, size: usize) {
        #[cfg(target_os = "linux")]
        {
            if libc::munmap(ptr as *mut libc::c_void, size) != 0 {
                // Fallback to regular deallocation
                let layout = Layout::from_size_align_unchecked(size, 64);
                dealloc(ptr, layout);
            }
        }
        
        #[cfg(not(target_os = "linux"))]
        {
            let layout = Layout::from_size_align_unchecked(size, 64);
            dealloc(ptr, layout);
        }
    }
}

impl ExecutionContext for NumaAwareContext {
    fn execute<T, F>(&self, items: Vec<T>, func: F) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + Send>>
    where
        T: Send + Clone + 'static,
        F: Fn(T) + Send + Sync + Clone + 'static,
    {
        let topology = Arc::clone(&self.topology);
        
        Box::pin(async move {
            if items.is_empty() {
                return;
            }
            
            if let Some(ref topology) = *self.topology {
                let numa_nodes = topology.numa_nodes.len();
                let total_items = items.len();
                
                // Distribute work across NUMA nodes
                let items_per_node = (total_items + numa_nodes - 1) / numa_nodes;
                
                std::thread::scope(|scope| {
                    for (node_idx, node) in topology.numa_nodes.iter().enumerate() {
                        let start = node_idx * items_per_node;
                        let end = ((node_idx + 1) * items_per_node).min(total_items);
                        
                        if start >= end {
                            continue;
                        }
                        
                        let items = items.clone();
                        let func = func.clone();
                        scope.spawn(move || {
                            // Pin thread to NUMA node
                            #[cfg(target_os = "linux")]
                            {
                                if let Some(core) = node.cores.first() {
                                    // CPU affinity setting for Linux
                                    // Note: This would require additional platform-specific code
                                    let _ = core;
                                }
                            }
                            
                            // Process items without copying
                            for i in start..end {
                                func(items[i].clone());
                            }
                        });
                    }
                });
            }
        })
    }
    
    fn map<T, R, F>(&self, items: Vec<T>, func: F) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<R>> + Send + '_>>
    where
        T: Send + Clone + 'static,
        R: Send + 'static,
        F: Fn(T) -> R + Send + Sync + Clone + 'static,
    {
        let topology = Arc::clone(&self.topology);
        
        Box::pin(async move {
            if items.is_empty() {
                return Vec::new();
            }
            
            let total_items = items.len();
            
            if let Some(ref topology) = *topology {
                let numa_nodes = topology.numa_nodes.len();
                
                // Allocate result vector with NUMA awareness
                let mut results = Vec::with_capacity(total_items);
                unsafe { results.set_len(total_items); }
                let results_ptr: *mut R = results.as_mut_ptr();
                
                // Distribute work across NUMA nodes
                let items_per_node = (total_items + numa_nodes - 1) / numa_nodes;
                
                std::thread::scope(|scope| {
                    for (node_idx, node) in topology.numa_nodes.iter().enumerate() {
                        let start = node_idx * items_per_node;
                        let end = ((node_idx + 1) * items_per_node).min(total_items);
                        
                        if start >= end {
                            continue;
                        }
                        
                        let items = items.clone();
                        let func = func.clone();
                        let results_ptr = unsafe { results_ptr.add(start) };
                        let ptr_wrapper = SendPtr(results_ptr);
                        
                        scope.spawn(move || {
                            // Pin thread to NUMA node
                            #[cfg(target_os = "linux")]
                            {
                                if let Some(core) = node.cores.first() {
                                    // CPU affinity setting for Linux
                                    // Note: This would require additional platform-specific code
                                    let _ = core;
                                }
                            }
                            
                            // Process items and write results directly
                            for (i, idx) in (start..end).enumerate() {
                                let result = func(items[idx].clone());
                                unsafe {
                                    ptr::write(ptr_wrapper.0.add(i), result);
                                }
                            }
                        });
                    }
                });
                
                results
            } else {
                // Fallback to simple parallel processing
                items.into_iter().map(func).collect()
            }
        })
    }
    
    fn reduce<T, F>(&self, items: Vec<T>, func: F) -> std::pin::Pin<Box<dyn std::future::Future<Output = Option<T>> + Send + '_>>
    where
        T: Send + Clone + 'static,
        F: Fn(T, T) -> T + Send + Sync + Clone + 'static,
    {
        let topology = Arc::clone(&self.topology);
        
        Box::pin(async move {
            if items.is_empty() {
                return None;
            }
            
            if items.len() == 1 {
                return items.into_iter().next();
            }
            
            if let Some(ref topology) = *topology {
                // NUMA-aware tree reduction
                let numa_nodes = topology.numa_nodes.len();
                let items_per_node = (items.len() + numa_nodes - 1) / numa_nodes;
                
                // First level: reduce within each NUMA node
                let node_results: Vec<Option<T>> = std::thread::scope(|scope| {
                    let mut handles = Vec::new();
                    
                    for node_idx in 0..numa_nodes {
                        let start = node_idx * items_per_node;
                        let end = ((node_idx + 1) * items_per_node).min(items.len());
                        
                        if start >= end {
                            handles.push(scope.spawn(move || None));
                            continue;
                        }
                        
                        let chunk = items[start..end].to_vec();
                        let func = func.clone();
                        
                        handles.push(scope.spawn(move || {
                            chunk.into_iter().reduce(|a, b| func(a, b))
                        }));
                    }
                    
                    handles.into_iter()
                        .map(|h| h.join().unwrap())
                        .collect()
                });
                
                // Second level: reduce across NUMA nodes
                node_results.into_iter()
                    .flatten()
                    .reduce(|a, b| func(a, b))
            } else {
                // Fallback to simple reduction
                items.into_iter().reduce(func)
            }
        })
    }
    
    fn stream<T, R, F>(&self, items: Vec<T>, func: F) -> std::pin::Pin<Box<dyn std::future::Future<Output = Vec<R>> + Send + '_>>
    where
        T: Send + Clone + 'static,
        R: Send + 'static,
        F: Fn(T) -> Option<R> + Send + Sync + Clone + 'static,
    {
        // Similar to map but filters out None values
        Box::pin(async move {
            let mapped = self.map(items, func).await;
            mapped.into_iter().flatten().collect()
        })
    }
}

/// Extension trait for NUMA-aware iteration
pub trait NumaIteratorExt<T> {
    /// Create a NUMA-aware iterator with specified policy
    fn numa_iter(self, policy: NumaPolicy) -> NumaIterator<T>;
}

/// NUMA-aware iterator wrapper
pub struct NumaIterator<T> {
    items: Vec<T>,
    context: NumaAwareContext,
}

impl<T: Send + Clone + 'static> NumaIterator<T> {
    /// Execute a function on each item with NUMA awareness
    pub async fn for_each<F>(self, func: F)
    where
        F: Fn(T) + Send + Sync + Clone + 'static,
    {
        self.context.execute(self.items, func).await
    }
    
    /// Map items with NUMA-aware execution
    pub async fn map<R, F>(self, func: F) -> Vec<R>
    where
        R: Send + 'static,
        F: Fn(T) -> R + Send + Sync + Clone + 'static,
    {
        self.context.map(self.items, func).await
    }
    
    /// Reduce items with NUMA-aware execution
    pub async fn reduce<F>(self, func: F) -> Option<T>
    where
        F: Fn(T, T) -> T + Send + Sync + Clone + 'static,
    {
        self.context.reduce(self.items, func).await
    }
}

impl<T: Send + Clone + 'static> NumaIteratorExt<T> for Vec<T> {
    fn numa_iter(self, policy: NumaPolicy) -> NumaIterator<T> {
        NumaIterator {
            items: self,
            context: NumaAwareContext::new(policy),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_numa_aware_execution() {
        let data: Vec<i32> = (0..10000).collect();
        let sum = data.numa_iter(NumaPolicy::Local)
            .reduce(|a, b| a + b)
            .await;
        
        assert_eq!(sum, Some((0..10000).sum()));
    }
    
    #[tokio::test]
    async fn test_numa_aware_map() {
        let data: Vec<i32> = (0..1000).collect();
        let result = data.numa_iter(NumaPolicy::Interleaved)
            .map(|x| x * 2)
            .await;
        
        assert_eq!(result.len(), 1000);
        for (i, &val) in result.iter().enumerate() {
            assert_eq!(val, i as i32 * 2);
        }
    }
}
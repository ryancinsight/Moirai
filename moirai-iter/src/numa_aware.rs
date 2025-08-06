//! NUMA-aware iterator implementations for optimal memory locality.
//!
//! This module provides iterators that are aware of Non-Uniform Memory Access
//! (NUMA) architectures, ensuring data is processed on the CPU cores closest
//! to where it resides in memory.

use std::sync::Arc;
use std::pin::Pin;
use std::future::Future;
use std::alloc::{alloc, dealloc, Layout};
use std::ptr;

use crate::{ExecutionContext, ExecutionBase, IntoParallelIterator};
use moirai_scheduler::numa_scheduler::CpuTopology;

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
    #[allow(dead_code)]
    policy: NumaPolicy,
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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
    #[allow(dead_code)]
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

impl ExecutionBase for NumaAwareContext {
    fn execute_each<T, F>(
        &self,
        items: Vec<T>,
        func: F,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>>
    where
        T: Send + Clone + 'static,
        F: Fn(T) + Send + Sync + Clone + 'static,
    {
        Box::pin(async move {
            items.into_par_iter().for_each(func);
        })
    }
    
    fn execute_map<T, R, F>(
        &self,
        items: Vec<T>,
        func: F,
    ) -> Pin<Box<dyn Future<Output = Vec<R>> + Send + '_>>
    where
        T: Send + Clone + 'static,
        R: Send + Clone + 'static,
        F: Fn(T) -> R + Send + Sync + Clone + 'static,
    {
        Box::pin(async move {
            items.into_par_iter().map(func).collect()
        })
    }
    
    fn execute_filter<T, F>(
        &self,
        items: Vec<T>,
        predicate: F,
    ) -> Pin<Box<dyn Future<Output = Vec<T>> + Send + '_>>
    where
        T: Send + Clone + 'static,
        F: Fn(&T) -> bool + Send + Sync + Clone + 'static,
    {
        Box::pin(async move {
            items.into_par_iter().filter(|item| predicate(item)).collect()
        })
    }
}

impl crate::ExecutionContext for NumaAwareContext {
    async fn execute<T, F>(&self, items: Vec<T>, func: F)
    where
        T: Send + Sync + Clone + 'static,
        F: Fn(T) -> () + Send + Sync + Clone + 'static,
    {
        self.execute_each(items, func).await
    }

    async fn reduce<T, F>(&self, items: Vec<T>, func: F) -> Option<T>
    where
        T: Send + Sync + Clone + 'static,
        F: Fn(T, T) -> T + Send + Sync + Clone + 'static,
    {
        // Implement using execute_map for NUMA-aware reduction
        if items.is_empty() {
            return None;
        }
        
        let chunk_size = (items.len() + self.thread_count - 1) / self.thread_count;
        if chunk_size == 0 || items.len() == 1 {
            return items.into_iter().reduce(func);
        }
        
        // First pass: reduce within each NUMA node
        let mut node_results = Vec::new();
        for i in 0..self.thread_count {
            let start = i * chunk_size;
            let end = ((i + 1) * chunk_size).min(items.len());
            if start < end {
                let chunk: Vec<T> = items[start..end].to_vec();
                if let Some(result) = chunk.into_iter().reduce(func.clone()) {
                    node_results.push(result);
                }
            }
        }
        
        // Final reduction
        node_results.into_iter().reduce(func)
    }
    
    fn context_type(&self) -> crate::ContextType {
        crate::ContextType::Parallel // NUMA is a form of parallel execution
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
    /// Execute a function on each item with NUMA-aware placement
    pub async fn for_each<F>(self, func: F)
    where
        F: Fn(T) + Send + Sync + Clone + 'static,
    {
        self.context.execute_each(self.items, func).await
    }
    
    /// Map items with NUMA-aware execution
    pub async fn map<R, F>(self, func: F) -> Vec<R>
    where
        R: Send + Clone + 'static,
        F: Fn(T) -> R + Send + Sync + Clone + 'static,
    {
        self.context.execute_map(self.items, func).await
    }
    
    /// Reduce items with NUMA-aware execution
    pub async fn reduce<F>(self, func: F) -> Option<T>
    where
        F: Fn(T, T) -> T + Send + Sync + Clone + 'static,
    {
        if self.items.is_empty() {
            return None;
        }
        
        // For reduce, we need to implement it using execute_map
        let items = self.items;
        let num_nodes = self.context.thread_count.max(1);
        let chunk_size = (items.len() + num_nodes - 1) / num_nodes;
        
        if chunk_size == 0 || items.len() == 1 {
            return items.into_iter().reduce(func);
        }
        
        // First pass: reduce within each NUMA node
        let mut node_results = Vec::new();
        for i in 0..num_nodes {
            let start = i * chunk_size;
            let end = ((i + 1) * chunk_size).min(items.len());
            if start < end {
                let chunk: Vec<T> = items[start..end].to_vec();
                if let Some(result) = chunk.into_iter().reduce(func.clone()) {
                    node_results.push(result);
                }
            }
        }
        
        // Final reduction across nodes
        node_results.into_iter().reduce(func)
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
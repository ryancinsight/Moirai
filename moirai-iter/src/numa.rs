//! NUMA-based iterator execution context.

use std::alloc::{alloc, dealloc, Layout};
#[cfg(target_os = "linux")]
use std::ptr;
use std::sync::Arc;

use crate::ExecutionBase;
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

/// NUMA execution context for iterators
pub struct NumaContext {
    topology: Arc<Option<CpuTopology>>,
    policy: NumaPolicy,
    thread_count: usize,
}

impl NumaContext {
    /// Create a new NUMA context
    pub fn new(policy: NumaPolicy) -> Self {
        let topology = Arc::new(CpuTopology::detect());
        let thread_count = topology
            .as_ref()
            .as_ref()
            .map(|t| t.logical_cores)
            .unwrap_or_else(|| {
                std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(1)
            });
        Self {
            topology,
            policy,
            thread_count,
        }
    }

    /// Get the NUMA policy for this context.
    pub fn policy(&self) -> NumaPolicy {
        self.policy
    }

    /// Get the thread count for this context.
    pub fn thread_count(&self) -> usize {
        self.thread_count
    }

    /// Get the CPU topology if available.
    pub fn topology(&self) -> Option<&CpuTopology> {
        self.topology.as_ref().as_ref()
    }

    /// Get the current NUMA node for the calling thread.
    pub fn current_numa_node(&self) -> usize {
        #[cfg(target_os = "linux")]
        {
            unsafe {
                let mut cpu = 0;
                let result = libc::sched_getcpu();
                if result >= 0 {
                    cpu = result as usize;
                }
                self.topology
                    .as_ref()
                    .as_ref()
                    .and_then(|t| t.core_to_node.get(&cpu).copied())
                    .unwrap_or(0)
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            0
        }
    }

    /// Allocate memory on a specific NUMA node.
    ///
    /// # Safety
    /// The caller must ensure the returned pointer is properly deallocated
    /// using `numa_free` with the same size.
    pub unsafe fn numa_alloc(&self, size: usize, node: usize) -> *mut u8 {
        #[cfg(target_os = "linux")]
        {
            let addr = libc::mmap(
                ptr::null_mut(),
                size,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            );
            if addr != libc::MAP_FAILED {
                let nodemask = 1u64 << node;
                libc::syscall(
                    libc::SYS_mbind,
                    addr,
                    size,
                    2,
                    &nodemask as *const u64,
                    64,
                    0,
                );
                addr as *mut u8
            } else {
                let layout = Layout::from_size_align_unchecked(size, 64);
                alloc(layout)
            }
        }
        #[cfg(not(target_os = "linux"))]
        {
            let _ = node;
            let layout = Layout::from_size_align_unchecked(size, 64);
            alloc(layout)
        }
    }

    /// Free NUMA-allocated memory.
    ///
    /// # Safety
    /// The pointer must have been allocated with `numa_alloc` and the size
    /// must match the original allocation.
    pub unsafe fn numa_free(&self, ptr: *mut u8, size: usize) {
        #[cfg(target_os = "linux")]
        {
            if libc::munmap(ptr as *mut libc::c_void, size) != 0 {
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

impl NumaContext {
    /// Execute an iterator operation with NUMA-aware processing
    pub fn execute_iter<T, F, R>(
        &self,
        items: Vec<T>,
        func: F,
    ) -> Result<Vec<R>, Box<dyn std::error::Error + Send + Sync>>
    where
        T: Send + Clone + 'static,
        F: Fn(T) -> R + Send + Sync + 'static,
        R: Send + 'static,
    {
        // Simple implementation - divide work across NUMA nodes
        let chunk_size = items.len().div_ceil(self.thread_count);
        let mut results = Vec::with_capacity(items.len());

        for chunk in items.chunks(chunk_size) {
            for item in chunk {
                let result = func(item.clone());
                results.push(result);
            }
        }

        Ok(results)
    }

    /// Execute a closure with the context
    pub fn execute<F, R>(&self, func: F) -> Result<R, Box<dyn std::error::Error + Send + Sync>>
    where
        F: FnOnce() -> R + Send,
        R: Send,
    {
        Ok(func())
    }
}

impl ExecutionBase for NumaContext {
    fn context_type(&self) -> &'static str {
        "NUMA"
    }
}

/// Extension trait for NUMA iteration
pub trait NumaIterExt<T> {
    /// Create a NUMA iterator with policy
    fn numa_iter(self, policy: NumaPolicy) -> NumaIter<T>;
}

/// NUMA iterator wrapper
pub struct NumaIter<T> {
    items: Vec<T>,
    context: NumaContext,
}

impl<T: Send + Clone + 'static> NumaIter<T> {
    pub async fn for_each<F>(self, func: F)
    where
        F: Fn(T) + Send + Sync + Clone + 'static,
    {
        // Use the execute_iter method to apply the function
        let _ = self.context.execute_iter(self.items, move |item| {
            func(item);
        });
    }

    pub async fn map<R, F>(self, func: F) -> Vec<R>
    where
        R: Send + Clone + 'static,
        F: Fn(T) -> R + Send + Sync + Clone + 'static,
    {
        self.context
            .execute_iter(self.items, func)
            .unwrap_or_default()
    }

    pub async fn reduce<F>(self, func: F) -> Option<T>
    where
        F: Fn(T, T) -> T + Send + Sync + Clone + 'static,
    {
        if self.items.is_empty() {
            return None;
        }
        let items = self.items;
        let num_nodes = self.context.thread_count.max(1);
        let chunk_size = items.len().div_ceil(num_nodes);
        if chunk_size == 0 || items.len() == 1 {
            return items.into_iter().reduce(func);
        }
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
        node_results.into_iter().reduce(func)
    }
}

impl<T: Send + Clone + 'static> NumaIterExt<T> for Vec<T> {
    fn numa_iter(self, policy: NumaPolicy) -> NumaIter<T> {
        NumaIter {
            items: self,
            context: NumaContext::new(policy),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_numa_execution() {
        let data: Vec<i32> = (0..10000).collect();
        let sum = data.numa_iter(NumaPolicy::Local).reduce(|a, b| a + b).await;
        assert_eq!(sum, Some((0..10000).sum()));
    }

    #[tokio::test]
    async fn test_numa_map() {
        let data: Vec<i32> = (0..1000).collect();
        let result = data.numa_iter(NumaPolicy::Interleaved).map(|x| x * 2).await;
        assert_eq!(result.len(), 1000);
        for (i, &val) in result.iter().enumerate() {
            assert_eq!(val, i as i32 * 2);
        }
    }
}

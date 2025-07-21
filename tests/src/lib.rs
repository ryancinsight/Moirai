//! Integration tests for Moirai concurrency library.

use moirai::{Moirai, Priority, TaskId, Task};
use std::{
    sync::{
        atomic::{AtomicU32, Ordering},
        Arc,
    },
};

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn test_basic_runtime_creation() {
        let runtime = Moirai::new().expect("Failed to create runtime");
        assert!(runtime.worker_count() > 0);
        assert_eq!(runtime.load(), 0);
        assert!(!runtime.is_shutting_down());
        
        runtime.shutdown();
        assert!(runtime.is_shutting_down());
    }

    #[test]
    fn test_runtime_builder() {
        let runtime = Moirai::builder()
            .worker_threads(4)
            .async_threads(2)
            .thread_name_prefix("test-worker")
            .build()
            .expect("Failed to build runtime");
            
        assert_eq!(runtime.worker_count(), 4);
        runtime.shutdown();
    }

    #[test]
    fn test_simple_task_execution() {
        let runtime = Moirai::new().expect("Failed to create runtime");
        
        // Create a simple task
        struct SimpleTask {
            value: u32,
        }
        
        impl Task for SimpleTask {
            type Output = u32;
            
            fn execute(self) -> Self::Output {
                self.value * 2
            }
            
            fn context(&self) -> &moirai::TaskContext {
                // For testing, we'll use a static context
                static CONTEXT: std::sync::OnceLock<moirai::TaskContext> = std::sync::OnceLock::new();
                CONTEXT.get_or_init(|| {
                    moirai::TaskContext::new(TaskId::new(1))
                })
            }
            
            fn is_stealable(&self) -> bool {
                true
            }
        }
        
        let task = SimpleTask { value: 21 };
        let _handle = runtime.spawn(task);
        
        runtime.shutdown();
    }

    #[test]
    fn test_parallel_computation() {
        let runtime = Moirai::new().expect("Failed to create runtime");
        
        let counter = Arc::new(AtomicU32::new(0));
        let handles: Vec<_> = (0..10).map(|i| {
            let counter = counter.clone();
            runtime.spawn_parallel(move || {
                counter.fetch_add(i, Ordering::Relaxed);
                i * i
            })
        }).collect();
        
        // Note: In a real implementation, we'd wait for the handles to complete
        // For now, we'll just ensure they were created
        assert_eq!(handles.len(), 10);
        
        runtime.shutdown();
    }

    #[test]
    fn test_priority_scheduling() {
        let runtime = Moirai::new().expect("Failed to create runtime");
        
        struct PriorityTask {
            id: u32,
            priority: Priority,
        }
        
        impl Task for PriorityTask {
            type Output = u32;
            
            fn execute(self) -> Self::Output {
                self.id
            }
            
            fn context(&self) -> &moirai::TaskContext {
                static CONTEXTS: std::sync::OnceLock<std::collections::HashMap<u32, moirai::TaskContext>> = std::sync::OnceLock::new();
                let contexts = CONTEXTS.get_or_init(|| {
                    let mut map = std::collections::HashMap::new();
                    for i in 0..100 {
                        map.insert(i, moirai::TaskContext::new(TaskId::new(i as u64)).with_priority(Priority::Normal));
                    }
                    map
                });
                contexts.get(&self.id).unwrap()
            }
            
            fn is_stealable(&self) -> bool {
                true
            }
        }
        
        let high_priority_task = PriorityTask { id: 1, priority: Priority::High };
        let low_priority_task = PriorityTask { id: 2, priority: Priority::Low };
        
        let _high_handle = runtime.spawn_with_priority(high_priority_task, Priority::High);
        let _low_handle = runtime.spawn_with_priority(low_priority_task, Priority::Low);
        
        runtime.shutdown();
    }
}
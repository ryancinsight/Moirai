//! Task registry and lifecycle management.
//!
//! This module provides centralized task tracking and metadata management,
//! following the Information Expert pattern where the registry owns all
//! task-related data and provides efficient operations for task management.

use moirai_core::{TaskId, Priority, TaskStatus};
use std::{
    collections::HashMap,
    future::Future,
    pin::Pin,
    sync::{
        atomic::{AtomicU64, Ordering},
        RwLock,
    },
    task::{Context, Poll, Waker},
    time::Instant,
};

/// Task registry for tracking active tasks.
/// 
/// Follows the Information Expert pattern by owning task metadata
/// and providing efficient lookups for task management operations.
/// Designed for high-concurrency scenarios with minimal lock contention.
pub struct TaskRegistry {
    /// Map of task IDs to their metadata, protected by RwLock for concurrent access
    tasks: RwLock<HashMap<TaskId, TaskMetadata>>,
    /// Atomic counter for generating unique task IDs
    next_id: AtomicU64,
}

/// Metadata about a task in the registry.
/// 
/// Contains all information necessary to track a task's lifecycle,
/// performance characteristics, and execution context.
#[derive(Debug, Clone)]
pub struct TaskMetadata {
    /// Unique task identifier
    pub id: TaskId,
    /// Current execution status
    pub status: TaskStatus,
    /// Task scheduling priority
    pub priority: Priority,
    /// When the task was initially spawned
    pub spawn_time: Instant,
    /// When task execution actually began (None if not started)
    pub start_time: Option<Instant>,
    /// When task execution completed (None if not completed)
    pub completion_time: Option<Instant>,
    /// Optional waker for async task coordination
    pub waker: Option<Waker>,
    /// Number of times this task has been preempted
    pub preemption_count: u32,
    /// Total CPU time consumed in nanoseconds
    pub cpu_time_ns: u64,
    /// Memory usage in bytes
    pub memory_used_bytes: u64,
}

impl TaskRegistry {
    /// Create a new task registry.
    pub fn new() -> Self {
        Self {
            tasks: RwLock::new(HashMap::new()),
            next_id: AtomicU64::new(0), // Start from 0 to match test expectations
        }
    }

    /// Generate a new unique task ID.
    /// 
    /// Uses atomic increment to ensure thread-safe ID generation
    /// without requiring locks, enabling high-throughput task spawning.
    pub fn generate_id(&self) -> TaskId {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        TaskId::new(id)
    }

    /// Register a new task with the given priority.
    /// 
    /// Returns the assigned task ID for future reference.
    /// The task is initially in Pending status.
    pub fn register_task(&self, priority: Priority) -> TaskId {
        let id = self.generate_id();
        let metadata = TaskMetadata {
            id,
            status: TaskStatus::Pending,
            priority,
            spawn_time: Instant::now(),
            start_time: None,
            completion_time: None,
            waker: None,
            preemption_count: 0,
            cpu_time_ns: 0,
            memory_used_bytes: 0,
        };

        {
            let mut tasks = self.tasks.write().unwrap();
            tasks.insert(id, metadata);
        }

        id
    }

    /// Update task status with atomic metadata changes.
    /// 
    /// Ensures consistent state transitions and timing information.
    /// Returns true if the task was found and updated, false otherwise.
    pub fn update_status(&self, task_id: TaskId, new_status: TaskStatus) -> bool {
        let mut tasks = self.tasks.write().unwrap();
        if let Some(metadata) = tasks.get_mut(&task_id) {
            metadata.status = new_status;
            
            // Update timing information based on status transition
            match new_status {
                TaskStatus::Running => {
                    if metadata.start_time.is_none() {
                        metadata.start_time = Some(Instant::now());
                    }
                }
                TaskStatus::Completed | TaskStatus::Failed(_) => {
                    metadata.completion_time = Some(Instant::now());
                }
                _ => {}
            }
            
            true
        } else {
            false
        }
    }

    /// Set a waker for async task coordination.
    pub fn set_waker(&self, task_id: TaskId, waker: Waker) -> bool {
        let mut tasks = self.tasks.write().unwrap();
        if let Some(metadata) = tasks.get_mut(&task_id) {
            metadata.waker = Some(waker);
            true
        } else {
            false
        }
    }

    /// Wake a task if it has a registered waker.
    pub fn wake_task(&self, task_id: TaskId) -> bool {
        let mut tasks = self.tasks.write().unwrap();
        if let Some(metadata) = tasks.get_mut(&task_id) {
            if let Some(waker) = metadata.waker.take() {
                waker.wake();
                true
            } else {
                false
            }
        } else {
            false
        }
    }

    /// Get task metadata by ID.
    /// 
    /// Returns a clone of the metadata to avoid holding read locks
    /// for extended periods, enabling high concurrency.
    pub fn get_task(&self, task_id: TaskId) -> Option<TaskMetadata> {
        let tasks = self.tasks.read().unwrap();
        tasks.get(&task_id).cloned()
    }

    /// Get all tasks with a specific status.
    pub fn get_tasks_by_status(&self, status: TaskStatus) -> Vec<TaskMetadata> {
        let tasks = self.tasks.read().unwrap();
        tasks
            .values()
            .filter(|metadata| metadata.status == status)
            .cloned()
            .collect()
    }

    /// Remove completed or failed tasks from the registry.
    /// 
    /// Returns the number of tasks removed for monitoring purposes.
    /// Helps prevent memory leaks from accumulating task metadata.
    pub fn cleanup_completed_tasks(&self) -> usize {
        let mut tasks = self.tasks.write().unwrap();
        let initial_count = tasks.len();
        
        tasks.retain(|_, metadata| {
            !matches!(metadata.status, TaskStatus::Completed | TaskStatus::Failed(_))
        });
        
        initial_count - tasks.len()
    }

    /// Get the total number of registered tasks.
    pub fn task_count(&self) -> usize {
        let tasks = self.tasks.read().unwrap();
        tasks.len()
    }

    /// Update task performance metrics.
    pub fn update_task_metrics(&self, task_id: TaskId, cpu_time_ns: u64, memory_bytes: u64) -> bool {
        let mut tasks = self.tasks.write().unwrap();
        if let Some(metadata) = tasks.get_mut(&task_id) {
            metadata.cpu_time_ns = cpu_time_ns;
            metadata.memory_used_bytes = memory_bytes;
            true
        } else {
            false
        }
    }

    /// Increment preemption count for a task.
    pub fn increment_preemption(&self, task_id: TaskId) -> bool {
        let mut tasks = self.tasks.write().unwrap();
        if let Some(metadata) = tasks.get_mut(&task_id) {
            metadata.preemption_count += 1;
            true
        } else {
            false
        }
    }

    /// Get registry statistics for monitoring.
    pub fn get_statistics(&self) -> RegistryStatistics {
        let tasks = self.tasks.read().unwrap();
        let mut stats = RegistryStatistics::default();
        
        for metadata in tasks.values() {
            match metadata.status {
                TaskStatus::Pending => stats.pending_count += 1,
                TaskStatus::Running => stats.running_count += 1,
                TaskStatus::Completed => stats.completed_count += 1,
                TaskStatus::Failed(_) => stats.failed_count += 1,
            }
            stats.total_cpu_time_ns += metadata.cpu_time_ns;
            stats.total_memory_bytes += metadata.memory_used_bytes;
        }
        
        stats.total_count = tasks.len();
        stats
    }
}

impl Default for TaskRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Statistics about the task registry state.
#[derive(Debug, Clone, Default)]
pub struct RegistryStatistics {
    /// Total number of registered tasks
    pub total_count: usize,
    /// Number of tasks in pending state
    pub pending_count: usize,
    /// Number of tasks currently running
    pub running_count: usize,
    /// Number of completed tasks
    pub completed_count: usize,
    /// Number of failed tasks
    pub failed_count: usize,
    /// Total CPU time consumed by all tasks
    pub total_cpu_time_ns: u64,
    /// Total memory used by all tasks
    pub total_memory_bytes: u64,
}

/// Future for waiting on task completion.
/// 
/// Provides an async interface for waiting on task status changes,
/// enabling efficient coordination between async and parallel execution.
pub struct TaskWaitFuture {
    /// The task being waited on
    task_id: TaskId,
    /// Reference to the task registry
    registry: std::sync::Arc<TaskRegistry>,
    /// Whether this future has been polled before
    polled: bool,
}

impl TaskWaitFuture {
    /// Create a new future for waiting on a task.
    pub fn new(task_id: TaskId, registry: std::sync::Arc<TaskRegistry>) -> Self {
        Self {
            task_id,
            registry,
            polled: false,
        }
    }
}

impl Future for TaskWaitFuture {
    type Output = TaskStatus;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        if let Some(metadata) = self.registry.get_task(self.task_id) {
            match metadata.status {
                TaskStatus::Completed | TaskStatus::Failed(_) => {
                    Poll::Ready(metadata.status)
                }
                _ => {
                    // Register waker for notification when task completes
                    if !self.polled {
                        self.registry.set_waker(self.task_id, cx.waker().clone());
                        self.polled = true;
                    }
                    Poll::Pending
                }
            }
        } else {
            // Task not found - assume it was cleaned up after completion
            Poll::Ready(TaskStatus::Completed)
        }
    }
}
/// A unique identifier for tasks in the Moirai runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TaskId(pub u64);

impl core::fmt::Display for TaskId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Task#{}", self.0)
    }
}

impl TaskId {
    /// Create a new task ID.
    pub const fn new(id: u64) -> Self {
        Self(id)
    }
}

/// Priority levels for task scheduling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum Priority {
    /// Low priority tasks (background work)
    Low = 0,
    /// Normal priority tasks (default)
    #[default]
    Normal = 1,
    /// High priority tasks (interactive work)
    High = 2,
    /// Critical priority tasks (system-level work)
    Critical = 3,
}

/// Task execution context and metadata.
#[derive(Debug, Clone)]
pub struct TaskContext {
    /// Unique identifier for this task
    pub id: TaskId,
    /// Priority level for scheduling
    pub priority: Priority,
    /// Optional name for debugging
    pub name: Option<&'static str>,
}

impl TaskContext {
    /// Create a new task context.
    pub const fn new(id: TaskId) -> Self {
        Self {
            id,
            priority: Priority::Normal,
            name: None,
        }
    }

    /// Set the priority for this task.
    pub const fn with_priority(mut self, priority: Priority) -> Self {
        self.priority = priority;
        self
    }

    /// Set the name for this task.
    pub const fn with_name(mut self, name: &'static str) -> Self {
        self.name = Some(name);
        self
    }
}

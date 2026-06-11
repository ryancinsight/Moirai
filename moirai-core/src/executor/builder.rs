//! Executor builder implementation.

use super::config::{CleanupConfig, ExecutorConfig, MemoryConfig, PreemptionConfig};
use super::plugin::ExecutorPlugin;
use crate::platform::{Box, String, Vec};

/// Builder for creating executors with custom configuration.
pub struct ExecutorBuilder {
    pub(crate) config: ExecutorConfig,
    #[allow(dead_code)]
    pub(crate) plugins: Vec<Box<dyn ExecutorPlugin>>,
}

impl ExecutorBuilder {
    /// Creates a new executor builder with default settings.
    ///
    /// # Returns
    /// A new builder instance ready for configuration
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: ExecutorConfig::default(),
            plugins: Vec::new(),
        }
    }

    /// Sets the number of worker threads for CPU-bound tasks.
    ///
    /// # Arguments
    /// * `count` - Number of worker threads to create
    ///
    /// # Returns
    /// The builder instance for method chaining
    #[must_use]
    pub fn worker_threads(mut self, count: usize) -> Self {
        self.config.worker_threads = count;
        self
    }

    /// Sets the number of threads for async task execution.
    ///
    /// # Arguments
    /// * `count` - Number of async threads to create
    ///
    /// # Returns
    /// The builder instance for method chaining
    #[must_use]
    pub fn async_threads(mut self, count: usize) -> Self {
        self.config.async_threads = count;
        self
    }

    /// Sets the maximum size of the global task queue.
    ///
    /// # Arguments
    /// * `size` - Maximum number of tasks in the global queue
    ///
    /// # Returns
    /// The builder instance for method chaining
    #[must_use]
    pub fn max_global_queue_size(mut self, size: usize) -> Self {
        self.config.max_global_queue_size = size;
        self
    }

    /// Sets the maximum size of per-worker local queues.
    ///
    /// # Arguments
    /// * `size` - Maximum number of tasks in each local queue
    ///
    /// # Returns
    /// The builder instance for method chaining
    #[must_use]
    pub fn max_local_queue_size(mut self, size: usize) -> Self {
        self.config.max_local_queue_size = size;
        self
    }

    /// Sets the thread name prefix for executor threads.
    ///
    /// # Arguments
    /// * `prefix` - String prefix for thread names
    ///
    /// # Returns
    /// The builder instance for method chaining
    #[must_use]
    pub fn thread_name_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.config.thread_name_prefix = prefix.into();
        self
    }

    /// Enable or disable NUMA awareness.
    #[cfg(feature = "numa")]
    #[must_use]
    pub fn numa_aware(mut self, enabled: bool) -> Self {
        self.config.numa_aware = enabled;
        self
    }

    /// Enable or disable metrics collection.
    #[cfg(feature = "metrics")]
    #[must_use]
    pub fn enable_metrics(mut self, enabled: bool) -> Self {
        self.config.enable_metrics = enabled;
        self
    }

    /// Configures preemption behavior for the executor.
    ///
    /// # Arguments
    /// * `config` - Preemption configuration settings
    ///
    /// # Returns
    /// The builder instance for method chaining
    #[must_use]
    pub fn preemption_config(mut self, config: PreemptionConfig) -> Self {
        self.config.preemption = config;
        self
    }

    /// Configures memory management settings.
    ///
    /// # Arguments
    /// * `config` - Memory configuration settings
    ///
    /// # Returns
    /// The builder instance for method chaining
    #[must_use]
    pub fn memory_config(mut self, config: MemoryConfig) -> Self {
        self.config.memory = config;
        self
    }

    /// Configures cleanup and maintenance settings.
    ///
    /// # Arguments
    /// * `config` - Cleanup configuration settings
    ///
    /// # Returns
    /// The builder instance for method chaining
    #[must_use]
    pub fn cleanup_config(mut self, config: CleanupConfig) -> Self {
        self.config.cleanup = config;
        self
    }
}

impl Default for ExecutorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

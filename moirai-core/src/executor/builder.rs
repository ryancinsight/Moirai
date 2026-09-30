//! Executor builder implementation.

use super::{
    config::{CleanupConfig, ExecutorConfig, MemoryConfig, PreemptionConfig},
    placement::WorkerPlacement,
};
use crate::platform::String;

/// Builder for creating executors with custom configuration.
pub struct ExecutorBuilder {
    pub(crate) config: ExecutorConfig,
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

    /// Sets whether each worker is confined to one logical processor.
    ///
    /// # Arguments
    /// * `placement` - The worker placement policy
    ///
    /// # Returns
    /// The builder instance for method chaining
    #[must_use]
    pub fn worker_placement(mut self, placement: WorkerPlacement) -> Self {
        self.config.worker_placement = placement;
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

    /// Sets the aggregate maximum of the workers' external admission queues.
    ///
    /// # Arguments
    /// * `size` - Maximum queued tasks across all worker injectors
    ///
    /// # Returns
    /// The builder instance for method chaining
    #[must_use]
    pub fn max_global_queue_size(mut self, size: usize) -> Self {
        self.config.max_global_queue_size = size;
        self
    }

    /// Sets the initial capacity of each resizable local priority queue.
    ///
    /// # Arguments
    /// * `size` - Requested initial slots per local priority queue
    ///
    /// # Returns
    /// The builder instance for method chaining
    #[must_use]
    pub fn local_queue_initial_capacity(mut self, size: usize) -> Self {
        self.config.local_queue_initial_capacity = size;
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

#[cfg(test)]
mod tests {
    use super::{ExecutorBuilder, WorkerPlacement};

    #[test]
    fn local_queue_initial_capacity_updates_the_configuration() {
        let builder = ExecutorBuilder::new().local_queue_initial_capacity(17);

        assert_eq!(builder.config.local_queue_initial_capacity, 17);
    }

    #[test]
    fn worker_placement_defaults_to_unbound_and_is_settable() {
        assert_eq!(
            ExecutorBuilder::new().config.worker_placement,
            WorkerPlacement::Unbound
        );
        assert_eq!(
            ExecutorBuilder::new()
                .worker_placement(WorkerPlacement::Pinned)
                .config
                .worker_placement,
            WorkerPlacement::Pinned
        );
    }
}

use crate::Moirai;
use moirai_core::{error::ExecutorResult, executor::ExecutorConfig};
use moirai_executor::HybridExecutor;
use std::sync::Arc;

/// Builder for configuring the Moirai runtime.
pub struct MoiraiBuilder {
    config: ExecutorConfig,
}

impl MoiraiBuilder {
    /// Create a new builder with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: ExecutorConfig::default(),
        }
    }

    /// Set the number of worker threads for parallel tasks.
    #[must_use]
    pub fn worker_threads(mut self, count: usize) -> Self {
        self.config.worker_threads = count;
        self
    }

    /// Set the number of threads dedicated to async tasks.
    #[must_use]
    pub fn async_threads(mut self, count: usize) -> Self {
        self.config.async_threads = count;
        self
    }

    /// Set the maximum global queue size.
    #[must_use]
    pub fn max_global_queue_size(mut self, size: usize) -> Self {
        self.config.max_global_queue_size = size;
        self
    }

    /// Set the maximum local queue size.
    #[must_use]
    pub fn max_local_queue_size(mut self, size: usize) -> Self {
        self.config.max_local_queue_size = size;
        self
    }

    /// Enable or disable NUMA awareness.
    #[cfg(feature = "numa")]
    #[must_use]
    pub fn numa_aware(mut self, enabled: bool) -> Self {
        self.config.numa_aware = enabled;
        self
    }

    /// Set the thread name prefix.
    #[must_use]
    pub fn thread_name_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.config.thread_name_prefix = prefix.into();
        self
    }

    /// Enable or disable metrics collection.
    #[cfg(feature = "metrics")]
    #[must_use]
    pub fn enable_metrics(self, enabled: bool) -> Self {
        // Metrics configuration would go here
        let _ = enabled; // Suppress unused variable warning
        self
    }

    /// Build the Moirai runtime.
    ///
    /// # Errors
    ///
    /// Returns an error if the runtime cannot be initialized.
    pub fn build(self) -> ExecutorResult<Moirai> {
        let executor = HybridExecutor::new(self.config)?;
        Ok(Moirai {
            executor: Arc::new(executor),
        })
    }
}

impl Default for MoiraiBuilder {
    fn default() -> Self {
        Self::new()
    }
}

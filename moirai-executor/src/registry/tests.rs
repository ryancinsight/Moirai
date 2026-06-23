#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use std::time::Duration;

    use super::super::registry::TaskRegistry;
    use super::super::state::TASK_STATE_BLOCK_SIZE;

    #[test]
    fn lifecycle_token_records_started_and_completed_metadata() {
        let mut registry = TaskRegistry::new();
        let lifecycle = registry.register_task_with_id(7);

        let running = lifecycle.start(3);
        let started = registry.get_metadata(7).unwrap();
        assert_eq!(started.id, 7);
        assert_eq!(started.worker_id, Some(3));
        assert!(started.started_at.is_some());
        assert!(started.completed_at.is_none());

        let execution_time = running.complete();

        let completed = registry.get_metadata(7).unwrap();
        assert!(completed.completed_at.is_some());
        assert!(completed.execution_duration().is_some());
        assert_eq!(completed.execution_duration(), Some(execution_time));
        assert!(registry.is_completed(7));
    }

    #[test]
    fn register_next_task_returns_id_and_lifecycle_token() {
        let mut registry = TaskRegistry::new();
        let (task_id, lifecycle) = registry.register_next_task();

        let running = lifecycle.start(2);
        let execution_time = running.complete();

        let metadata = registry.get_metadata(task_id).unwrap();
        assert_eq!(metadata.id, task_id);
        assert_eq!(metadata.worker_id, Some(2));
        assert!(metadata.started_at.is_some());
        assert!(metadata.completed_at.is_some());
        assert_eq!(metadata.execution_duration(), Some(execution_time));
    }

    #[test]
    fn running_lifecycle_token_completes_on_drop() {
        let mut registry = TaskRegistry::new();
        let lifecycle = registry.register_task_with_id(8);

        drop(lifecycle.start(1));

        assert!(registry.is_completed(8));
    }

    #[test]
    fn lifecycle_blocks_preserve_sparse_metadata_and_cleanup_completed_slots() {
        let mut registry = TaskRegistry::new();
        let first_id = (TASK_STATE_BLOCK_SIZE - 1) as u64;
        let second_id = TASK_STATE_BLOCK_SIZE as u64;

        let first = registry.register_task_with_id(first_id).start(0);
        first.complete();

        let second = registry.register_task_with_id(second_id).start(1);

        assert!(registry.is_completed(first_id));
        assert_eq!(registry.get_metadata(second_id).unwrap().worker_id, Some(1));
        assert_eq!(registry.active_count(), 1);
        assert_eq!(registry.completed_count(), 1);

        registry.cleanup_completed(Duration::ZERO);

        assert!(registry.get_metadata(first_id).is_none());
        assert!(registry.get_metadata(second_id).is_some());

        second.complete();
        assert!(registry.is_completed(second_id));
    }

    #[test]
    fn cleanup_completed_releases_empty_trailing_blocks() {
        let mut registry = TaskRegistry::new();
        let first_id = (TASK_STATE_BLOCK_SIZE - 1) as u64;
        let second_id = TASK_STATE_BLOCK_SIZE as u64;

        registry.register_task_with_id(first_id).start(0).complete();
        registry
            .register_task_with_id(second_id)
            .start(0)
            .complete();
        assert_eq!(registry.blocks.len(), 2);

        registry.cleanup_completed(Duration::ZERO);

        assert!(registry.blocks.is_empty());
        assert!(registry.get_metadata(first_id).is_none());
        assert!(registry.get_metadata(second_id).is_none());
    }

    #[test]
    #[should_panic(expected = "task ID must not be re-registered while active")]
    fn lifecycle_registry_rejects_active_id_reuse() {
        let mut registry = TaskRegistry::new();
        let _running = registry.register_task_with_id(21).start(0);

        let _duplicate = registry.register_task_with_id(21);
    }
}

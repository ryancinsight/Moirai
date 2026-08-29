#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

#[cfg(test)]
#[allow(clippy::module_inception)]
mod tests {
    use std::{sync::Arc, time::Duration};

    use moirai_core::Priority;

    use super::super::registry::{CancelOutcome, TaskRegistry};
    use super::super::state::{task_location, PRIORITY_FROM_INDEX, TASK_STATE_BLOCK_SIZE};

    #[test]
    fn lifecycle_token_records_started_and_completed_metadata() {
        let registry = TaskRegistry::new();
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
        let registry = TaskRegistry::new();
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
    fn lifecycle_token_keeps_its_dense_block_alive_after_registry_drop() {
        let (lifecycle, block) = {
            let registry = TaskRegistry::new();
            let (task_id, lifecycle) = registry.register_next_task();
            let (block_index, _) = task_location(task_id);
            let block =
                Arc::downgrade(&registry.blocks.read().expect("test registry lock")[block_index]);
            assert_eq!(block.strong_count(), 2);
            (lifecycle, block)
        };

        assert_eq!(block.strong_count(), 1);
        lifecycle.start(5).complete();
        assert!(
            block.upgrade().is_none(),
            "the token must release its sole block owner after completion"
        );
    }

    #[test]
    fn cleanup_retains_completed_state_until_running_token_retires() {
        let registry = TaskRegistry::new();
        let (task_id, lifecycle) = registry.register_next_task();
        let running = lifecycle.start(3);

        // A registry observer can publish completion independently, but the
        // running token still owns lifecycle access until it is consumed.
        registry.mark_completed(task_id);
        registry.cleanup_completed(Duration::ZERO);
        assert!(registry.get_metadata(task_id).is_some());

        running.complete();
        registry.cleanup_completed(Duration::ZERO);
        assert!(registry.get_metadata(task_id).is_none());
    }

    #[test]
    fn unstarted_lifecycle_token_drop_publishes_rejection_completion() {
        let registry = TaskRegistry::new();
        let (task_id, lifecycle) = registry.register_next_task();

        drop(lifecycle);

        let metadata = registry
            .get_metadata(task_id)
            .expect("registered task metadata must remain readable");
        assert_eq!(metadata.started_at, None);
        assert!(!metadata.cancelled);
        assert!(metadata.completed_at.is_some());
        assert_eq!(registry.active_count(), 0);
        assert_eq!(registry.completed_count(), 1);
    }

    #[test]
    fn running_lifecycle_token_completes_on_drop() {
        let registry = TaskRegistry::new();
        let lifecycle = registry.register_task_with_id(8);

        drop(lifecycle.start(1));

        assert!(registry.is_completed(8));
    }

    #[test]
    fn lifecycle_blocks_preserve_sparse_metadata_and_cleanup_completed_slots() {
        let registry = TaskRegistry::new();
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
        let registry = TaskRegistry::new();
        let first_id = (TASK_STATE_BLOCK_SIZE - 1) as u64;
        let second_id = TASK_STATE_BLOCK_SIZE as u64;

        registry.register_task_with_id(first_id).start(0).complete();
        registry
            .register_task_with_id(second_id)
            .start(0)
            .complete();
        assert_eq!(registry.blocks.read().expect("test registry lock").len(), 2);

        registry.cleanup_completed(Duration::ZERO);

        assert!(registry
            .blocks
            .read()
            .expect("test registry lock")
            .is_empty());
        assert!(registry.get_metadata(first_id).is_none());
        assert!(registry.get_metadata(second_id).is_none());
    }

    #[test]
    fn priority_index_round_trips() {
        // PRIORITY_FROM_INDEX is the inverse of Priority::index (SSOT pair).
        for priority in [
            Priority::Low,
            Priority::Normal,
            Priority::High,
            Priority::Critical,
        ] {
            assert_eq!(PRIORITY_FROM_INDEX[priority.index()], priority);
        }
        assert_eq!(Priority::Low.index(), 0);
        assert_eq!(Priority::Normal.index(), 1);
        assert_eq!(Priority::High.index(), 2);
        assert_eq!(Priority::Critical.index(), 3);
    }

    #[test]
    fn lifecycle_token_records_spawn_priority() {
        let registry = TaskRegistry::new();
        let (task_id, lifecycle) = registry.register_next_task();
        lifecycle.set_priority(Priority::Critical);

        assert_eq!(
            registry.get_metadata(task_id).unwrap().priority,
            Priority::Critical
        );

        lifecycle.start(0).complete();
        // The recorded priority survives completion.
        assert_eq!(
            registry.get_metadata(task_id).unwrap().priority,
            Priority::Critical
        );
    }

    #[test]
    fn cancel_before_start_skips_body_and_completes_as_cancelled() {
        let registry = TaskRegistry::new();
        let (task_id, lifecycle) = registry.register_next_task();

        assert_eq!(
            registry.request_cancel(task_id),
            Some(CancelOutcome::Requested)
        );

        // The job-start gate observes the request and never yields a running token.
        assert!(lifecycle.start_unless_cancelled(4).is_none());

        let metadata = registry.get_metadata(task_id).unwrap();
        assert!(metadata.cancelled);
        assert!(metadata.completed_at.is_some());
        assert!(metadata.started_at.is_none());
        assert!(registry.is_completed(task_id));
    }

    #[test]
    fn cancel_after_completion_reports_already_completed() {
        let registry = TaskRegistry::new();
        let (task_id, lifecycle) = registry.register_next_task();
        lifecycle.start(0).complete();

        assert_eq!(
            registry.request_cancel(task_id),
            Some(CancelOutcome::AlreadyCompleted)
        );
        assert!(!registry.get_metadata(task_id).unwrap().cancelled);
    }

    #[test]
    fn cancel_unknown_task_returns_none() {
        let registry = TaskRegistry::new();
        assert_eq!(registry.request_cancel(999), None);
    }

    /// Concurrent registration must hand out distinct ids and distinct slots.
    ///
    /// The registry is shared as `Arc<TaskRegistry>` without an outer mutex, so
    /// this is the contract that replaced it: many producers registering at
    /// once, across the 1024-id block boundary so the directory has to grow
    /// underneath them. A lost `fetch_add`, a torn directory growth, or two
    /// producers claiming one slot would show up as a duplicate id or a
    /// missing metadata entry.
    #[test]
    fn concurrent_registration_yields_distinct_ids_across_block_growth() {
        use std::collections::BTreeSet;
        use std::sync::Mutex;

        const PRODUCERS: usize = 8;
        const PER_PRODUCER: usize = 400; // 3200 ids: spans several 1024-id blocks

        let registry = Arc::new(TaskRegistry::new());
        let ids: Mutex<Vec<u64>> = Mutex::new(Vec::with_capacity(PRODUCERS * PER_PRODUCER));

        std::thread::scope(|scope| {
            for _ in 0..PRODUCERS {
                let registry = Arc::clone(&registry);
                let ids = &ids;
                scope.spawn(move || {
                    let mut local = Vec::with_capacity(PER_PRODUCER);
                    for _ in 0..PER_PRODUCER {
                        local.push(registry.register_task());
                    }
                    ids.lock().expect("test id collector").extend(local);
                });
            }
        });

        let ids = ids.into_inner().expect("test id collector");
        assert_eq!(ids.len(), PRODUCERS * PER_PRODUCER);
        let unique: BTreeSet<u64> = ids.iter().copied().collect();
        assert_eq!(
            unique.len(),
            ids.len(),
            "concurrent registration handed out a duplicate task id"
        );
        for id in ids {
            assert!(
                registry.get_metadata(id).is_some(),
                "registered task {id} has no slot"
            );
        }
    }

    #[test]
    #[should_panic(expected = "task ID must not be re-registered while active")]
    fn lifecycle_registry_rejects_active_id_reuse() {
        let registry = TaskRegistry::new();
        let _running = registry.register_task_with_id(21).start(0);

        let _duplicate = registry.register_task_with_id(21);
    }
}

fn direct_registry_lifecycle(registry: &TaskRegistry) -> usize {
    let id = registry.register_task();
    registry.mark_started(id, 0);
    registry.mark_completed(id);

    verify_ready_value(READY_VALUE)
}

#[cfg(feature = "registry-diagnostics")]
fn direct_registry_token_lifecycle(registry: &TaskRegistry) -> usize {
    let duration = registry.diagnostic_register_next_and_complete_with_token();
    black_box(duration.as_nanos() as usize)
}

#[cfg(feature = "registry-diagnostics")]
fn direct_registry_retained_token_lifecycle(registry: &TaskRegistry) -> usize {
    let duration = registry.diagnostic_register_next_and_complete_with_retained_token();
    black_box(duration.as_nanos() as usize)
}

fn registry_shared_acquire_only(registry: &TaskRegistry) -> usize {
    black_box(registry.diagnostic_directory_shared_acquire());

    verify_ready_value(READY_VALUE)
}

#[cfg(feature = "registry-diagnostics")]
fn registry_block_lookup(registry: &TaskRegistry) -> usize {
    black_box(registry.diagnostic_block_lookup() as usize)
}

#[cfg(feature = "registry-diagnostics")]
fn registry_slot_initialize(registry: &TaskRegistry) -> usize {
    black_box(registry.diagnostic_slot_initialize() as usize)
}

#[cfg(feature = "registry-diagnostics")]
fn registry_lifecycle_timestamp_publication() -> usize {
    let duration = TaskRegistry::diagnostic_lifecycle_timestamp_publication();
    black_box(duration.as_nanos() as usize)
}

#[cfg(feature = "registry-diagnostics")]
fn registry_task_state_construct() -> usize {
    black_box(TaskRegistry::diagnostic_task_state_construct())
}

#[cfg(feature = "registry-diagnostics")]
fn registry_mark_started_existing_slot(registry: &TaskRegistry, task_id: u64) -> usize {
    black_box(registry.diagnostic_mark_started(task_id, 0) as usize)
}

#[cfg(feature = "registry-diagnostics")]
fn registry_mark_completed_existing_slot(
    registry: &TaskRegistry,
    task_id: u64,
    started_after_ns: u64,
) -> usize {
    let duration = registry.diagnostic_mark_completed_since(task_id, started_after_ns);
    black_box(duration.as_nanos() as usize)
}

#[cfg(feature = "registry-diagnostics")]
fn registry_elapsed_nanos_since_origin(origin: Instant) -> usize {
    black_box(elapsed_nanos_since(origin))
}

#[cfg(feature = "registry-diagnostics")]
fn registry_start_release_publication(
    started_after_ns: &AtomicUsize,
    worker_id: &AtomicUsize,
) -> usize {
    let offset = black_box(READY_VALUE);
    started_after_ns.store(offset, Ordering::Release);
    worker_id.store(black_box(BLOCKING_NORMAL_WORKER), Ordering::Release);
    black_box(offset)
}

#[cfg(feature = "registry-diagnostics")]
fn registry_completion_release_publication(completed_after_ns: &AtomicUsize) -> usize {
    let offset = black_box(READY_VALUE);
    completed_after_ns.store(offset, Ordering::Release);
    black_box(offset)
}

#[cfg(feature = "registry-diagnostics")]
fn registry_duration_offset_math() -> usize {
    let started_after_ns = black_box(READY_VALUE);
    let completed_after_ns = black_box(READY_VALUE + CAPTURED_READY_VALUE);
    debug_assert!(
        completed_after_ns >= started_after_ns,
        "diagnostic completion offset must not precede start offset"
    );
    black_box(completed_after_ns - started_after_ns)
}

fn shared_registry_register(registry: &TaskRegistry) -> usize {
    let id = registry.register_task();

    black_box(id as usize)
}

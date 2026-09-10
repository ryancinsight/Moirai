#[test]
fn scheduled_job_storage_keeps_inline_capacity_without_slot_alignment() {
    let source = read_benchmark("../moirai-executor/src/schedule/job/mod.rs");

    for required in [
        "const INLINE_JOB_WORDS: usize = 14",
        "#[repr(C)]",
        "pub(crate) struct ScheduledJob",
        "job: InlineJob",
        "InlineJob::new(boxed_job(task))",
        "fn boxed_job<F>(task: F) -> impl FnOnce(usize) + Send",
        "drop_consumed",
        "inline_job_uses_natural_alignment_with_same_capacity",
        "maximum_inline_capacity_job_uses_inline_storage",
        "over_aligned_job_uses_typed_boxed_trampoline",
        "oversized_job_uses_boxed_inline_trampoline",
        "inline_job_drops_capture_once_before_and_after_execution",
        "oversized_job_drops_capture_once_before_and_after_execution",
        "over_aligned_job_drops_capture_once_before_and_after_execution",
    ] {
        assert!(
            source.contains(required),
            "scheduled job storage must retain {required}"
        );
    }

    for prohibited in [
        "#[repr(C, align(64))]",
        "pub(crate) enum ScheduledJob",
        "struct HeapJob",
        "Box<dyn FnOnce",
        "execute_heap",
        "drop_heap",
    ] {
        assert!(
            !source.contains(prohibited),
            "scheduled job storage must not reintroduce {prohibited}"
        );
    }
}

#[test]
fn scheduler_scope_buffers_inline_scheduled_jobs() {
    let source = read_benchmark("../moirai-executor/src/schedule/runtime/mod.rs");

    for required in [
        "jobs: RefCell<Vec<ScheduledJob>>",
        "self.state().register_task()",
        "let complete = move |succeeded: bool|",
        "ScheduledJob::new_scoped_with_completion(task, complete)",
        "self.jobs.borrow_mut().push(job)",
        "fn schedule_single(&self, job: ScheduledJob)",
        "let spread_start = self",
        ".then(|| self.scheduler.select_worker::<C>(self.priority, None))",
        "let locality_hint = self.locality_hint.or_else(||",
        "start.wrapping_add(chunk_index) % worker_count",
        // Admission leaves a refused job in the caller's slot so `flush` can run
        // it on the calling lane instead of dropping it (ISSUE-221).
        ".admit_job::<C>(self.priority, locality_hint, &mut job)",
        "fn run_if_refused",
        "self.scheduler.record_admission_caller_run()",
        "job.execute(self.scheduler.caller_lane_id())",
        "fn schedule_chunk(",
        "jobs: Vec<ScheduledJob>",
        "let _ = job.execute(worker_id)",
    ] {
        assert!(
            source.contains(required),
            "scheduler scope must retain inline scheduled-job buffering through {required}"
        );
    }

    for prohibited in ["ScopedJobFn", "Box<dyn FnOnce", "Box::new(move |_|"] {
        assert!(
            !source.contains(prohibited),
            "scheduler scope must not reintroduce dynamic scoped-job dispatch through {prohibited}"
        );
    }
}

#[test]
fn rejected_scheduler_inline_handoff_candidate_stays_removed() {
    let executor_manifest = read_benchmark("../moirai-executor/Cargo.toml");
    let benchmark_manifest = read_benchmark("Cargo.toml");
    let runtime_source = read_benchmark("../moirai-executor/src/schedule/runtime/mod.rs");

    for prohibited in [
        "scheduler-inline-handoff",
        "InlineHandoffSlot",
        "HANDOFF_EMPTY",
        "HANDOFF_WRITING",
        "HANDOFF_READY",
        "try_publish_handoff",
        "take_handoff(&self) -> Option<ScheduledJob>",
    ] {
        assert!(
            !executor_manifest.contains(prohibited)
                && !benchmark_manifest.contains(prohibited)
                && !runtime_source.contains(prohibited),
            "rejected scheduler inline handoff candidate must stay removed: {prohibited}"
        );
    }
}

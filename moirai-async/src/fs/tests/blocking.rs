//! File operations run on the file-system pool, never inside `poll`: the
//! executor keeps running while one is held, a dropped operation releases its
//! slot at every stage, and cancellation keeps stream order.

use super::test_path;
use crate::blocking::job_lifecycle::{self, Operation, Subject};
use crate::blocking::test_hooks;
use crate::executor::AsyncExecutor;
use crate::fs::{File, FileOpenOptions, metadata, pool};
use futures::executor::block_on;
use std::future::Future;
use std::io::SeekFrom;
use std::sync::{Arc, mpsc};
use std::task::Context;
use std::time::Duration;

/// Allowance for the executor thread to run a ready task on a loaded CI
/// runner; a task blocked behind a held file operation waits the full
/// `test_hooks::STAGE_LIMIT` instead.
const SCHEDULING_MARGIN: Duration = Duration::from_secs(1);

/// One path operation: `metadata` of the temp directory, which always exists.
fn stat_temp_dir() -> Operation {
    Box::pin(async { metadata(std::env::temp_dir()).await.map(drop) })
}

const FILE_SYSTEM: Subject = Subject {
    pool,
    operation: stat_temp_dir,
};

#[test]
fn executor_runs_other_tasks_while_a_file_operation_is_held() {
    let _exclusive = test_hooks::exclusive();
    let hooks = pool().hooks();
    let baseline = hooks.progress();
    let executor = Arc::new(AsyncExecutor::new().expect("async executor must start"));
    let runner_executor = Arc::clone(&executor);
    let runner = std::thread::spawn(move || runner_executor.run());

    hooks.set_gate_closed(true);
    let held = executor.spawn(async { metadata(std::env::temp_dir()).await.map(|m| m.is_dir()) });
    hooks.wait_until(|progress| progress.started == baseline.started + 1);

    let (sender, answered) = mpsc::channel();
    let other = executor.spawn(async move {
        sender.send(6_u32 * 7).expect("the test awaits the answer");
    });
    let answer = answered.recv_timeout(SCHEDULING_MARGIN);
    let still_held = hooks.progress().live;
    hooks.set_gate_closed(false);

    assert_eq!(
        answer,
        Ok(42),
        "a task queued behind a held file operation did not run"
    );
    assert_eq!(still_held, 1, "the file operation completed before release");
    block_on(other);
    assert!(block_on(held).expect("metadata must succeed"));
    executor.stop().expect("executor stop must wake reactor");
    runner
        .join()
        .expect("executor thread must not panic")
        .expect("executor run must stop cleanly");
}

#[test]
fn dropped_admission_waiter_returns_no_slot() {
    job_lifecycle::dropped_admission_waiter_returns_no_slot(FILE_SYSTEM);
}

#[test]
fn dropped_queued_operation_is_skipped() {
    job_lifecycle::dropped_queued_job_is_skipped(FILE_SYSTEM);
}

#[test]
fn dropped_running_operation_releases_its_slot_on_return() {
    job_lifecycle::dropped_running_job_releases_its_slot_on_return(FILE_SYSTEM);
}

#[test]
fn panicking_operations_fail_alone() {
    job_lifecycle::panicking_jobs_fail_alone(FILE_SYSTEM);
}

#[test]
fn panicking_waker_leaves_the_file_system_pool_serving() {
    job_lifecycle::panicking_waker_leaves_the_worker_serving(FILE_SYSTEM);
}

/// Poll `future` once while the gate holds its job, then drop it mid-flight.
pub(super) fn abandon_in_flight<F: Future>(future: F, started: usize) {
    let mut future = Box::pin(future);
    let waker = futures::task::noop_waker();
    assert!(
        future
            .as_mut()
            .poll(&mut Context::from_waker(&waker))
            .is_pending()
    );
    pool()
        .hooks()
        .wait_until(|progress| progress.started == started + 1);
}

#[test]
fn dropped_read_delivers_its_bytes_to_the_next_read() {
    let _exclusive = test_hooks::exclusive();
    let path = test_path("dropped-read.bin");
    std::fs::write(&path, b"abcdef").expect("source write must succeed");
    let mut file = block_on(File::open(&path)).expect("open must succeed");
    let hooks = pool().hooks();
    let baseline = hooks.progress();

    hooks.set_gate_closed(true);
    let mut dropped = [0_u8; 4];
    abandon_in_flight(file.read(&mut dropped), baseline.started);
    hooks.set_gate_closed(false);

    let mut first = [0_u8; 2];
    assert_eq!(
        block_on(file.read(&mut first)).expect("read must succeed"),
        2
    );
    assert_eq!(&first, b"ab", "the dropped read's bytes come first");
    let mut rest = [0_u8; 8];
    let read = block_on(file.read(&mut rest)).expect("read must succeed");
    assert_eq!(&rest[..read], b"cd", "then the rest of the dropped read");
    let read = block_on(file.read(&mut rest)).expect("read must succeed");
    assert_eq!(&rest[..read], b"ef", "then the file past it");
    assert_eq!(
        dropped, [0; 4],
        "a dropped read writes nothing to its buffer"
    );

    drop(file);
    std::fs::remove_file(&path).expect("test file cleanup must succeed");
}

#[test]
fn write_and_seek_after_a_dropped_read_start_at_the_undelivered_bytes() {
    let _exclusive = test_hooks::exclusive();
    let path = test_path("rewind.bin");
    std::fs::write(&path, b"abcdef").expect("source write must succeed");
    let mut file = block_on(File::open_with_options(
        &path,
        FileOpenOptions::read_write(),
    ))
    .expect("open must succeed");
    let hooks = pool().hooks();

    // The dropped read consumes "abcd" from the OS cursor; none of it reaches
    // a caller, so the write must land at offset 0.
    let baseline = hooks.progress();
    hooks.set_gate_closed(true);
    let mut dropped = [0_u8; 4];
    abandon_in_flight(file.read(&mut dropped), baseline.started);
    hooks.set_gate_closed(false);
    block_on(file.write_all(b"XY")).expect("write must succeed");
    assert_eq!(
        block_on(file.stream_position()).expect("position must succeed"),
        2
    );

    // Same for a relative seek.
    let baseline = hooks.progress();
    hooks.set_gate_closed(true);
    abandon_in_flight(file.read(&mut dropped), baseline.started);
    hooks.set_gate_closed(false);
    assert_eq!(
        block_on(file.seek(SeekFrom::Current(1))).expect("seek must succeed"),
        3
    );

    drop(file);
    assert_eq!(
        std::fs::read(&path).expect("read back must succeed"),
        b"XYcdef"
    );
    std::fs::remove_file(&path).expect("test file cleanup must succeed");
}

#[test]
fn dropped_write_still_lands_in_call_order() {
    let _exclusive = test_hooks::exclusive();
    let path = test_path("dropped-write.bin");
    let mut file = block_on(File::create(&path)).expect("create must succeed");
    let hooks = pool().hooks();
    let baseline = hooks.progress();

    hooks.set_gate_closed(true);
    abandon_in_flight(file.write(b"first-"), baseline.started);
    hooks.set_gate_closed(false);
    block_on(file.write_all(b"second")).expect("write must succeed");
    drop(file);

    assert_eq!(
        std::fs::read(&path).expect("read back must succeed"),
        b"first-second"
    );
    std::fs::remove_file(&path).expect("test file cleanup must succeed");
}

#[test]
fn poll_based_read_and_write_run_on_the_pool() {
    use crate::io::{AsyncRead, AsyncWrite};

    let path = test_path("poll-based.bin");
    let mut file = block_on(File::open_with_options(
        &path,
        FileOpenOptions::read_write_truncate(),
    ))
    .expect("open must succeed");
    let written = block_on(std::future::poll_fn(|cx| {
        std::pin::Pin::new(&mut file).poll_write(cx, b"pollable")
    }))
    .expect("poll_write must succeed");
    assert_eq!(written, 8);
    block_on(std::future::poll_fn(|cx| {
        std::pin::Pin::new(&mut file).poll_flush(cx)
    }))
    .expect("flush reports the queued write");
    block_on(file.seek(SeekFrom::Start(4))).expect("seek must succeed");
    let mut tail = [0_u8; 8];
    let read = block_on(std::future::poll_fn(|cx| {
        std::pin::Pin::new(&mut file).poll_read(cx, &mut tail)
    }))
    .expect("poll_read must succeed");
    assert_eq!(&tail[..read], b"able");

    drop(file);
    std::fs::remove_file(&path).expect("test file cleanup must succeed");
}

#[test]
fn write_queued_when_its_file_drops_still_lands() {
    let _exclusive = test_hooks::exclusive();
    let path = test_path("queued-write.bin");
    let mut file = block_on(File::create(&path)).expect("create must succeed");
    let hooks = pool().hooks();
    let baseline = hooks.progress();

    // Hold every worker so the write stays queued behind them.
    hooks.set_gate_closed(true);
    let held: Vec<Operation> = (0..crate::fs::FS_WORKERS)
        .map(|_| {
            let mut operation = stat_temp_dir();
            let waker = futures::task::noop_waker();
            assert!(
                operation
                    .as_mut()
                    .poll(&mut Context::from_waker(&waker))
                    .is_pending()
            );
            operation
        })
        .collect();
    hooks.wait_until(|progress| progress.started == baseline.started + crate::fs::FS_WORKERS);
    {
        let mut write = Box::pin(file.write(b"kept"));
        let waker = futures::task::noop_waker();
        assert!(
            write
                .as_mut()
                .poll(&mut Context::from_waker(&waker))
                .is_pending()
        );
    }
    drop(file);
    hooks.set_gate_closed(false);
    held.into_iter()
        .for_each(|operation| block_on(operation).expect("metadata must succeed"));

    let settled = crate::fs::FS_WORKERS + 1;
    hooks.wait_until(|progress| progress.disposed == baseline.disposed + settled);
    assert_eq!(
        std::fs::read(&path).expect("read back must succeed"),
        b"kept",
        "a submitted write must land after its handle is dropped"
    );
    std::fs::remove_file(&path).expect("test file cleanup must succeed");
}

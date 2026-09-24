//! Observers see queued writes, deferred errors reach the next caller, and
//! bytes carried by a dropped read stay in stream order on every path.

use super::blocking::abandon_in_flight;
use super::test_path;
use crate::blocking::test_hooks;
use crate::fs::file::request::test_hooks as stream_hooks;
use crate::fs::{File, FileOpenOptions, pool};
use crate::io::{AsyncReadAt, AsyncWrite};
use futures::executor::block_on;
use std::future::{Future, poll_fn};
use std::io::SeekFrom;
use std::pin::Pin;
use std::task::Context;

fn poll_write(file: &mut File, bytes: &[u8]) -> usize {
    block_on(poll_fn(|cx| Pin::new(&mut *file).poll_write(cx, bytes)))
        .expect("a queued write reports its length")
}

/// Opens `path` for a test that counts free admission slots, returning only
/// after the open job has given its slot back. A job replies before it
/// releases its slot, so without this wait a count taken right after the
/// open can still include the open's slot on a loaded host.
fn open_settled(path: &std::path::Path) -> File {
    let hooks = pool().hooks();
    let baseline = hooks.progress();
    let file = block_on(File::open_with_options(
        path,
        FileOpenOptions::read_write_truncate(),
    ))
    .expect("open must succeed");
    hooks.wait_until(|progress| progress.disposed > baseline.disposed);
    file
}

fn poll_flush(file: &mut File) -> std::io::Result<()> {
    block_on(poll_fn(|cx| Pin::new(&mut *file).poll_flush(cx)))
}

/// Rounds of write-then-observe. Without ordering, the observer's job lands
/// on another worker ahead of the queued write in a fraction of rounds (2 in
/// 2,000 measured); the gated test below forces that case every time.
const ROUNDS: usize = 2_000;

#[test]
fn positioned_read_waits_for_a_queued_write() {
    let _exclusive = test_hooks::exclusive();
    let path = test_path("observe-gated.bin");
    let mut file = open_settled(&path);
    let hooks = pool().hooks();
    let baseline = hooks.progress();

    hooks.set_gate_closed(true);
    assert_eq!(poll_write(&mut file, b"fresh"), 5);
    hooks.wait_until(|progress| progress.started == baseline.started + 1);
    let mut observed = [0_u8; 5];
    let mut read_at = Box::pin(file.read_at(0, &mut observed));
    let waker = futures::task::noop_waker();
    assert!(
        read_at
            .as_mut()
            .poll(&mut Context::from_waker(&waker))
            .is_pending()
    );
    let free = pool().free_admissions();
    hooks.set_gate_closed(false);
    block_on(read_at).expect("positioned read must succeed");

    assert_eq!(
        free,
        pool().admissions() - 1,
        "the positioned read was queued beside the held write instead of after it"
    );
    assert_eq!(&observed, b"fresh");
    drop(file);
    std::fs::remove_file(&path).expect("test file cleanup must succeed");
}

#[test]
fn observers_always_see_queued_writes() {
    let _exclusive = test_hooks::exclusive();
    let path = test_path("observe-rounds.bin");
    let mut file = block_on(File::open_with_options(
        &path,
        FileOpenOptions::read_write_truncate(),
    ))
    .expect("open must succeed");
    for round in 0..ROUNDS {
        let bytes = u32::try_from(round)
            .expect("round count fits u32")
            .to_le_bytes();
        let offset = u64::try_from(round * bytes.len()).expect("offset fits u64");
        assert_eq!(poll_write(&mut file, &bytes), bytes.len());
        let mut observed = [0_u8; 4];
        block_on(file.read_at(offset, &mut observed)).expect("positioned read must succeed");
        assert_eq!(observed, bytes, "round {round} missed its own queued write");
        let length = block_on(file.metadata())
            .expect("metadata must succeed")
            .len();
        assert_eq!(
            length,
            offset + 4,
            "round {round} metadata missed the write"
        );
    }
    drop(file);
    std::fs::remove_file(&path).expect("test file cleanup must succeed");
}

#[test]
fn write_behind_failure_surfaces_on_flush() {
    let path = test_path("deferred-flush.bin");
    std::fs::write(&path, b"read-only").expect("source write must succeed");
    let mut file = block_on(File::open(&path)).expect("open must succeed");

    assert_eq!(poll_write(&mut file, b"x"), 1, "the write is queued");
    let failure = poll_flush(&mut file).expect_err("the queued write failed on a read-only handle");
    assert!(
        failure.raw_os_error().is_some(),
        "the flush reports the write's own OS error, got {failure}"
    );
    poll_flush(&mut file).expect("a reported error is not reported twice");

    drop(file);
    assert_eq!(
        std::fs::read(&path).expect("read back must succeed"),
        b"read-only"
    );
    std::fs::remove_file(&path).expect("test file cleanup must succeed");
}

#[test]
fn abandoned_operation_failure_surfaces_on_the_next_call() {
    let _exclusive = test_hooks::exclusive();
    let path = test_path("deferred-next.bin");
    std::fs::write(&path, b"read-only").expect("source write must succeed");
    let mut file = block_on(File::open(&path)).expect("open must succeed");
    let hooks = pool().hooks();
    let baseline = hooks.progress();

    hooks.set_gate_closed(true);
    abandon_in_flight(file.write(b"x"), baseline.started);
    hooks.set_gate_closed(false);

    block_on(file.seek(SeekFrom::Start(0)))
        .expect_err("the abandoned write failed; the next call reports it and does not run");
    assert_eq!(
        block_on(file.seek(SeekFrom::Start(4))).expect("the call after that runs"),
        4
    );
    drop(file);
    std::fs::remove_file(&path).expect("test file cleanup must succeed");
}

#[test]
fn queued_write_after_a_dropped_read_starts_at_the_undelivered_bytes() {
    let _exclusive = test_hooks::exclusive();
    let path = test_path("poll-write-rewind.bin");
    std::fs::write(&path, b"abcdef").expect("source write must succeed");
    let mut file = block_on(File::open_with_options(
        &path,
        FileOpenOptions::read_write(),
    ))
    .expect("open must succeed");
    let hooks = pool().hooks();
    let baseline = hooks.progress();

    hooks.set_gate_closed(true);
    let mut dropped = [0_u8; 4];
    abandon_in_flight(file.read(&mut dropped), baseline.started);
    hooks.set_gate_closed(false);
    assert_eq!(poll_write(&mut file, b"XY"), 2);
    poll_flush(&mut file).expect("the queued write must succeed");

    drop(file);
    assert_eq!(
        std::fs::read(&path).expect("read back must succeed"),
        b"XYcdef"
    );
    std::fs::remove_file(&path).expect("test file cleanup must succeed");
}

#[test]
fn read_to_end_after_a_dropped_read_starts_with_its_bytes() {
    let _exclusive = test_hooks::exclusive();
    let path = test_path("read-to-end-carry.bin");
    std::fs::write(&path, b"abcdef").expect("source write must succeed");
    let mut file = block_on(File::open(&path)).expect("open must succeed");
    let hooks = pool().hooks();
    let baseline = hooks.progress();

    hooks.set_gate_closed(true);
    let mut dropped = [0_u8; 4];
    abandon_in_flight(file.read(&mut dropped), baseline.started);
    hooks.set_gate_closed(false);

    assert_eq!(
        block_on(file.read_to_end()).expect("read_to_end must succeed"),
        b"abcdef"
    );
    drop(file);
    std::fs::remove_file(&path).expect("test file cleanup must succeed");
}

#[test]
fn dropped_write_all_is_one_job_written_in_full() {
    let _exclusive = test_hooks::exclusive();
    let path = test_path("write-all-once.bin");
    let mut file = block_on(File::create(&path)).expect("create must succeed");
    let hooks = pool().hooks();
    let baseline = hooks.progress();
    let payload: Vec<u8> = (0..=255_u8).cycle().take(1 << 20).collect();

    hooks.set_gate_closed(true);
    abandon_in_flight(file.write_all(&payload), baseline.started);
    hooks.set_gate_closed(false);
    block_on(file.flush()).expect("the abandoned write_all must succeed");

    let progress = hooks.wait_until(|progress| progress.disposed == baseline.disposed + 1);
    assert_eq!(
        progress.started - baseline.started,
        1,
        "write_all must submit the whole buffer as one job"
    );
    drop(file);
    assert_eq!(
        std::fs::read(&path).expect("read back must succeed"),
        payload
    );
    std::fs::remove_file(&path).expect("test file cleanup must succeed");
}

#[test]
fn write_all_lands_in_full_when_single_writes_return_short() {
    let _exclusive = test_hooks::exclusive();
    let path = test_path("write-all-short.bin");
    let mut file = block_on(File::create(&path)).expect("create must succeed");
    let hooks = pool().hooks();
    let baseline = hooks.progress();

    // Three bytes per single write: a per-write loop needs four jobs for ten
    // bytes, and dropping its future after the first leaves "abc".
    stream_hooks::cap_single_writes(3);
    hooks.set_gate_closed(true);
    abandon_in_flight(file.write_all(b"abcdefghij"), baseline.started);
    hooks.set_gate_closed(false);
    let flushed = block_on(file.flush());
    stream_hooks::cap_single_writes(0);
    flushed.expect("the abandoned write_all must succeed");

    drop(file);
    assert_eq!(
        std::fs::read(&path).expect("read back must succeed"),
        b"abcdefghij",
        "a submitted write_all must land in full"
    );
    std::fs::remove_file(&path).expect("test file cleanup must succeed");
}

#[test]
fn observer_waits_until_the_write_job_has_finished() {
    let _exclusive = test_hooks::exclusive();
    let path = test_path("ticket-timing.bin");
    let mut file = open_settled(&path);
    let reached = stream_hooks::reached();

    // Hold the write after its syscall and before it releases its ticket.
    stream_hooks::set_hold_after_run(true);
    assert_eq!(poll_write(&mut file, b"fresh"), 5);
    stream_hooks::wait_reached(reached + 1);
    let mut observed = [0_u8; 5];
    let mut read_at = Box::pin(file.read_at(0, &mut observed));
    let waker = futures::task::noop_waker();
    let first_poll = read_at.as_mut().poll(&mut Context::from_waker(&waker));
    let free = pool().free_admissions();
    stream_hooks::set_hold_after_run(false);

    assert!(first_poll.is_pending());
    assert_eq!(
        free,
        pool().admissions() - 1,
        "the positioned read was submitted before the write job released its ticket"
    );
    block_on(read_at).expect("positioned read must succeed");
    assert_eq!(&observed, b"fresh");
    drop(file);
    std::fs::remove_file(&path).expect("test file cleanup must succeed");
}

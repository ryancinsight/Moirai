#![cfg_attr(test, allow(clippy::unwrap_used, reason = "test scope"))]

use super::error::IpcError;
use super::memory::SharedMemory;
use super::queue::{SendError, SharedQueue};

#[test]
fn test_shared_memory() {
    let name = "/moirai_test_shm";
    let size = 1024;

    // Create shared memory
    let mut shm1 = SharedMemory::create(name, size).unwrap();

    // Write some data
    let data = b"Hello, shared memory!";
    // SAFETY: `shm2` does not exist yet, so this handle is the only one.
    let bytes = unsafe { shm1.as_mut_slice() };
    bytes[..data.len()].copy_from_slice(data);

    // Open from another "process"
    let shm2 = SharedMemory::open(name, size).unwrap();

    // Read the data
    // SAFETY: nothing writes the segment after the copy above.
    assert_eq!(&unsafe { shm2.as_slice() }[..data.len()], data);
}

#[test]
fn open_of_missing_segment_reports_system_error() {
    let result = SharedMemory::open("/moirai_test_no_such_segment", 64);
    assert!(matches!(result, Err(IpcError::SystemError(_))));
}

#[test]
fn zero_size_segment_is_rejected() {
    let result = SharedMemory::create("/moirai_test_zero", 0);
    assert!(matches!(result, Err(IpcError::InvalidArgument)));
}

#[test]
fn open_larger_than_the_segment_is_rejected() {
    // A mapping must cover every byte `as_slice` hands out. POSIX `mmap` accepts
    // a length past the end of the object and leaves the surplus pages unbacked,
    // so reading them raises SIGBUS; `open` has to reject the oversized request
    // itself. (Win32 `MapViewOfFile` already refuses a view beyond the mapping.)
    let name = "/moirai_test_open_oversize";
    let _creator = SharedMemory::create(name, 4096).expect("create must succeed");

    // Far enough past the segment that the surplus is whole unbacked pages, not
    // the zero-filled tail of the segment's own final page.
    let result = SharedMemory::open(name, 1 << 20);

    // Which error depends on who caught it: POSIX `mmap` would have accepted the
    // oversized length, so `open` checks the segment itself with `fstat` and
    // reports `InvalidArgument`; on Windows the refusal comes from
    // `MapViewOfFile` and surfaces as the OS error, whose code is the
    // platform's, so only the variant is asserted. Either way the refusal is
    // the size check's, not a `NotFound` from a `create` that silently failed.
    match result {
        Err(IpcError::InvalidArgument) if cfg!(unix) => {}
        Err(IpcError::SystemError(_)) if cfg!(windows) => {}
        Err(other) => panic!(
            "the oversized open must be refused by the segment size check on unix or the mapping call on windows: got {other:?}"
        ),
        Ok(_) => panic!("opening a 4 KiB segment as 1 MiB must be rejected, not mapped"),
    }
}

#[test]
fn open_smaller_than_the_segment_maps_a_prefix() {
    // The size check rejects only mappings the segment cannot back; opening a
    // prefix stays legal, so the guard above is not simply refusing every open.
    let name = "/moirai_test_open_prefix";
    let mut creator = SharedMemory::create(name, 4096).expect("create must succeed");
    // SAFETY: the opener below is not created yet.
    let bytes = unsafe { creator.as_mut_slice() };
    bytes[..4].copy_from_slice(b"ipc!");

    let opener = SharedMemory::open(name, 1024).expect("prefix open must succeed");

    // SAFETY: the creator writes nothing after the copy above.
    let mapped = unsafe { opener.as_slice() };
    assert_eq!(mapped.len(), 1024);
    assert_eq!(&mapped[..4], b"ipc!");
}

#[cfg(all(unix, target_pointer_width = "64"))]
#[test]
fn segment_larger_than_off_t_is_rejected_before_creation() {
    let result = SharedMemory::create("/moirai_test_off_t_overflow", usize::MAX);
    assert!(matches!(result, Err(IpcError::InvalidArgument)));
}

#[test]
fn a_multi_page_segment_reads_back_across_its_whole_length() {
    // Sizing a segment is not backing it: on tmpfs `ftruncate` leaves the pages
    // sparse, so `create` reserves the store before handing out a mapping.
    // Writing every page is what would fault on a host that could not satisfy
    // that reservation, and reading it back is what fails if the new call
    // started refusing segments the store can perfectly well hold.
    let name = "/moirai_test_backed_pages";
    let size = 256 * 1024;
    // 251 is the largest prime below 256, so the pattern's period is coprime
    // with the page size: no two pages begin with the same byte, and a page
    // that silently read back as another's contents would not match.
    let marker = |index: usize| {
        u8::try_from(index % 251).expect("invariant: a remainder mod 251 fits in u8")
    };

    let mut segment = SharedMemory::create(name, size).expect("create must succeed");
    // SAFETY: `segment` is the only handle to this name.
    for (index, byte) in unsafe { segment.as_mut_slice() }.iter_mut().enumerate() {
        *byte = marker(index);
    }

    // SAFETY: as above; the writer borrow ended.
    let written = unsafe { segment.as_slice() };
    assert_eq!(written.len(), size);
    assert_eq!(
        written
            .iter()
            .enumerate()
            .position(|(index, &byte)| byte != marker(index)),
        None,
        "every byte of a created segment must read back what was written to it"
    );
}

/// The reservation `create` performs is classified from `posix_fallocate`'s
/// return value, and that classification is the part with no deterministic
/// end-to-end test: reaching the `ENOSPC` arm for real means exhausting the
/// host's tmpfs, which no test may do to the machine running it. Requesting an
/// absurd length does not substitute — it returns `ENOSPC` early only when
/// `/dev/shm` carries a size limit, and on a mount without one the kernel goes
/// away and tries to allocate it. What is deterministic is the decision itself,
/// over one representative code per outcome.
#[cfg(any(target_os = "linux", target_os = "android"))]
mod reservation_classification {
    use super::super::backing_store::{Reservation, classify_reservation};

    #[test]
    fn a_committed_reservation_is_the_only_success() {
        assert_eq!(classify_reservation(0), Reservation::Reserved);
    }

    #[test]
    fn a_shortage_fails_creation_instead_of_deferring_a_fault() {
        // The reason the call exists: a segment the store cannot hold has to be
        // refused here, not become a `SIGBUS` in whichever process writes it.
        for code in [libc::ENOSPC, libc::EFBIG] {
            assert_eq!(classify_reservation(code), Reservation::Failed(code));
        }
    }

    #[test]
    fn a_kernel_that_will_not_preallocate_leaves_creation_alone() {
        // tmpfs before Linux 3.5, or any object refusing fallocate. None of
        // these reports a shortage, and failing on them would break segments
        // that create fine today. `create` rejects a non-positive length before
        // reaching the call, so `EINVAL` cannot be our own argument error.
        for code in [libc::EOPNOTSUPP, libc::ENOSYS, libc::EINVAL] {
            assert_eq!(classify_reservation(code), Reservation::Unsupported);
        }
    }

    #[test]
    fn an_interrupted_reservation_is_reissued_rather_than_accepted() {
        // Reading `EINTR` as completion would leave behind exactly the sparse
        // segment the reservation was meant to eliminate.
        assert_eq!(classify_reservation(libc::EINTR), Reservation::Interrupted);
    }

    #[test]
    fn an_unrecognized_code_is_carried_through_as_a_failure() {
        // The default arm must not quietly widen into "unsupported": an `EIO` or
        // `EBADF` is a real failure and has to reach the caller intact.
        for code in [libc::EIO, libc::EBADF, libc::EPERM] {
            assert_eq!(classify_reservation(code), Reservation::Failed(code));
        }
    }
}

#[test]
fn test_shared_queue() {
    let name = "/moirai_test_queue";
    let capacity = 10;

    // Create queue
    let mut queue = SharedQueue::<u32>::create(name, capacity).unwrap();

    // Send some values
    queue.send(1).unwrap();
    queue.send(2).unwrap();
    queue.send(3).unwrap();

    // Receive values
    assert_eq!(queue.recv(), Ok(Some(1)));
    assert_eq!(queue.recv(), Ok(Some(2)));
    assert_eq!(queue.recv(), Ok(Some(3)));
    assert_eq!(queue.recv(), Ok(None));
}

#[test]
fn zero_capacity_queue_is_rejected() {
    // A zero capacity would make `% capacity` divide by zero on send/recv.
    let result = SharedQueue::<u32>::create("/moirai_test_queue_zero_cap", 0);
    assert!(matches!(result, Err(IpcError::InvalidArgument)));
}

#[test]
fn capacity_overflow_is_rejected_before_mapping() {
    // capacity * size_of::<u32>() overflows usize; the checked layout math must
    // reject it rather than request an undersized mapping and write out of bounds.
    let result = SharedQueue::<u32>::create("/moirai_test_queue_overflow", usize::MAX);
    assert!(matches!(result, Err(IpcError::InvalidArgument)));
}

#[test]
fn open_with_mismatched_capacity_is_rejected() {
    // Creator records capacity 20 in the segment header; a peer opening with a
    // smaller capacity maps a smaller, header-inclusive view, reads the recorded
    // capacity, and is rejected before touching the data region.
    let name = "/moirai_test_queue_cap_mismatch";
    let _creator = SharedQueue::<u32>::create(name, 20).expect("create must succeed");
    let result = SharedQueue::<u32>::open(name, 10);
    assert!(matches!(result, Err(IpcError::InvalidArgument)));
}

#[test]
fn open_with_matching_capacity_shares_data_across_handles() {
    // Two handles over the same segment: a value sent through one is received
    // through the other, proving the capacity header did not disturb the layout.
    let name = "/moirai_test_queue_shared_handles";
    let mut creator = SharedQueue::<u32>::create(name, 4).expect("create must succeed");
    let mut opener = SharedQueue::<u32>::open(name, 4).expect("open must succeed");

    creator.send(42).expect("send must succeed");
    creator.send(7).expect("send must succeed");
    assert_eq!(opener.recv(), Ok(Some(42)));
    assert_eq!(opener.recv(), Ok(Some(7)));
    assert_eq!(opener.recv(), Ok(None));
}

#[test]
fn full_queue_rejects_send_at_capacity() {
    let name = "/moirai_test_queue_full_boundary";
    let mut queue = SharedQueue::<u32>::create(name, 2).expect("create must succeed");
    queue.send(1).expect("first send must succeed");
    queue.send(2).expect("second send must succeed");
    assert_eq!(
        queue.send(3),
        Err(SendError::Full(3)),
        "send past capacity must return the value"
    );
}

#[test]
fn creating_a_taken_name_fails_and_leaves_the_live_segment_intact() {
    // A second `create` used to reach the live POSIX object and ftruncate it to
    // its own size, shrinking mappings already open on it.
    let name = "/moirai_test_create_exclusive";
    let mut first = SharedMemory::create(name, 8192).expect("first create must succeed");
    // SAFETY: no other handle has mapped the segment yet.
    let bytes = unsafe { first.as_mut_slice() };
    bytes[8191] = 0xA5;

    let second = SharedMemory::create(name, 4096);
    assert!(matches!(second, Err(IpcError::AlreadyExists)));

    // SAFETY: the failed create mapped nothing, so no writer exists.
    assert_eq!(unsafe { first.as_slice() }[8191], 0xA5);
    let reopened = SharedMemory::open(name, 8192).expect("the original size must survive");
    // SAFETY: nothing writes the segment.
    assert_eq!(unsafe { reopened.as_slice() }[8191], 0xA5);
}

#[test]
fn a_queue_name_can_be_created_once() {
    let name = "/moirai_test_queue_create_exclusive";
    let mut live = SharedQueue::<u32>::create(name, 4).expect("create must succeed");
    live.send(9).expect("send must succeed");

    let again = SharedQueue::<u32>::create(name, 4);
    assert!(matches!(again, Err(IpcError::AlreadyExists)));

    // The failed create must not have reset the live header.
    assert_eq!(live.recv(), Ok(Some(9)));
}

#[test]
fn a_second_sender_handle_is_refused_and_no_message_is_lost() {
    let name = "/moirai_test_queue_one_sender";
    let mut first = SharedQueue::<u32>::create(name, 4).expect("create must succeed");
    let mut second = SharedQueue::<u32>::open(name, 4).expect("open must succeed");

    first.send(1).expect("the first sender claims the endpoint");
    assert_eq!(second.send(2), Err(SendError::EndpointInUse(2)));
    first.send(3).expect("the claim holder keeps sending");

    assert_eq!(second.recv(), Ok(Some(1)));
    assert_eq!(second.recv(), Ok(Some(3)));
    assert_eq!(second.recv(), Ok(None));
}

#[test]
fn a_second_receiver_handle_is_refused() {
    let name = "/moirai_test_queue_one_receiver";
    let mut sender = SharedQueue::<u32>::create(name, 4).expect("create must succeed");
    let mut first = SharedQueue::<u32>::open(name, 4).expect("open must succeed");
    let mut second = SharedQueue::<u32>::open(name, 4).expect("open must succeed");

    sender.send(5).expect("send must succeed");
    assert_eq!(first.recv(), Ok(Some(5)));
    assert_eq!(second.recv(), Err(IpcError::EndpointInUse));
}

#[test]
fn dropping_a_handle_releases_its_endpoints() {
    let name = "/moirai_test_queue_endpoint_release";
    let mut receiver = SharedQueue::<u32>::create(name, 4).expect("create must succeed");

    let mut first = SharedQueue::<u32>::open(name, 4).expect("open must succeed");
    first.send(1).expect("the first sender claims the endpoint");
    let mut refused = SharedQueue::<u32>::open(name, 4).expect("open must succeed");
    assert_eq!(refused.send(2), Err(SendError::EndpointInUse(2)));

    drop(first);
    refused.send(3).expect("the released endpoint is claimable");

    assert_eq!(receiver.recv(), Ok(Some(1)));
    assert_eq!(receiver.recv(), Ok(Some(3)));
    assert_eq!(receiver.recv(), Ok(None));
}

#[test]
fn concurrent_senders_cannot_both_deliver() {
    // Two threads race to send distinct values through handles to one queue.
    // Exactly one wins the endpoint; every value it sends arrives exactly once
    // and in order, and the loser never enqueues.
    //
    // Both handles must be live when both threads attempt the claim. A claim is
    // released when its holder drops, so a thread delayed past the other's
    // entire send would claim a free endpoint instead of racing for a held one
    // — the property would never be exercised, and the late claimant would spin
    // on a full ring nobody drains. `attempted` holds each thread after its
    // first `send`, which is the last point at which either handle can drop, so
    // the winner is decided before either thread proceeds.
    const PER_SENDER: u32 = 500;
    let name = "/moirai_test_queue_racing_senders";
    let mut receiver = SharedQueue::<u32>::create(name, 8).expect("create must succeed");
    let senders = [
        SharedQueue::<u32>::open(name, 8).expect("open must succeed"),
        SharedQueue::<u32>::open(name, 8).expect("open must succeed"),
    ];
    let ready = std::sync::Barrier::new(2);
    let attempted = std::sync::Barrier::new(2);

    let outcomes: Vec<bool> = std::thread::scope(|scope| {
        let workers: Vec<_> = senders
            .into_iter()
            .enumerate()
            .map(|(lane, mut queue)| {
                let ready = &ready;
                let attempted = &attempted;
                scope.spawn(move || {
                    ready.wait();
                    let base = u32::try_from(lane).expect("two lanes") * PER_SENDER;
                    // The claim race. Neither handle has sent yet and neither
                    // can drop before the other has attempted, so exactly one
                    // of these two calls returns `Ok`.
                    let first = queue.send(base);
                    attempted.wait();
                    match first {
                        Ok(()) => {}
                        Err(SendError::EndpointInUse(_)) => return false,
                        // The other handle has sent at most this one value into
                        // a ring of eight, so it cannot be full.
                        Err(SendError::Full(value)) => {
                            unreachable!("the ring is empty at the first send, got Full({value})")
                        }
                        Err(SendError::Closed(_)) => unreachable!("queue is never closed"),
                    }

                    let mut sent = 1;
                    while sent < PER_SENDER {
                        match queue.send(base + sent) {
                            Ok(()) => sent += 1,
                            Err(SendError::Full(_)) => std::thread::yield_now(),
                            // The claim is held for this handle's lifetime, so
                            // it cannot be lost mid-send.
                            Err(SendError::EndpointInUse(_)) => {
                                unreachable!("this handle holds the sender endpoint")
                            }
                            Err(SendError::Closed(_)) => unreachable!("queue is never closed"),
                        }
                    }
                    true
                })
            })
            .collect();

        let mut received = Vec::new();
        while received.len() < PER_SENDER as usize {
            match receiver
                .recv()
                .expect("this handle owns the receiver endpoint")
            {
                Some(value) => received.push(value),
                None => std::thread::yield_now(),
            }
        }
        let outcomes = workers
            .into_iter()
            .map(|worker| worker.join().expect("sender thread must not panic"))
            .collect();

        let base = received[0] - received[0] % PER_SENDER;
        let expected: Vec<u32> = (base..base + PER_SENDER).collect();
        assert_eq!(
            received, expected,
            "the winner delivers every value once, in order"
        );
        outcomes
    });

    assert_eq!(
        outcomes.iter().filter(|&&won| won).count(),
        1,
        "exactly one handle may send on the queue"
    );
}

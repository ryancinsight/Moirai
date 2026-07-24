use super::error::IpcError;
use super::memory::SharedMemory;
use super::queue::SharedQueue;

#[test]
fn test_shared_memory() {
    let name = "/moirai_test_shm";
    let size = 1024;

    // Create shared memory
    let mut shm1 = SharedMemory::create(name, size).unwrap();

    // Write some data
    let data = b"Hello, shared memory!";
    shm1.as_mut_slice()[..data.len()].copy_from_slice(data);

    // Open from another "process"
    let shm2 = SharedMemory::open(name, size).unwrap();

    // Read the data
    assert_eq!(&shm2.as_slice()[..data.len()], data);
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

    assert!(
        result.is_err(),
        "opening a 4 KiB segment as 1 MiB must be rejected, not mapped"
    );

    // Which error depends on who caught it: POSIX `mmap` would have accepted the
    // oversized length, so `open` checks the segment itself and reports
    // `InvalidArgument`; on Windows the refusal comes from `MapViewOfFile` and
    // surfaces as the OS error.
    #[cfg(unix)]
    assert!(matches!(result, Err(IpcError::InvalidArgument)));
}

#[test]
fn open_smaller_than_the_segment_maps_a_prefix() {
    // The size check rejects only mappings the segment cannot back; opening a
    // prefix stays legal, so the guard above is not simply refusing every open.
    let name = "/moirai_test_open_prefix";
    let mut creator = SharedMemory::create(name, 4096).expect("create must succeed");
    creator.as_mut_slice()[..4].copy_from_slice(b"ipc!");

    let opener = SharedMemory::open(name, 1024).expect("prefix open must succeed");

    assert_eq!(opener.as_slice().len(), 1024);
    assert_eq!(&opener.as_slice()[..4], b"ipc!");
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
    for (index, byte) in segment.as_mut_slice().iter_mut().enumerate() {
        *byte = marker(index);
    }

    let written = segment.as_slice();
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
    use super::super::memory::{classify_reservation, Reservation};

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
    assert_eq!(queue.recv(), Some(1));
    assert_eq!(queue.recv(), Some(2));
    assert_eq!(queue.recv(), Some(3));
    assert_eq!(queue.recv(), None);
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
    assert_eq!(opener.recv(), Some(42));
    assert_eq!(opener.recv(), Some(7));
    assert_eq!(opener.recv(), None);
}

#[test]
fn full_queue_rejects_send_at_capacity() {
    let name = "/moirai_test_queue_full_boundary";
    let mut queue = SharedQueue::<u32>::create(name, 2).expect("create must succeed");
    queue.send(1).expect("first send must succeed");
    queue.send(2).expect("second send must succeed");
    assert_eq!(
        queue.send(3),
        Err(3),
        "send past capacity must return the value"
    );
}

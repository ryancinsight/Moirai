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
fn zero_size_segment_is_rejected_on_windows() {
    #[cfg(windows)]
    {
        let result = SharedMemory::create("/moirai_test_zero", 0);
        assert!(matches!(result, Err(IpcError::InvalidArgument)));
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

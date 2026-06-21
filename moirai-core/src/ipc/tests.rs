use super::memory::SharedMemory;
use super::queue::SharedQueue;
use super::error::IpcError;

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
    let queue = SharedQueue::<u32>::create(name, capacity).unwrap();

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

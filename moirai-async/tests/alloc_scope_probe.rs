//! Deterministic probe for the defect behind `task_poll_allocations.rs`.
//!
//! That test reads a **process-global** allocation counter across a window it
//! does not own exclusively, so another thread's allocations are booked as the
//! executor's poll-path cost and the assertion goes flaky. This probe
//! demonstrates the mechanism on demand instead of waiting for the race: it
//! allocates on a helper thread and shows that a process-global counter sees it
//! while a thread-local counter does not.

use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

static PROCESS_GLOBAL: AtomicUsize = AtomicUsize::new(0);
static HELPER_DONE: AtomicBool = AtomicBool::new(false);

// `const` initialization keeps this slot allocation-free and free of lazy
// registration, so reading it from inside `alloc` cannot recurse.
thread_local! {
    static THIS_THREAD: Cell<usize> = const { Cell::new(0) };
}

#[inline]
fn observe() {
    PROCESS_GLOBAL.fetch_add(1, Ordering::Relaxed);
    // `try_with` tolerates a destroyed TLS slot during thread teardown.
    let _ = THIS_THREAD.try_with(|count| count.set(count.get() + 1));
}

fn allocations_this_thread() -> usize {
    THIS_THREAD.with(Cell::get)
}

struct CountingAllocator;

// SAFETY: every operation delegates unchanged pointers and layouts to the
// system allocator; the counter observes calls without altering allocation.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        observe();
        // SAFETY: `layout` is forwarded unchanged to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        observe();
        // SAFETY: `layout` is forwarded unchanged to the system allocator.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: `pointer` and `layout` came from this delegated allocator.
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        observe();
        // SAFETY: the arguments are forwarded unchanged to the system
        // allocator that created `pointer`.
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

/// Spawn a helper that allocates 64 blocks, and return its handle. The spawn
/// itself happens before any measurement window opens.
fn spawn_allocating_helper() -> std::thread::JoinHandle<()> {
    std::thread::spawn(|| {
        let mut keep: Vec<Box<[u8; 1024]>> = Vec::with_capacity(64);
        for _ in 0..64 {
            keep.push(Box::new([0u8; 1024]));
        }
        std::hint::black_box(&keep);
        HELPER_DONE.store(true, Ordering::Release);
    })
}

/// A process-global counter books the helper thread's allocations.
#[test]
fn process_global_counter_books_another_threads_allocations() {
    let before = PROCESS_GLOBAL.load(Ordering::Relaxed);
    let helper = spawn_allocating_helper();
    while !HELPER_DONE.load(Ordering::Acquire) {
        std::hint::spin_loop();
    }
    let after = PROCESS_GLOBAL.load(Ordering::Relaxed);
    helper.join().expect("helper thread must join");

    let delta = after.saturating_sub(before);
    println!("process-global delta across a helper thread: {delta}");
    assert!(
        delta > 0,
        "a process-global counter must book the helper thread's allocations"
    );
}

/// A thread-local counter ignores the helper thread's allocations, which is
/// what makes the repaired `task_poll_allocations.rs` measurement sound.
#[test]
fn thread_local_counter_ignores_another_threads_allocations() {
    let helper = spawn_allocating_helper();
    let before = allocations_this_thread();
    while !HELPER_DONE.load(Ordering::Acquire) {
        std::hint::spin_loop();
    }
    let after = allocations_this_thread();
    helper.join().expect("helper thread must join");

    let delta = after.saturating_sub(before);
    println!("thread-local delta across a helper thread: {delta}");
    assert_eq!(
        after, before,
        "a thread-local counter must ignore the helper thread's allocations"
    );
}

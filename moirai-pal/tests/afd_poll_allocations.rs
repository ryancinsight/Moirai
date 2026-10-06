//! Allocation contract for arming and dispatching AFD polls.
//!
//! This binary installs a counting global allocator, so it stays isolated from
//! the ordinary unit-test harness. After one warm-up cycle (device open and
//! lazy process state), a full arm, readiness, dequeue, and dispatch cycle must
//! perform no heap allocation.

#![cfg(windows)]

use moirai_pal::Interest;
use moirai_pal::windows::afd::AfdPort;
use std::alloc::{GlobalAlloc, Layout, System};
use std::net::{TcpListener, TcpStream};
use std::os::windows::io::AsRawSocket;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Duration;

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every operation delegates unchanged pointers and layouts to the
// system allocator; the counter observes calls without altering allocation.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `layout` is forwarded unchanged to the system allocator.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: `layout` is forwarded unchanged to the system allocator.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: `pointer` and `layout` came from this delegated allocator.
        unsafe { System.dealloc(pointer, layout) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        // SAFETY: the arguments are forwarded unchanged to the system
        // allocator that created `pointer`.
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

#[test]
fn an_arm_and_dispatch_cycle_allocates_nothing() {
    const CYCLES: usize = 64;
    let listener = TcpListener::bind("127.0.0.1:0").expect("listener bind");
    let _client = TcpStream::connect(listener.local_addr().expect("listener address"))
        .expect("client connect");
    let (server, _) = listener.accept().expect("server accept");
    let port = AfdPort::new(8).expect("port");
    let socket = server.as_raw_socket();

    let cycle = || {
        port.arm(socket, Interest::WRITABLE).expect("arm");
        let mut delivered = 0_usize;
        let polls = port
            .poll(Some(Duration::from_secs(5)), |_, event| {
                delivered += usize::from(event.is_ok_and(|event| event.writable));
            })
            .expect("poll");
        assert_eq!((polls, delivered), (1, 1));
    };
    cycle();

    let before = ALLOCATIONS.load(Ordering::Relaxed);
    for _ in 0..CYCLES {
        cycle();
    }
    let allocated = ALLOCATIONS.load(Ordering::Relaxed) - before;
    assert_eq!(
        allocated, 0,
        "{CYCLES} arm and dispatch cycles performed {allocated} allocations"
    );
}

//! Temporary probe: who allocates the 36 864-byte global block per worker at
//! pool warmup. Wraps the system allocator, and on the first request of that
//! exact size captures a backtrace (re-entrancy guarded).
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

struct Probe;
static IN_HOOK: AtomicBool = AtomicBool::new(false);
static CAPTURED: AtomicBool = AtomicBool::new(false);
static COUNT: AtomicUsize = AtomicUsize::new(0);
const TARGET: usize = 36_864;

unsafe impl GlobalAlloc for Probe {
    unsafe fn alloc(&self, l: Layout) -> *mut u8 {
        let p = unsafe { System.alloc(l) };
        if l.size() == TARGET {
            COUNT.fetch_add(1, Ordering::Relaxed);
            if !IN_HOOK.swap(true, Ordering::SeqCst) {
                if !CAPTURED.swap(true, Ordering::SeqCst) {
                    let bt = std::backtrace::Backtrace::force_capture();
                    eprintln!("=== first {TARGET}-byte allocation (align {}) ===\n{bt}", l.align());
                }
                IN_HOOK.store(false, Ordering::SeqCst);
            }
        }
        p
    }
    unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
        unsafe { System.dealloc(p, l) }
    }
}

#[global_allocator]
static GLOBAL: Probe = Probe;

fn main() {
    let mut data = vec![0u64; 65536];
    moirai::for_each_chunk_mut_with::<moirai::Parallel, _, _>(&mut data, 256, |row| {
        std::hint::black_box(row);
    });
    eprintln!("total {TARGET}-byte allocations: {}", COUNT.load(Ordering::Relaxed));
}

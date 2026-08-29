//! Warm allocation contract for homogeneous multi-buffer chunk traversal.

use core::sync::atomic::{AtomicUsize, Ordering};
use moirai_parallel::{for_each_chunk_buffers_mut_enumerated_with, Parallel};
use std::alloc::{GlobalAlloc, Layout, System};

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

fn mutate_six_buffers(buffers: [&mut [u64]; 6]) {
    for_each_chunk_buffers_mut_enumerated_with::<Parallel, _, _, 6>(buffers, 256, |_, chunks| {
        for chunk in chunks {
            for value in chunk {
                *value += 1;
            }
        }
    })
    .expect("equal allocation-probe buffers must validate");
}

#[test]
fn warmed_chunk_buffer_traversal_allocates_nothing() {
    let mut buffers = core::array::from_fn::<_, 6, _>(|_| vec![0_u64; 4_096]);
    let [a, b, c, d, e, f] = &mut buffers;
    mutate_six_buffers([a, b, c, d, e, f]);

    ALLOCATIONS.store(0, Ordering::Relaxed);
    let [a, b, c, d, e, f] = &mut buffers;
    mutate_six_buffers([a, b, c, d, e, f]);
    let allocations = ALLOCATIONS.load(Ordering::Relaxed);

    assert_eq!(allocations, 0, "warmed provider overhead must be zero");
    assert!(
        buffers.iter().flatten().all(|&value| value == 2),
        "both traversals must mutate every element exactly once"
    );
}

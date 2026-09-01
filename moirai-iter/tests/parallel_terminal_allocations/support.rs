use moirai_iter::{
    cache::{CacheIterExt, CACHE_CHUNK_SIZE},
    iter_ops::ParallelIter,
    AsyncContext, ExecutionContext, MoiraiIterator,
};
use std::{
    alloc::{GlobalAlloc, Layout, System},
    future::Future,
    mem::size_of,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    task::Poll,
};

struct CountingAllocator;

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);
static ALLOCATED_BYTES: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every operation forwards its original pointer and layout to the
// system allocator. Relaxed counters carry no synchronization obligation.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        // SAFETY: the caller supplies the GlobalAlloc contract to this method.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: pointer and layout are forwarded unchanged.
        unsafe { System.dealloc(pointer, layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(layout.size(), Ordering::Relaxed);
        // SAFETY: the caller supplies the GlobalAlloc contract to this method.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
        ALLOCATED_BYTES.fetch_add(new_size, Ordering::Relaxed);
        // SAFETY: pointer and layout are forwarded unchanged with new_size.
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

pub(crate) const LEN: usize = 65_536;
pub(crate) const ALLOCATION_BUDGET: usize = LEN / 8;
pub(crate) const MAP_LEN: usize = 131_072;
pub(crate) const MAP_OUTPUT_BYTES: usize = MAP_LEN * size_of::<u64>();
pub(crate) const ZERO_COPY_MAP_LEN: usize = 1_024;
pub(crate) const ZERO_COPY_MAP_OUTPUT_BYTES: usize = ZERO_COPY_MAP_LEN * size_of::<u64>();
pub(crate) const ZERO_COPY_PARALLEL_MAP_LEN: usize =
    (CACHE_CHUNK_SIZE / size_of::<u64>()) * 1_024 + 1;
pub(crate) const ZERO_COPY_PARALLEL_MAP_OUTPUT_BYTES: usize =
    ZERO_COPY_PARALLEL_MAP_LEN * size_of::<u64>();
pub(crate) const CONTEXT_MAP_LEN: usize = 1_024;

const CONTEXT_MAP_OUTPUT_BYTES: usize = CONTEXT_MAP_LEN * size_of::<u64>();
const CONTEXT_FIXED_ALLOCATION_BUDGET: usize = 16;
const CONTEXT_FOR_EACH_FIXED_ALLOCATION_BUDGET: usize = 8;
const CONTEXT_SLOT_BYTE_BUDGET: usize = 64;
const CONTEXT_MAP_FIXED_BYTE_BUDGET: usize = 12_288;
const CONTEXT_FOR_EACH_FIXED_BYTE_BUDGET: usize = 2_048;

pub(crate) fn source() -> Vec<u64> {
    (0..LEN as u64).collect()
}

pub(crate) fn map_source() -> Vec<u64> {
    (0..MAP_LEN as u64)
        .map(|value| value.wrapping_mul(31).wrapping_add(7))
        .collect()
}

pub(crate) fn map_values(data: Vec<u64>) -> Vec<u64> {
    ParallelIter::new(data).map(|value| value.wrapping_mul(3).wrapping_add(1))
}

pub(crate) fn zero_copy_parallel_map_values(data: &[u64]) -> Vec<u64> {
    data.zero_copy_par_iter()
        .map(|value| value.wrapping_mul(3).wrapping_add(1))
}

pub(crate) fn allocation_snapshot() -> (usize, usize) {
    (
        ALLOCATIONS.load(Ordering::Relaxed),
        ALLOCATED_BYTES.load(Ordering::Relaxed),
    )
}

pub(crate) fn allocations_of<T>(operation: impl Fn() -> T) -> (T, usize) {
    let _warm = operation();
    let (before, _) = allocation_snapshot();
    let value = operation();
    let (after, _) = allocation_snapshot();
    (value, after.saturating_sub(before))
}

pub(crate) fn warmed_allocation_ledger<T, R>(
    warm_input: T,
    measured_input: T,
    mut operation: impl FnMut(T) -> R,
) -> (R, usize, usize) {
    let _warm = operation(warm_input);
    let (before_allocations, before_bytes) = allocation_snapshot();
    let result = operation(measured_input);
    let (after_allocations, after_bytes) = allocation_snapshot();
    (
        result,
        after_allocations.saturating_sub(before_allocations),
        after_bytes.saturating_sub(before_bytes),
    )
}

fn context_slot_count() -> usize {
    std::thread::available_parallelism().map_or(1, usize::from)
}

pub(crate) fn context_map_allocation_budget() -> usize {
    context_slot_count() + CONTEXT_FIXED_ALLOCATION_BUDGET
}

pub(crate) fn context_map_byte_budget() -> usize {
    CONTEXT_MAP_OUTPUT_BYTES
        + context_slot_count() * CONTEXT_SLOT_BYTE_BUDGET
        + CONTEXT_MAP_FIXED_BYTE_BUDGET
}

pub(crate) fn context_for_each_allocation_budget() -> usize {
    context_slot_count() + CONTEXT_FOR_EACH_FIXED_ALLOCATION_BUDGET
}

pub(crate) fn context_for_each_byte_budget() -> usize {
    context_slot_count() * CONTEXT_SLOT_BYTE_BUDGET + CONTEXT_FOR_EACH_FIXED_BYTE_BUDGET
}

pub(crate) fn context_map_values(data: Vec<u64>) -> Vec<u64> {
    futures::executor::block_on(async {
        MoiraiIterator::parallel(data)
            .map_async(|value| async move { value.wrapping_mul(5).wrapping_add(3) })
            .await
            .collect()
            .await
    })
}

fn pending_once<T>(value: T) -> impl Future<Output = T> {
    let mut value = Some(value);
    let mut first_poll = true;
    futures::future::poll_fn(move |context| {
        if first_poll {
            first_poll = false;
            context.waker().wake_by_ref();
            Poll::Pending
        } else {
            Poll::Ready(
                value
                    .take()
                    .expect("invariant: pending-once future polled after completion"),
            )
        }
    })
}

pub(crate) fn context_pending_map_values(data: Vec<u64>) -> Vec<u64> {
    futures::executor::block_on(async {
        MoiraiIterator::parallel(data)
            .map_async(|value| pending_once(value.wrapping_mul(5).wrapping_add(3)))
            .await
            .collect()
            .await
    })
}

pub(crate) fn context_pending_map_values_with_limit(
    data: Vec<u64>,
    max_concurrent: usize,
) -> Vec<u64> {
    let context = ExecutionContext::Async(AsyncContext::new().with_max_concurrent(max_concurrent));
    futures::executor::block_on(async {
        MoiraiIterator::new(data, context)
            .map_async(|value| pending_once(value.wrapping_mul(5).wrapping_add(3)))
            .await
            .collect()
            .await
    })
}

pub(crate) fn context_large_limit_map_values(data: Vec<u64>) -> Vec<u64> {
    context_pending_map_values_with_limit(data, usize::MAX)
}

pub(crate) fn context_pending_for_each(data: Vec<usize>, visits: Arc<Vec<AtomicUsize>>) {
    futures::executor::block_on(MoiraiIterator::parallel(data).for_each_async(move |index| {
        let visits = Arc::clone(&visits);
        async move {
            pending_once(()).await;
            visits[index].fetch_add(1, Ordering::SeqCst);
        }
    }));
}

//! Pointer-identity allocation accounting for the indexed executor probe.

use std::{
    alloc::{GlobalAlloc, Layout, System},
    sync::atomic::{AtomicIsize, AtomicPtr, AtomicUsize, Ordering},
};

/// Even values are closed; odd values identify one open measurement window.
static WINDOW_EPOCH: AtomicUsize = AtomicUsize::new(0);
/// Allocator and direct-provider events admitted before a window closes.
static ACTIVE_EVENTS: AtomicUsize = AtomicUsize::new(0);

const LEDGER_SLOTS: usize = 2_048;

struct AllocationCounter {
    allocations: AtomicUsize,
    live: AtomicIsize,
    peak: AtomicIsize,
    sizes: [AtomicUsize; LEDGER_SLOTS],
    pointers: [AtomicPtr<u8>; LEDGER_SLOTS],
    next: AtomicUsize,
    dropped: AtomicUsize,
}

impl AllocationCounter {
    const fn new() -> Self {
        Self {
            allocations: AtomicUsize::new(0),
            live: AtomicIsize::new(0),
            peak: AtomicIsize::new(0),
            sizes: [const { AtomicUsize::new(0) }; LEDGER_SLOTS],
            pointers: [const { AtomicPtr::new(std::ptr::null_mut()) }; LEDGER_SLOTS],
            next: AtomicUsize::new(0),
            dropped: AtomicUsize::new(0),
        }
    }

    fn reset(&self) {
        // Clear publication before resetting the slot cursor. A newly reserved
        // but unpublished slot must never expose a pointer from an older window.
        for pointer in &self.pointers {
            pointer.store(std::ptr::null_mut(), Ordering::Relaxed);
        }
        self.allocations.store(0, Ordering::Relaxed);
        self.live.store(0, Ordering::Relaxed);
        self.peak.store(0, Ordering::Relaxed);
        self.next.store(0, Ordering::Relaxed);
        self.dropped.store(0, Ordering::Relaxed);
    }

    fn push(&self, pointer: *mut u8, size: usize) {
        let slot = self.next.fetch_add(1, Ordering::Relaxed);
        if slot < LEDGER_SLOTS {
            self.sizes[slot].store(size, Ordering::Relaxed);
            self.pointers[slot].store(pointer, Ordering::Release);
        } else {
            self.dropped.fetch_add(1, Ordering::Relaxed);
        }
    }

    fn claim(&self, pointer: *mut u8) -> Option<(usize, usize)> {
        let filled = self.next.load(Ordering::Relaxed).min(LEDGER_SLOTS);
        for slot in 0..filled {
            if self.pointers[slot]
                .compare_exchange(
                    pointer,
                    std::ptr::null_mut(),
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .is_ok()
            {
                return Some((slot, self.sizes[slot].load(Ordering::Relaxed)));
            }
        }
        None
    }

    fn restore(&self, slot: usize, pointer: *mut u8) {
        self.pointers[slot].store(pointer, Ordering::Release);
    }

    fn track(&self, delta: isize) {
        let live = self.live.fetch_add(delta, Ordering::Relaxed) + delta;
        self.peak.fetch_max(live, Ordering::Relaxed);
    }

    fn record_alloc(&self, pointer: *mut u8, size: usize) {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        let Ok(delta) = isize::try_from(size) else {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return;
        };
        self.track(delta);
        self.push(pointer, size);
    }

    fn record_free(&self, pointer: *mut u8) {
        if let Some((_, size)) = self.claim(pointer) {
            let Ok(delta) = isize::try_from(size) else {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                return;
            };
            self.track(-delta);
        }
    }

    fn record_realloc_claimed(
        &self,
        old: Option<(usize, usize)>,
        new_pointer: *mut u8,
        new_size: usize,
    ) {
        self.allocations.fetch_add(1, Ordering::Relaxed);
        let Ok(new_delta) = isize::try_from(new_size) else {
            self.dropped.fetch_add(1, Ordering::Relaxed);
            return;
        };
        if let Some((_, old_size)) = old {
            let Ok(old_delta) = isize::try_from(old_size) else {
                self.dropped.fetch_add(1, Ordering::Relaxed);
                return;
            };
            self.track(new_delta - old_delta);
        } else {
            self.track(new_delta);
        }
        self.push(new_pointer, new_size);
    }

    fn snapshot(&self) -> CounterSnapshot {
        #[cfg(feature = "mnemosyne")]
        let filled = self.next.load(Ordering::Relaxed).min(LEDGER_SLOTS);
        #[cfg(feature = "mnemosyne")]
        let mut blocks = Vec::<(usize, usize)>::new();
        #[cfg(feature = "mnemosyne")]
        for slot in 0..filled {
            if !self.pointers[slot].load(Ordering::Acquire).is_null() {
                let size = self.sizes[slot].load(Ordering::Relaxed);
                match blocks.iter_mut().find(|(candidate, _)| *candidate == size) {
                    Some((_, count)) => *count += 1,
                    None => blocks.push((size, 1)),
                }
            }
        }
        #[cfg(feature = "mnemosyne")]
        blocks.sort_unstable_by(|left, right| right.cmp(left));

        #[cfg(feature = "mnemosyne")]
        let retained = self.live.load(Ordering::Relaxed);
        #[cfg(feature = "mnemosyne")]
        let listed = blocks.iter().fold(0usize, |total, (size, count)| {
            total
                .checked_add(size.checked_mul(*count).expect("block total fits usize"))
                .expect("ledger total fits usize")
        });
        #[cfg(feature = "mnemosyne")]
        assert_eq!(
            retained,
            isize::try_from(listed).expect("ledger total fits isize"),
            "live-byte balance must equal the pointer-ledger total"
        );
        #[cfg(feature = "mnemosyne")]
        assert_eq!(
            self.dropped.load(Ordering::Relaxed),
            0,
            "allocation ledger overflowed; increase LEDGER_SLOTS before using the report"
        );

        CounterSnapshot {
            allocations: self.allocations.load(Ordering::Relaxed),
            #[cfg(feature = "mnemosyne")]
            peak: self.peak.load(Ordering::Relaxed),
            #[cfg(feature = "mnemosyne")]
            retained,
            #[cfg(feature = "mnemosyne")]
            blocks,
        }
    }
}

static GLOBAL_ALLOCATIONS: AllocationCounter = AllocationCounter::new();
static MNEMOSYNE_ALLOCATIONS: AllocationCounter = AllocationCounter::new();

fn enter_window() -> bool {
    let epoch = WINDOW_EPOCH.load(Ordering::Acquire);
    if epoch & 1 == 0 {
        return false;
    }
    ACTIVE_EVENTS.fetch_add(1, Ordering::AcqRel);
    if WINDOW_EPOCH.load(Ordering::Acquire) == epoch {
        true
    } else {
        ACTIVE_EVENTS.fetch_sub(1, Ordering::AcqRel);
        false
    }
}

fn leave_window() {
    ACTIVE_EVENTS.fetch_sub(1, Ordering::AcqRel);
}

struct AttributingAllocator;

// SAFETY: every method forwards to `System` unchanged; the counters only
// observe successful block identities and sizes.
unsafe impl GlobalAlloc for AttributingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let count = enter_window();
        // SAFETY: the caller supplies a valid layout and this wrapper forwards
        // it unchanged.
        let pointer = unsafe { System.alloc(layout) };
        if count && !pointer.is_null() {
            GLOBAL_ALLOCATIONS.record_alloc(pointer, layout.size());
        }
        if count {
            leave_window();
        }
        pointer
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        let count = enter_window();
        if count {
            GLOBAL_ALLOCATIONS.record_free(pointer);
        }
        // SAFETY: the wrapper preserves the live pointer and its layout.
        unsafe { System.dealloc(pointer, layout) };
        if count {
            leave_window();
        }
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let count = enter_window();
        let old = count.then(|| GLOBAL_ALLOCATIONS.claim(pointer)).flatten();
        // SAFETY: the wrapper preserves the live source block and replacement
        // size supplied by the caller.
        let replacement = unsafe { System.realloc(pointer, layout, new_size) };
        if count && !replacement.is_null() {
            GLOBAL_ALLOCATIONS.record_realloc_claimed(old, replacement, new_size);
        } else if let Some((slot, _)) = old {
            // A failed realloc leaves the source allocation live.
            GLOBAL_ALLOCATIONS.restore(slot, pointer);
        }
        if count {
            leave_window();
        }
        replacement
    }
}

#[global_allocator]
static ALLOCATOR: AttributingAllocator = AttributingAllocator;

#[cfg(feature = "mnemosyne")]
unsafe extern "C" fn mnemosyne_alloc_hook(pointer: *mut core::ffi::c_void, size: usize) {
    let count = enter_window();
    if count && !pointer.is_null() && size != 0 {
        MNEMOSYNE_ALLOCATIONS.record_alloc(pointer.cast(), size);
    }
    if count {
        leave_window();
    }
}

#[cfg(feature = "mnemosyne")]
unsafe extern "C" fn mnemosyne_free_hook(pointer: *mut core::ffi::c_void, _size: usize) {
    let count = enter_window();
    if count && !pointer.is_null() {
        MNEMOSYNE_ALLOCATIONS.record_free(pointer.cast());
    }
    if count {
        leave_window();
    }
}

#[cfg(feature = "mnemosyne")]
pub(super) struct MnemosyneHooks;

#[cfg(feature = "mnemosyne")]
impl MnemosyneHooks {
    pub(super) fn install() -> Self {
        mnemosyne::register_alloc_hook(Some(mnemosyne_alloc_hook));
        mnemosyne::register_free_hook(Some(mnemosyne_free_hook));
        Self
    }
}

#[cfg(feature = "mnemosyne")]
impl Drop for MnemosyneHooks {
    fn drop(&mut self) {
        mnemosyne::register_alloc_hook(None);
        mnemosyne::register_free_hook(None);
    }
}

struct OpenWindow {
    active: bool,
}

impl OpenWindow {
    fn start() -> Self {
        assert_eq!(
            ACTIVE_EVENTS.load(Ordering::Acquire),
            0,
            "allocation window opened with events still active"
        );
        GLOBAL_ALLOCATIONS.reset();
        MNEMOSYNE_ALLOCATIONS.reset();
        let closed_epoch = WINDOW_EPOCH.fetch_add(1, Ordering::AcqRel);
        assert_eq!(
            closed_epoch & 1,
            0,
            "allocation window opened while another window was active"
        );
        Self { active: true }
    }

    fn finish(mut self) {
        close_window();
        self.active = false;
    }
}

impl Drop for OpenWindow {
    fn drop(&mut self) {
        if self.active {
            close_window();
        }
    }
}

fn close_window() {
    let open_epoch = WINDOW_EPOCH.fetch_add(1, Ordering::AcqRel);
    assert_eq!(
        open_epoch & 1,
        1,
        "allocation window closed from an inactive state"
    );
    while ACTIVE_EVENTS.load(Ordering::Acquire) != 0 {
        std::hint::spin_loop();
    }
}

pub(super) struct CounterSnapshot {
    allocations: usize,
    #[cfg(feature = "mnemosyne")]
    peak: isize,
    #[cfg(feature = "mnemosyne")]
    retained: isize,
    #[cfg(feature = "mnemosyne")]
    blocks: Vec<(usize, usize)>,
}

impl CounterSnapshot {
    pub(super) fn allocations(&self) -> usize {
        self.allocations
    }

    #[cfg(feature = "mnemosyne")]
    pub(super) fn retained(&self) -> isize {
        self.retained
    }

    #[cfg(feature = "mnemosyne")]
    pub(super) fn block_count(&self, size: usize) -> usize {
        self.blocks
            .iter()
            .find_map(|(candidate, count)| (*candidate == size).then_some(*count))
            .unwrap_or(0)
    }

    #[cfg(feature = "mnemosyne")]
    pub(super) fn total_blocks(&self) -> usize {
        self.blocks.iter().map(|(_, count)| count).sum()
    }

    #[cfg(feature = "mnemosyne")]
    fn print(&self, label: &str, source: &str) {
        let blocks = self
            .blocks
            .iter()
            .map(|(size, count)| format!("{size}x{count}"))
            .collect::<Vec<_>>()
            .join(" ");
        println!(
            "  {label:<28} {source:<17} allocs {:>5}  peak {:>10}  retained {:>10}  blocks: {blocks}",
            self.allocations, self.peak, self.retained
        );
    }
}

pub(super) struct FootprintSnapshot {
    pub(super) global: CounterSnapshot,
    #[cfg(feature = "mnemosyne")]
    pub(super) direct: CounterSnapshot,
}

pub(super) fn measure<R>(operation: impl FnOnce() -> R) -> (R, FootprintSnapshot) {
    let window = OpenWindow::start();
    let output = operation();
    window.finish();
    (
        output,
        FootprintSnapshot {
            global: GLOBAL_ALLOCATIONS.snapshot(),
            #[cfg(feature = "mnemosyne")]
            direct: MNEMOSYNE_ALLOCATIONS.snapshot(),
        },
    )
}

#[cfg(feature = "mnemosyne")]
pub(super) fn footprint_window<R>(
    label: &str,
    operation: impl FnOnce() -> R,
) -> (R, FootprintSnapshot) {
    let (output, snapshot) = measure(operation);
    snapshot.global.print(label, "global");
    snapshot.direct.print(label, "Mnemosyne direct");
    (output, snapshot)
}

#[cfg(test)]
mod tests {
    use super::{AllocationCounter, Ordering};

    #[test]
    fn reset_hides_stale_slots_from_an_unpublished_current_slot() {
        static COUNTER: AllocationCounter = AllocationCounter::new();
        let mut byte = 0u8;
        let pointer = std::ptr::from_mut(&mut byte);

        COUNTER.reset();
        COUNTER.record_alloc(pointer, 64);
        COUNTER.reset();
        COUNTER.next.store(1, Ordering::Relaxed);
        COUNTER.record_free(pointer);

        assert_eq!(COUNTER.live.load(Ordering::Relaxed), 0);
        assert!(COUNTER.pointers[0].load(Ordering::Acquire).is_null());
    }

    #[test]
    fn claimed_realloc_identity_cannot_consume_a_reused_address() {
        static COUNTER: AllocationCounter = AllocationCounter::new();
        let mut old_byte = 0u8;
        let mut replacement_byte = 0u8;
        let old_pointer = std::ptr::from_mut(&mut old_byte);
        let replacement_pointer = std::ptr::from_mut(&mut replacement_byte);

        COUNTER.reset();
        COUNTER.record_alloc(old_pointer, 64);
        let old = COUNTER
            .claim(old_pointer)
            .expect("the realloc source must be tracked");
        COUNTER.record_alloc(old_pointer, 32);
        COUNTER.record_realloc_claimed(Some(old), replacement_pointer, 128);

        let (_, reused_size) = COUNTER
            .claim(old_pointer)
            .expect("the reused address must retain its own identity");
        let (_, replacement_size) = COUNTER
            .claim(replacement_pointer)
            .expect("the realloc result must be tracked independently");
        assert_eq!(reused_size, 32);
        assert_eq!(replacement_size, 128);
    }

    #[test]
    fn failed_realloc_restores_the_claimed_source() {
        static COUNTER: AllocationCounter = AllocationCounter::new();
        let mut byte = 0u8;
        let pointer = std::ptr::from_mut(&mut byte);

        COUNTER.reset();
        COUNTER.record_alloc(pointer, 64);
        let (slot, size) = COUNTER
            .claim(pointer)
            .expect("the realloc source must be tracked");
        COUNTER.restore(slot, pointer);

        let (_, restored_size) = COUNTER
            .claim(pointer)
            .expect("a failed realloc must leave its source tracked");
        assert_eq!(restored_size, size);
        assert_eq!(COUNTER.live.load(Ordering::Relaxed), 64);
    }
}

//! Block-based lock-free work-stealing deque.

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::StealResult;

const BLOCK_SIZE: usize = 64;

struct Block<T> {
    data: [UnsafeCell<MaybeUninit<T>>; BLOCK_SIZE],
    next: AtomicPtr<Block<T>>,
}

impl<T> Block<T> {
    fn new() -> *mut Self {
        unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc(layout) as *mut Self;
            if ptr.is_null() {
                panic!("Allocation of Block failed");
            }
            // Initialize data with raw cell wrappers without array initialization overhead
            for i in 0..BLOCK_SIZE {
                std::ptr::write(
                    &mut (*ptr).data[i],
                    UnsafeCell::new(MaybeUninit::uninit()),
                );
            }
            std::ptr::write(&mut (*ptr).next, AtomicPtr::new(std::ptr::null_mut()));
            ptr
        }
    }
}

/// A block-based lock-free work-stealing deque.
pub struct BlockBasedDeque<T> {
    head: AtomicPtr<Block<T>>,
    tail: AtomicPtr<Block<T>>,
    top: AtomicUsize,
    bottom: AtomicUsize,
    len: AtomicUsize,
    retired_blocks: Mutex<Vec<*mut Block<T>>>,
}

// Safety: BlockBasedDeque is safe to send and synchronize if elements are Send.
unsafe impl<T: Send> Send for BlockBasedDeque<T> {}
unsafe impl<T: Send> Sync for BlockBasedDeque<T> {}

impl<T> BlockBasedDeque<T> {
    /// Create a new block-based deque.
    pub fn new() -> Self {
        let first_block = Block::new();
        Self {
            head: AtomicPtr::new(first_block),
            tail: AtomicPtr::new(first_block),
            top: AtomicUsize::new(0),
            bottom: AtomicUsize::new(0),
            len: AtomicUsize::new(0),
            retired_blocks: Mutex::new(Vec::new()),
        }
    }

    /// Push an item to the bottom of the deque.
    pub fn push(&self, item: T) {
        let tail = self.tail.load(Ordering::Relaxed);
        let b = self.bottom.load(Ordering::Relaxed);

        if b < BLOCK_SIZE {
            unsafe {
                let cell = &(*tail).data[b];
                cell.get().write(MaybeUninit::new(item));
            }
            self.bottom.store(b + 1, Ordering::Release);
        } else {
            let new_block = Block::new();
            unsafe {
                let cell = &(*new_block).data[0];
                cell.get().write(MaybeUninit::new(item));
                (*tail).next.store(new_block, Ordering::Release);
            }
            self.tail.store(new_block, Ordering::Release);
            self.bottom.store(1, Ordering::Release);
        }
        self.len.fetch_add(1, Ordering::Relaxed);
    }

    /// Pop an item from the bottom of the deque.
    pub fn pop(&self) -> Option<T> {
        let tail = self.tail.load(Ordering::Relaxed);
        let b = self.bottom.load(Ordering::Relaxed);

        if b > 0 {
            let new_b = b - 1;

            // TSO Fence-Free pop optimization:
            #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
            {
                let head = self.head.load(Ordering::Relaxed);
                let t = if head == tail {
                    self.top.load(Ordering::Relaxed)
                } else {
                    0
                };
                if new_b > t + 1 {
                    self.bottom.store(new_b, Ordering::Release);
                } else {
                    self.bottom.store(new_b, Ordering::SeqCst);
                }
            }
            #[cfg(not(any(target_arch = "x86", target_arch = "x86_64")))]
            {
                self.bottom.store(new_b, Ordering::Relaxed);
                std::sync::atomic::fence(Ordering::SeqCst);
            }

            let head = self.head.load(Ordering::Relaxed);
            let t = if head == tail {
                self.top.load(Ordering::Acquire)
            } else {
                0
            };

            if head == tail {
                if t < new_b {
                    let item = unsafe {
                        let cell = &(*tail).data[new_b];
                        cell.get().read().assume_init()
                    };
                    self.len.fetch_sub(1, Ordering::Relaxed);
                    return Some(item);
                }
                if t == new_b {
                    if self
                        .top
                        .compare_exchange(t, t + 1, Ordering::SeqCst, Ordering::Relaxed)
                        .is_ok()
                    {
                        let item = unsafe {
                            let cell = &(*tail).data[new_b];
                            cell.get().read().assume_init()
                        };
                        self.bottom.store(new_b + 1, Ordering::Relaxed);
                        self.len.fetch_sub(1, Ordering::Relaxed);
                        return Some(item);
                    }
                }
                self.bottom.store(new_b + 1, Ordering::Relaxed);
                return None;
            } else {
                let item = unsafe {
                    let cell = &(*tail).data[new_b];
                    cell.get().read().assume_init()
                };
                self.bottom.store(new_b, Ordering::Release);
                self.len.fetch_sub(1, Ordering::Relaxed);
                return Some(item);
            }
        } else {
            // If the tail block is empty, but head != tail, we can steal from our own head.
            // If head == tail, the deque is truly empty.
            let head = self.head.load(Ordering::Relaxed);
            if head == tail {
                return None;
            }
            loop {
                match self.steal() {
                    StealResult::Success(item) => return Some(item),
                    StealResult::Empty => return None,
                    StealResult::Retry => continue,
                }
            }
        }
    }

    /// Steal an item from the top of the deque.
    pub fn steal(&self) -> StealResult<T> {
        loop {
            let head = self.head.load(Ordering::Acquire);
            let tail = self.tail.load(Ordering::Acquire);
            let t = self.top.load(Ordering::Acquire);
            let b = self.bottom.load(Ordering::Acquire);

            if head == tail {
                if t >= b {
                    return StealResult::Empty;
                }
                if self
                    .top
                    .compare_exchange(t, t + 1, Ordering::SeqCst, Ordering::Relaxed)
                    .is_ok()
                {
                    let item = unsafe {
                        let cell = &(*head).data[t];
                        cell.get().read().assume_init()
                    };
                    self.len.fetch_sub(1, Ordering::Relaxed);
                    return StealResult::Success(item);
                }
                return StealResult::Retry;
            } else {
                if t < BLOCK_SIZE {
                    if self
                        .top
                        .compare_exchange(t, t + 1, Ordering::SeqCst, Ordering::Relaxed)
                        .is_ok()
                    {
                        let item = unsafe {
                            let cell = &(*head).data[t];
                            cell.get().read().assume_init()
                        };
                        self.len.fetch_sub(1, Ordering::Relaxed);
                        return StealResult::Success(item);
                    }
                    return StealResult::Retry;
                } else {
                    let next = unsafe { (*head).next.load(Ordering::Acquire) };
                    if next.is_null() {
                        return StealResult::Empty;
                    }
                    if self
                        .head
                        .compare_exchange(head, next, Ordering::SeqCst, Ordering::Relaxed)
                        .is_ok()
                    {
                        self.top.store(0, Ordering::Release);
                        let mut retired = self.retired_blocks.lock().unwrap();
                        retired.push(head);
                    }
                    return StealResult::Retry;
                }
            }
        }
    }

    /// Steal multiple items from this deque, passing all but the first one to the closure
    /// and returning the first one.
    pub fn steal_batch_with<F>(&self, mut f: F) -> StealResult<T>
    where
        F: FnMut(T),
    {
        loop {
            let head = self.head.load(Ordering::Acquire);
            let tail = self.tail.load(Ordering::Acquire);
            let t = self.top.load(Ordering::Acquire);
            let b = self.bottom.load(Ordering::Acquire);

            if head == tail {
                if t >= b {
                    return StealResult::Empty;
                }
                let len = b - t;
                let n = (len / 2).max(1);
                if self
                    .top
                    .compare_exchange(t, t + n, Ordering::SeqCst, Ordering::Relaxed)
                    .is_ok()
                {
                    let first_item = unsafe {
                        let cell = &(*head).data[t];
                        cell.get().read().assume_init()
                    };
                    for i in 1..n {
                        let item = unsafe {
                            let cell = &(*head).data[t + i];
                            cell.get().read().assume_init()
                        };
                        f(item);
                    }
                    self.len.fetch_sub(n, Ordering::Relaxed);
                    return StealResult::Success(first_item);
                }
                return StealResult::Retry;
            } else {
                if t < BLOCK_SIZE {
                    let len = BLOCK_SIZE - t;
                    let n = (len / 2).max(1);
                    if self
                        .top
                        .compare_exchange(t, t + n, Ordering::SeqCst, Ordering::Relaxed)
                        .is_ok()
                    {
                        let first_item = unsafe {
                            let cell = &(*head).data[t];
                            cell.get().read().assume_init()
                        };
                        for i in 1..n {
                            let item = unsafe {
                                let cell = &(*head).data[t + i];
                                cell.get().read().assume_init()
                            };
                            f(item);
                        }
                        self.len.fetch_sub(n, Ordering::Relaxed);
                        return StealResult::Success(first_item);
                    }
                    return StealResult::Retry;
                } else {
                    let next = unsafe { (*head).next.load(Ordering::Acquire) };
                    if next.is_null() {
                        return StealResult::Empty;
                    }
                    if self
                        .head
                        .compare_exchange(head, next, Ordering::SeqCst, Ordering::Relaxed)
                        .is_ok()
                    {
                        self.top.store(0, Ordering::Release);
                        let mut retired = self.retired_blocks.lock().unwrap();
                        retired.push(head);
                    }
                    return StealResult::Retry;
                }
            }
        }
    }

    /// Deallocate retired blocks through an exclusive quiescent access path.
    pub fn reclaim_memory(&mut self) {
        let mut retired = self.retired_blocks.lock().unwrap();
        for ptr in retired.drain(..) {
            unsafe {
                let layout = std::alloc::Layout::new::<Block<T>>();
                std::alloc::dealloc(ptr as *mut u8, layout);
            }
        }
    }

    /// Get the approximate number of tasks in the deque.
    pub fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

    /// Check if the deque is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T> Default for BlockBasedDeque<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T> Drop for BlockBasedDeque<T> {
    fn drop(&mut self) {
        let head = self.head.load(Ordering::Relaxed);
        let mut curr = head;
        while !curr.is_null() {
            let next = unsafe { (*curr).next.load(Ordering::Relaxed) };
            unsafe {
                let start = if curr == head {
                    self.top.load(Ordering::Relaxed)
                } else {
                    0
                };
                let b = if curr == self.tail.load(Ordering::Relaxed) {
                    self.bottom.load(Ordering::Relaxed)
                } else {
                    BLOCK_SIZE
                };
                for i in start..b {
                    let cell = &(*curr).data[i];
                    std::ptr::drop_in_place(cell.get() as *mut T);
                }
                let layout = std::alloc::Layout::new::<Block<T>>();
                std::alloc::dealloc(curr as *mut u8, layout);
            }
            curr = next;
        }
        self.reclaim_memory();
    }
}

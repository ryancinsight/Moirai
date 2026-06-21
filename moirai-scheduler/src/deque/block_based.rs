use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::sync::atomic::{AtomicPtr, AtomicUsize, Ordering};
use std::sync::Mutex;

use super::chase_lev::StealResult;

const BLOCK_SIZE: usize = 64;
const MAX_BATCH_STEAL: usize = 16;

struct Block<T> {
    data: [UnsafeCell<MaybeUninit<T>>; BLOCK_SIZE],
    next: AtomicPtr<Block<T>>,
    next_retired: AtomicPtr<Block<T>>,
    top: AtomicUsize,
}

impl<T> Block<T> {
    fn new() -> *mut Self {
        unsafe {
            let layout = std::alloc::Layout::new::<Self>();
            let ptr = std::alloc::alloc(layout) as *mut Self;
            if ptr.is_null() {
                panic!("Allocation of Block failed");
            }
            for i in 0..BLOCK_SIZE {
                std::ptr::write(&mut (*ptr).data[i], UnsafeCell::new(MaybeUninit::uninit()));
            }
            std::ptr::write(&mut (*ptr).next, AtomicPtr::new(std::ptr::null_mut()));
            std::ptr::write(&mut (*ptr).next_retired, AtomicPtr::new(std::ptr::null_mut()));
            std::ptr::write(&mut (*ptr).top, AtomicUsize::new(0));
            ptr
        }
    }
}

/// A block-based lock-free work-stealing deque.
pub struct BlockBasedDeque<T> {
    head: AtomicPtr<Block<T>>,
    tail: AtomicPtr<Block<T>>,
    bottom: AtomicUsize,
    len: AtomicUsize,
    retired_head: AtomicPtr<Block<T>>,
    free_blocks: Mutex<Vec<*mut Block<T>>>,
}

unsafe impl<T: Send> Send for BlockBasedDeque<T> {}
unsafe impl<T: Send> Sync for BlockBasedDeque<T> {}

impl<T> BlockBasedDeque<T> {
    pub fn new() -> Self {
        let first_block = Block::new();
        Self {
            head: AtomicPtr::new(first_block),
            tail: AtomicPtr::new(first_block),
            bottom: AtomicUsize::new(0),
            len: AtomicUsize::new(0),
            retired_head: AtomicPtr::new(std::ptr::null_mut()),
            free_blocks: Mutex::new(Vec::new()),
        }
    }

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
            let new_block = {
                let mut free = self.free_blocks.lock().unwrap();
                if let Some(block) = free.pop() {
                    unsafe {
                        (*block).next.store(std::ptr::null_mut(), Ordering::Relaxed);
                        (*block).next_retired.store(std::ptr::null_mut(), Ordering::Relaxed);
                        (*block).top.store(0, Ordering::Relaxed);
                    }
                    block
                } else {
                    Block::new()
                }
            };
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

    pub fn pop(&self) -> Option<T> {
        let tail = self.tail.load(Ordering::Relaxed);
        let b = self.bottom.load(Ordering::Relaxed);

        if b > 0 {
            let new_b = b - 1;
            self.bottom.store(new_b, Ordering::Relaxed);

            // Fast path: if head != tail, they are different blocks, no contention!
            let head = self.head.load(Ordering::Relaxed);
            if head != tail {
                let item = unsafe {
                    let cell = &(*tail).data[new_b];
                    cell.get().read().assume_init()
                };
                self.len.fetch_sub(1, Ordering::Relaxed);
                return Some(item);
            }

            // Fallback path: same block, execute standard Chase-Lev synchronization
            std::sync::atomic::fence(Ordering::SeqCst);
            let t = unsafe { (*tail).top.load(Ordering::Acquire) };
            if t < new_b {
                let item = unsafe {
                    let cell = &(*tail).data[new_b];
                    cell.get().read().assume_init()
                };
                self.len.fetch_sub(1, Ordering::Relaxed);
                Some(item)
            } else if t == new_b {
                if unsafe { &*tail }
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
                    Some(item)
                } else {
                    self.bottom.store(new_b + 1, Ordering::Relaxed);
                    None
                }
            } else {
                self.bottom.store(new_b + 1, Ordering::Relaxed);
                None
            }
        } else {
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

    pub fn steal(&self) -> StealResult<T> {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        let t = unsafe { (*head).top.load(Ordering::Acquire) };
        std::sync::atomic::fence(Ordering::SeqCst);
        let b = self.bottom.load(Ordering::Acquire);

        if head == tail {
            if t >= b {
                return StealResult::Empty;
            }
            if unsafe { &*head }
                .top
                .compare_exchange(t, t + 1, Ordering::SeqCst, Ordering::Relaxed)
                .is_ok()
            {
                let item = unsafe {
                    let cell = &(*head).data[t];
                    cell.get().read().assume_init()
                };
                self.len.fetch_sub(1, Ordering::Relaxed);
                StealResult::Success(item)
            } else {
                StealResult::Retry
            }
        } else {
            if t < BLOCK_SIZE {
                if unsafe { &*head }
                    .top
                    .compare_exchange(t, t + 1, Ordering::SeqCst, Ordering::Relaxed)
                    .is_ok()
                {
                    let item = unsafe {
                        let cell = &(*head).data[t];
                        cell.get().read().assume_init()
                    };
                    self.len.fetch_sub(1, Ordering::Relaxed);
                    StealResult::Success(item)
                } else {
                    StealResult::Retry
                }
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
                    unsafe {
                        let mut old_retired = self.retired_head.load(Ordering::Relaxed);
                        loop {
                            (*head).next_retired.store(old_retired, Ordering::Relaxed);
                            match self.retired_head.compare_exchange_weak(
                                old_retired,
                                head,
                                Ordering::Release,
                                Ordering::Relaxed,
                            ) {
                                Ok(_) => break,
                                Err(actual) => old_retired = actual,
                            }
                        }
                    }
                }
                StealResult::Retry
            }
        }
    }

    pub fn steal_batch_with<F>(&self, mut f: F) -> StealResult<T>
    where
        F: FnMut(T),
    {
        let first_item = match self.steal() {
            StealResult::Success(item) => item,
            StealResult::Empty => return StealResult::Empty,
            StealResult::Retry => return StealResult::Retry,
        };

        let batch_extra = (self.len().saturating_sub(1) / 2).min(MAX_BATCH_STEAL - 1);
        for _ in 0..batch_extra {
            match self.steal() {
                StealResult::Success(item) => f(item),
                StealResult::Empty | StealResult::Retry => break,
            }
        }

        StealResult::Success(first_item)
    }

    pub fn reclaim_memory(&mut self) {
        let mut curr = self.retired_head.swap(std::ptr::null_mut(), Ordering::Acquire);
        let mut free = self.free_blocks.lock().unwrap();
        while !curr.is_null() {
            let next = unsafe { (*curr).next_retired.load(Ordering::Relaxed) };
            free.push(curr);
            curr = next;
        }
    }

    pub fn len(&self) -> usize {
        self.len.load(Ordering::Relaxed)
    }

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
                let start = (*curr).top.load(Ordering::Relaxed);
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

        let mut curr = self.retired_head.swap(std::ptr::null_mut(), Ordering::Acquire);
        while !curr.is_null() {
            let next = unsafe { (*curr).next_retired.load(Ordering::Relaxed) };
            unsafe {
                let layout = std::alloc::Layout::new::<Block<T>>();
                std::alloc::dealloc(curr as *mut u8, layout);
            }
            curr = next;
        }

        let mut free = self.free_blocks.lock().unwrap();
        for ptr in free.drain(..) {
            unsafe {
                let layout = std::alloc::Layout::new::<Block<T>>();
                std::alloc::dealloc(ptr as *mut u8, layout);
            }
        }
    }
}

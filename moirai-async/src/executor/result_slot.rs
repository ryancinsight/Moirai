use std::{
    cell::UnsafeCell,
    mem::MaybeUninit,
    sync::atomic::{AtomicU8, Ordering},
    task::Waker,
};

const ASYNC_RESULT_PENDING: u8 = 0;
const ASYNC_RESULT_WAITING: u8 = 1;
const ASYNC_RESULT_UPDATING_WAKER: u8 = 2;
const ASYNC_RESULT_WRITING: u8 = 3;
const ASYNC_RESULT_READY: u8 = 4;
const ASYNC_RESULT_TAKEN: u8 = 5;

/// Single-producer result slot for one async handle.
pub(super) struct AsyncResultSlot<T> {
    result: UnsafeCell<MaybeUninit<T>>,
    state: AtomicU8,
    waiter: UnsafeCell<MaybeUninit<Waker>>,
}

// Safety: the slot has one producer and one consumer. Atomic states serialize
// result publication, result consumption, and inline waker updates.
unsafe impl<T: Send> Send for AsyncResultSlot<T> {}

// Safety: shared access is mediated by the state machine; result and waker cells
// are touched only after the corresponding atomic transition succeeds.
unsafe impl<T: Send> Sync for AsyncResultSlot<T> {}

impl<T> AsyncResultSlot<T> {
    pub(super) fn new() -> Self {
        Self {
            result: UnsafeCell::new(MaybeUninit::uninit()),
            state: AtomicU8::new(ASYNC_RESULT_PENDING),
            waiter: UnsafeCell::new(MaybeUninit::uninit()),
        }
    }

    pub(super) fn complete(&self, result: T) {
        let Some(waiting) = self.begin_completion() else {
            return;
        };

        // Safety: WRITING is reachable only after `begin_completion` wins the
        // producer transition, so no consumer can read this cell yet.
        unsafe {
            (*self.result.get()).write(result);
        }

        self.state.store(ASYNC_RESULT_READY, Ordering::Release);

        if waiting {
            // Safety: WAITING is reachable only after `register_waker` writes
            // the waker and publishes it with a release transition.
            let waker = unsafe { (*self.waiter.get()).assume_init_read() };
            waker.wake();
        }
    }

    pub(super) fn try_take_ready(&self) -> Option<T> {
        if self
            .state
            .compare_exchange(
                ASYNC_RESULT_READY,
                ASYNC_RESULT_TAKEN,
                Ordering::Acquire,
                Ordering::Relaxed,
            )
            .is_ok()
        {
            // Safety: READY is published only after the producer initializes
            // the result cell; READY -> TAKEN is a unique consumer transition.
            Some(unsafe { (*self.result.get()).assume_init_read() })
        } else {
            None
        }
    }

    pub(super) fn register_waker(&self, waker: &Waker) {
        loop {
            match self.state.load(Ordering::Acquire) {
                ASYNC_RESULT_PENDING => {
                    // Safety: there is one consumer. If the publish CAS fails,
                    // this local clone is dropped before retry.
                    unsafe {
                        (*self.waiter.get()).write(waker.clone());
                    }

                    if self
                        .state
                        .compare_exchange(
                            ASYNC_RESULT_PENDING,
                            ASYNC_RESULT_WAITING,
                            Ordering::Release,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        return;
                    }

                    // Safety: the CAS failed, so no producer can observe this
                    // waiter cell as initialized by the WAITING state.
                    unsafe {
                        (*self.waiter.get()).assume_init_drop();
                    }
                }
                ASYNC_RESULT_WAITING => {
                    if self
                        .state
                        .compare_exchange(
                            ASYNC_RESULT_WAITING,
                            ASYNC_RESULT_UPDATING_WAKER,
                            Ordering::Acquire,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        // Safety: UPDATING_WAKER excludes the producer from
                        // reading the waiter cell while the consumer replaces it.
                        unsafe {
                            (*self.waiter.get()).assume_init_drop();
                            (*self.waiter.get()).write(waker.clone());
                        }
                        self.state.store(ASYNC_RESULT_WAITING, Ordering::Release);
                        return;
                    }
                }
                ASYNC_RESULT_UPDATING_WAKER | ASYNC_RESULT_WRITING => core::hint::spin_loop(),
                _ => return,
            }
        }
    }

    fn begin_completion(&self) -> Option<bool> {
        loop {
            match self.state.load(Ordering::Acquire) {
                ASYNC_RESULT_PENDING => {
                    if self
                        .state
                        .compare_exchange(
                            ASYNC_RESULT_PENDING,
                            ASYNC_RESULT_WRITING,
                            Ordering::Relaxed,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        return Some(false);
                    }
                }
                ASYNC_RESULT_WAITING => {
                    if self
                        .state
                        .compare_exchange(
                            ASYNC_RESULT_WAITING,
                            ASYNC_RESULT_WRITING,
                            Ordering::Acquire,
                            Ordering::Acquire,
                        )
                        .is_ok()
                    {
                        return Some(true);
                    }
                }
                ASYNC_RESULT_UPDATING_WAKER => core::hint::spin_loop(),
                _ => return None,
            }
        }
    }
}

impl<T> Drop for AsyncResultSlot<T> {
    fn drop(&mut self) {
        match self.state.load(Ordering::Acquire) {
            ASYNC_RESULT_READY => {
                // Safety: READY means the result cell is initialized and no
                // consumer took it because drop has exclusive access.
                unsafe {
                    self.result.get_mut().assume_init_drop();
                }
            }
            ASYNC_RESULT_WAITING => {
                // Safety: WAITING means the waiter cell is initialized and no
                // producer can access it because drop has exclusive access.
                unsafe {
                    self.waiter.get_mut().assume_init_drop();
                }
            }
            _ => {}
        }
    }
}

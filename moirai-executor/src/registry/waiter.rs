//! Owned registrations for task-completion notification.

#![forbid(unsafe_code)]

use std::{
    any::Any,
    panic::{AssertUnwindSafe, catch_unwind, resume_unwind},
    sync::{Arc, Mutex, MutexGuard},
    task::Waker,
};

/// The per-task pointer to lazily allocated waiter storage.
///
/// Callback-bearing destruction never runs while the pointer is locked.
pub(super) struct WaiterSetSlot {
    set: Mutex<Option<Arc<WaiterSet>>>,
}

impl WaiterSetSlot {
    pub(super) const fn new() -> Self {
        Self {
            set: Mutex::new(None),
        }
    }

    fn lock(&self) -> MutexGuard<'_, Option<Arc<WaiterSet>>> {
        self.set
            .lock()
            .expect("invariant: pointer-slot mutations contain no caller callbacks")
    }

    /// Return shared registration storage unless completion was published.
    pub(super) fn get_or_create(&self, completed: impl FnOnce() -> bool) -> Option<Arc<WaiterSet>> {
        let mut set = self.lock();
        if completed() {
            return None;
        }
        let result = Arc::clone(set.get_or_insert_with(|| Arc::new(WaiterSet::new())));
        drop(set);
        Some(result)
    }

    /// Detach the registration storage from its task state.
    pub(super) fn take(&self) -> Option<Arc<WaiterSet>> {
        self.lock().take()
    }

    pub(super) fn is_some(&self) -> bool {
        self.lock().is_some()
    }
}

#[derive(Clone, Copy)]
struct WaiterKey(usize);

struct WaiterSlots {
    closed: bool,
    inline: Option<Waker>,
    spill: Vec<Option<Waker>>,
}

impl WaiterSlots {
    const fn new() -> Self {
        Self {
            closed: false,
            inline: None,
            spill: Vec::new(),
        }
    }

    fn insert(&mut self, waker: Waker) -> Result<WaiterKey, Waker> {
        if self.closed {
            return Err(waker);
        }
        if self.inline.is_none() {
            self.inline = Some(waker);
            return Ok(WaiterKey(0));
        }
        if let Some((index, slot)) = self
            .spill
            .iter_mut()
            .enumerate()
            .find(|(_, slot)| slot.is_none())
        {
            *slot = Some(waker);
            return Ok(WaiterKey(index + 1));
        }
        self.spill.push(Some(waker));
        Ok(WaiterKey(self.spill.len()))
    }

    fn update(&mut self, key: WaiterKey, waker: Waker) -> Result<Waker, Waker> {
        if self.closed {
            return Err(waker);
        }
        let slot = if key.0 == 0 {
            &mut self.inline
        } else if let Some(slot) = self.spill.get_mut(key.0 - 1) {
            slot
        } else {
            return Err(waker);
        };
        match slot.take() {
            Some(previous) => {
                *slot = Some(waker);
                Ok(previous)
            }
            None => Err(waker),
        }
    }

    fn remove(&mut self, key: WaiterKey) -> Option<Waker> {
        let removed = if key.0 == 0 {
            self.inline.take()
        } else {
            self.spill.get_mut(key.0 - 1).and_then(Option::take)
        };
        if self.inline.is_none() && self.spill.iter().all(Option::is_none) {
            self.spill = Vec::new();
        }
        removed
    }

    fn close(&mut self) -> usize {
        self.closed = true;
        self.spill.len() + 1
    }
}

/// Shared storage whose lifetime is independent of the registry block.
pub(super) struct WaiterSet {
    slots: Mutex<WaiterSlots>,
}

impl WaiterSet {
    const fn new() -> Self {
        Self {
            slots: Mutex::new(WaiterSlots::new()),
        }
    }

    pub(super) fn register(self: &Arc<Self>, waker: Waker) -> Result<TaskWaiter, Waker> {
        let key = {
            let mut slots = self
                .slots
                .lock()
                .expect("invariant: waiter bookkeeping never panics while locked");
            slots.insert(waker)?
        };
        Ok(TaskWaiter {
            set: Arc::clone(self),
            key,
        })
    }

    fn update(&self, key: WaiterKey, waker: Waker) -> Result<Waker, Waker> {
        self.slots
            .lock()
            .expect("invariant: waiter bookkeeping never panics while locked")
            .update(key, waker)
    }

    fn remove(&self, key: WaiterKey) -> Option<Waker> {
        self.slots
            .lock()
            .expect("invariant: waiter bookkeeping never panics while locked")
            .remove(key)
    }

    pub(super) fn close(self: &Arc<Self>) -> WaiterNotifications {
        let slot_count = self
            .slots
            .lock()
            .expect("invariant: waiter bookkeeping never panics while locked")
            .close();
        WaiterNotifications {
            set: Some(Arc::clone(self)),
            slot_count,
        }
    }
}

/// An owned registration for one task-completion notification.
///
/// Dropping the value unregisters its waker immediately. Re-polling a future
/// should call [`TaskWaiter::update`] so this stable registration replaces its
/// waker rather than retaining poll history. The value intentionally does not
/// implement [`Clone`]: independent waits have independent slot identities,
/// even when they present the same waker.
///
/// # Examples
///
/// ```
/// use moirai_executor::TaskRegistry;
///
/// let registry = TaskRegistry::new();
/// let task_id = registry.register_task();
/// let mut waiter = registry
///     .register_waker(task_id, std::task::Waker::noop())
///     .expect("invariant: a newly registered task is pending");
/// assert!(waiter.update(std::task::Waker::noop()));
/// drop(waiter);
/// registry.mark_completed(task_id);
/// ```
#[must_use = "dropping the waiter unregisters the completion notification"]
pub struct TaskWaiter {
    set: Arc<WaiterSet>,
    key: WaiterKey,
}

impl TaskWaiter {
    /// Replace this registration's waker.
    ///
    /// Returns `false` after completion closes the registration set. The
    /// incoming waker is cloned before any lock is acquired, and destruction
    /// of the replaced or unused clone runs after every lock is released.
    ///
    /// # Panics
    ///
    /// Propagates a panic from the incoming waker's clone callback or from
    /// destruction of the replaced clone. The replacement is installed before
    /// destruction begins, so the panic cannot erase another registration.
    pub fn update(&mut self, waker: &Waker) -> bool {
        let incoming = waker.clone();
        match self.set.update(self.key, incoming) {
            Ok(previous) => {
                drop_and_resume(previous);
                true
            }
            Err(unused) => {
                drop_and_resume(unused);
                false
            }
        }
    }

    #[cfg(test)]
    pub(crate) fn spill_slots(&self) -> usize {
        self.set
            .slots
            .lock()
            .expect("invariant: waiter bookkeeping never panics while locked")
            .spill
            .len()
    }
}

impl Drop for TaskWaiter {
    fn drop(&mut self) {
        if let Some(waker) = self.set.remove(self.key) {
            // `Drop` cannot propagate a user callback panic. The scheduler's
            // established cleanup boundary contains it and applies the fatal
            // policy if destruction of its panic payload also panics.
            crate::schedule::job::run_caught(|| drop(waker));
        }
    }
}

/// A closed waiter set whose callbacks are claimed one at a time.
///
/// Each claim moves one waker out under the bookkeeping lock. Its callback then
/// runs without any lock held. Registrations not yet claimed remain removable
/// by their owning [`TaskWaiter`].
pub(super) struct WaiterNotifications {
    set: Option<Arc<WaiterSet>>,
    slot_count: usize,
}

impl WaiterNotifications {
    pub(super) const fn empty() -> Self {
        Self {
            set: None,
            slot_count: 0,
        }
    }

    pub(super) fn notify(self) {
        let mut first_panic = None;
        let mut later_panics = Vec::new();
        if let Some(set) = self.set {
            for index in 0..self.slot_count {
                if let Some(waker) = set.remove(WaiterKey(index)) {
                    retain_wake_panic(waker, &mut first_panic, &mut later_panics);
                }
            }
        }
        for payload in later_panics {
            crate::schedule::job::run_caught(|| drop(payload));
        }
        if let Some(payload) = first_panic {
            resume_unwind(payload);
        }
    }
}

fn retain_wake_panic(
    waker: Waker,
    first: &mut Option<Box<dyn Any + Send>>,
    later: &mut Vec<Box<dyn Any + Send>>,
) {
    if let Err(payload) = catch_unwind(AssertUnwindSafe(|| waker.wake())) {
        if first.is_none() {
            *first = Some(payload);
        } else {
            later.push(payload);
        }
    }
}

fn drop_and_resume(waker: Waker) {
    if let Err(payload) = catch_unwind(AssertUnwindSafe(|| drop(waker))) {
        resume_unwind(payload);
    }
}

pub(super) fn dispose_unregistered(waker: Waker) {
    drop_and_resume(waker);
}

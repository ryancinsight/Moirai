//! Shared datagram readiness with per-subscriber cancellation.

use super::{SocketLease, register_readiness};
use crate::Interest;
use crate::reactor::{IoReactor, waiter_cancellation::WaiterCancellation};
use std::io;
use std::net::UdpSocket;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, Weak};
use std::task::{Context, Poll, Wake, Waker};

fn lock<T>(value: &Mutex<T>) -> MutexGuard<'_, T> {
    value.lock().unwrap_or_else(PoisonError::into_inner)
}

#[derive(Default)]
struct ReadySlot {
    state: Mutex<(bool, Option<Waker>)>,
}

#[derive(Default)]
struct ReadySignal {
    slots: Mutex<Vec<Weak<ReadySlot>>>,
    dispatched: AtomicBool,
}

impl Wake for ReadySignal {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.dispatched.store(true, Ordering::Release);
        let slots: Vec<_> = {
            let mut registered = lock(&self.slots);
            registered.retain(|slot| slot.strong_count() > 0);
            registered.iter().filter_map(Weak::upgrade).collect()
        };
        for slot in slots {
            let waker = {
                let mut state = lock(&slot.state);
                state.0 = true;
                state.1.take()
            };
            if let Some(waker) = waker {
                waker.wake();
            }
        }
    }
}

pub(super) struct ReadyGroup {
    // Drop the cancellation owner before closing the descriptor. The reactor
    // owns only ReadySignal, not this group: no registration/socket cycle.
    registration: Mutex<Option<WaiterCancellation>>,
    socket: Arc<UdpSocket>,
    interest: Interest,
    signal: Arc<ReadySignal>,
}

impl ReadyGroup {
    pub(super) fn new(socket: Arc<UdpSocket>, interest: Interest) -> Arc<Self> {
        Arc::new(Self {
            registration: Mutex::new(None),
            socket,
            interest,
            signal: Arc::new(ReadySignal::default()),
        })
    }

    pub(super) fn subscribe(this: &Arc<Self>) -> UdpReady {
        let mut slots = lock(&this.signal.slots);
        let slot = Arc::new(ReadySlot {
            state: Mutex::new((this.signal.dispatched.load(Ordering::Acquire), None)),
        });
        slots.retain(|slot| slot.strong_count() > 0);
        slots.push(Arc::downgrade(&slot));
        UdpReady {
            slot,
            group: Arc::clone(this),
        }
    }
}

/// Independently owned UDP readiness subscriber.
///
/// One reactor registration wakes all subscribers. A ready event is a hint;
/// callers retry nonblocking I/O and rearm after `WouldBlock`. Each subscriber
/// retains its own task waker and readiness flag, so polling one cannot consume
/// a sibling's notification. Readiness is never a guarantee of successful I/O.
pub struct UdpReady {
    slot: Arc<ReadySlot>,
    group: Arc<ReadyGroup>,
}

impl UdpReady {
    /// Wait for the subscribed socket interest without periodic polling.
    ///
    /// # Errors
    /// Returns registration errors or a missing native reactor.
    pub fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        let already_ready = {
            let mut state = lock(&self.slot.state);
            let ready = std::mem::take(&mut state.0);
            if !ready
                && state
                    .1
                    .as_ref()
                    .is_none_or(|waker| !waker.will_wake(cx.waker()))
            {
                state.1 = Some(cx.waker().clone());
            }
            ready
        };
        let mut registration = lock(&self.group.registration);
        if already_ready {
            // Consumption and retirement are one registration-lock transaction.
            // Otherwise a late subscriber can observe the consumed flag and
            // retain a fired registration just before another waiter retires it.
            let retired = self
                .group
                .signal
                .dispatched
                .swap(false, Ordering::Acquire)
                .then(|| registration.take())
                .flatten();
            drop(registration);
            drop(retired);
            return Poll::Ready(Ok(()));
        }
        if registration.is_none() {
            self.group.signal.dispatched.store(false, Ordering::Release);
            let owner = SocketLease::from(&self.group.socket);
            let waker = Waker::from(Arc::clone(&self.group.signal));
            let signal_context = Context::from_waker(&waker);
            let result = IoReactor::with_current(|reactor| {
                let reactor = reactor
                    .ok_or_else(|| io::Error::other("UDP readiness requires a native reactor"))?;
                register_readiness(reactor, &owner, self.group.interest, &signal_context)
            });
            match result {
                Ok(token) => *registration = Some(token),
                Err(error) => return Poll::Ready(Err(error)),
            }
        }
        // Register first, then recheck the subscriber flag under its lock.
        // A concurrent dispatch cannot fall between the check and registration.
        let became_ready = std::mem::take(&mut lock(&self.slot.state).0);
        if became_ready {
            let retired = self
                .group
                .signal
                .dispatched
                .swap(false, Ordering::Acquire)
                .then(|| registration.take())
                .flatten();
            drop(registration);
            drop(retired);
            Poll::Ready(Ok(()))
        } else {
            Poll::Pending
        }
    }
}

impl Drop for UdpReady {
    fn drop(&mut self) {
        let retired = {
            let mut registration = lock(&self.group.registration);
            let empty = {
                let own = Arc::downgrade(&self.slot);
                let mut slots = lock(&self.group.signal.slots);
                slots.retain(|slot| !slot.ptr_eq(&own) && slot.strong_count() > 0);
                slots.is_empty()
            };
            if empty { registration.take() } else { None }
        };
        drop(retired);
    }
}

/// Polls nonblocking datagram I/O through shared, cancellable readiness.
pub struct UdpIoPoller {
    ready: UdpReady,
}

impl UdpIoPoller {
    pub(super) const fn new(ready: UdpReady) -> Self {
        Self { ready }
    }

    /// Attempt `operation`, then wait for readiness if it returns `WouldBlock`.
    ///
    /// The operation receives no stored user buffer; cancellation only retires
    /// readiness. A readiness hint is consumed before retrying, so a drained
    /// socket rearms rather than spinning.
    ///
    /// # Errors
    /// Propagates the operation's error or a readiness registration failure.
    pub fn poll_io<T>(
        &mut self,
        cx: &mut Context<'_>,
        mut operation: impl FnMut() -> io::Result<T>,
    ) -> Poll<io::Result<T>> {
        loop {
            match operation() {
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    match self.ready.poll_ready(cx) {
                        Poll::Ready(Ok(())) => {}
                        Poll::Ready(Err(error)) => return Poll::Ready(Err(error)),
                        Poll::Pending => return Poll::Pending,
                    }
                }
                result => return Poll::Ready(result),
            }
        }
    }
}

#[cfg(test)]
mod tests;

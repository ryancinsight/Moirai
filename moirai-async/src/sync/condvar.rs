use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use crate::sync::wait_queue::{WaitQueue, WaiterPoll};

use super::mutex::{Mutex, MutexGuard};

pub struct Condvar {
    state: std::sync::Mutex<CondvarState>,
}

struct CondvarState {
    waiters: WaitQueue<()>,
}

impl Condvar {
    pub fn new() -> Self {
        Self {
            state: std::sync::Mutex::new(CondvarState {
                waiters: WaitQueue::new(),
            }),
        }
    }

    pub async fn wait<'a, T>(&self, guard: MutexGuard<'a, T>) -> MutexGuard<'a, T> {
        let mutex_ref: &'a Mutex<T> = guard.mutex;
        let notified = CondvarNotifyFuture {
            condvar: self,
            id: None,
        };
        drop(guard);
        notified.await;
        mutex_ref.lock().await
    }

    pub async fn wait_while<'a, T, F>(
        &self,
        guard: MutexGuard<'a, T>,
        mut condition: F,
    ) -> MutexGuard<'a, T>
    where
        F: FnMut(&T) -> bool,
    {
        let mut guard = guard;
        while condition(&guard) {
            guard = self.wait(guard).await;
        }
        guard
    }

    pub fn notify_one(&self) {
        let mut state = self.state.lock().unwrap();
        if let Some(waker) = state.waiters.grant_oldest(()) {
            waker.wake();
        }
    }

    pub fn notify_all(&self) {
        let mut state = self.state.lock().unwrap();
        let wakers = state.waiters.grant_all(());
        drop(state);
        for waker in wakers {
            waker.wake();
        }
    }
}

impl Default for Condvar {
    fn default() -> Self {
        Self::new()
    }
}

struct CondvarNotifyFuture<'a> {
    condvar: &'a Condvar,
    id: Option<u64>,
}

impl<'a> Future for CondvarNotifyFuture<'a> {
    type Output = ();

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let mut state = self.condvar.state.lock().unwrap();

        if let Some(id) = self.id {
            match state.waiters.poll_waiter(id, cx.waker()) {
                WaiterPoll::Granted(()) => {
                    self.id = None;
                    return Poll::Ready(());
                }
                WaiterPoll::Pending => return Poll::Pending,
                WaiterPoll::NotRegistered => {}
            }
        }

        if self.id.is_none() {
            self.id = Some(state.waiters.register(cx.waker().clone()));
        }

        Poll::Pending
    }
}

impl<'a> Drop for CondvarNotifyFuture<'a> {
    fn drop(&mut self) {
        if let Some(id) = self.id {
            if let Ok(mut state) = self.condvar.state.lock() {
                state.waiters.deregister(id);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;


    #[test]
    fn test_condvar_notify_one() {
        let cv = Condvar::new();
        cv.notify_one();
    }

    #[test]
    fn test_condvar_notify_all() {
        let cv = Condvar::new();
        cv.notify_all();
    }
}

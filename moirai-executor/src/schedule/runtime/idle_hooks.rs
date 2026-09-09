//! Worker-side idle hooks for quiescent resource reclamation.
//!
//! Executor workers are long-lived: they outlive individual data-parallel
//! operations, so thread-local scratch buffers and allocator arenas can remain
//! resident for the process lifetime. A consumer that can name a cheap,
//! owner-thread-only reclamation step registers it here; workers run the hooks
//! after exhausting their spin budget and before parking for more work.
//!
//! Registration is bounded and allocation-free. The fixed capacity makes an
//! accepted registration a guarantee that the hook appears in every later
//! snapshot, instead of accepting entries that the old snapshot bound could
//! silently discard.

use std::fmt;
use std::sync::{Mutex, MutexGuard, OnceLock};

/// Maximum number of process-wide worker idle hooks.
///
/// A registration beyond this capacity returns
/// [`IdleHookRegistrationError::CapacityExhausted`] without changing the
/// registry.
pub const MAX_IDLE_HOOKS: usize = 16;

/// A worker idle hook: a plain function pointer called on the worker thread
/// immediately before it parks for work.
pub type IdleHook = fn();

/// Failure returned when a worker idle hook cannot be admitted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum IdleHookRegistrationError {
    /// The fixed registry has no remaining registration slot.
    CapacityExhausted,
}

impl fmt::Display for IdleHookRegistrationError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::CapacityExhausted => write!(
                formatter,
                "worker idle-hook registry capacity ({MAX_IDLE_HOOKS}) is exhausted"
            ),
        }
    }
}

impl std::error::Error for IdleHookRegistrationError {}

/// Fixed-capacity registry used by the process-wide API and isolated tests.
struct HookRegistry {
    hooks: Mutex<[Option<IdleHook>; MAX_IDLE_HOOKS]>,
}

impl HookRegistry {
    const fn new() -> Self {
        Self {
            hooks: Mutex::new([None; MAX_IDLE_HOOKS]),
        }
    }

    fn lock(&self) -> MutexGuard<'_, [Option<IdleHook>; MAX_IDLE_HOOKS]> {
        self.hooks
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn register(&self, hook: IdleHook) -> Result<(), IdleHookRegistrationError> {
        let mut hooks = self.lock();
        let Some(slot) = hooks.iter_mut().find(|entry| entry.is_none()) else {
            return Err(IdleHookRegistrationError::CapacityExhausted);
        };
        *slot = Some(hook);
        Ok(())
    }

    fn run(&self) {
        let snapshot = {
            let hooks = self.lock();
            *hooks
        };
        for hook in snapshot.into_iter().flatten() {
            hook();
        }
    }
}

/// Process-wide worker idle-hook registry.
static HOOKS: OnceLock<HookRegistry> = OnceLock::new();

fn registry() -> &'static HookRegistry {
    HOOKS.get_or_init(HookRegistry::new)
}

/// Registers `hook` to run on every worker thread before it parks for work.
///
/// Registration fills the next fixed slot and never allocates. Hooks run in
/// registration order, and duplicate function pointers are retained as
/// separate registrations. The registry lock is released before callbacks
/// execute, so a callback may register a later hook without deadlocking.
///
/// # Errors
/// Returns [`IdleHookRegistrationError::CapacityExhausted`] when all
/// [`MAX_IDLE_HOOKS`] slots are occupied. A rejected registration leaves every
/// existing slot unchanged.
pub fn register_idle_hook(hook: IdleHook) -> Result<(), IdleHookRegistrationError> {
    registry().register(hook)
}

/// Runs the hooks registered for the calling worker.
///
/// The fixed function-pointer snapshot is copied while the registry lock is
/// held and callbacks run after that lock is released. A callback panic
/// propagates to its worker and stops the current snapshot; it does not poison
/// the registry because callbacks never run while the lock is held.
pub fn run_idle_hooks() {
    registry().run();
}

#[cfg(test)]
mod tests;

//! Worker-side idle hooks for quiescent resource reclamation.
//!
//! Executor workers are long-lived: they outlive any single data-parallel
//! operation, so thread-local state a job builds up — scratch buffers, plan
//! caches, allocator arenas — stays resident for the process lifetime. A
//! consumer that can name a cheap, owner-thread-only reclamation step
//! registers it here; every worker runs all registered hooks at the one point
//! in its life that is both *on the owning thread* (thread-local storage can
//! only be released by its owner) and *quiescent by construction* (the worker
//! just exhausted its spin budget with no work found, and is about to block).
//!
//! Contract for hooks:
//!
//! - **Cheap when idle.** The hook runs on every park event, so it must return
//!   quickly when there is nothing to reclaim; any expensive sweep is the
//!   hook's own responsibility to throttle.
//! - **No reentrancy into the registry.** Hooks run *after* the registry lock
//!   is released, so a hook may call [`register_idle_hook`], but it must not
//!   block on anything a worker holding no other locks would not block on.
//! - **No ordering guarantees.** Hooks are independent; none may assume it
//!   runs before or after another.
//!
//! Registration is idempotent-safe but not deduplicating: registering the same
//! function twice runs it twice per park event.

use std::{
    panic::{catch_unwind, AssertUnwindSafe},
    sync::{Mutex, OnceLock},
};

/// A worker idle hook: a plain function pointer, called on the worker thread
/// right before it blocks for work.
pub type IdleHook = fn();

/// Failure returned when the bounded idle-hook registry has no free slot.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum IdleHookRegistrationError {
    /// The registry's fixed capacity is exhausted.
    CapacityExhausted,
}

/// The registry capacity is a process-wide resource bound. Idle hooks are
/// provider-level integrations, so a small fixed table avoids an allocation on
/// every worker park while still leaving room for the stack's providers.
const MAX_IDLE_HOOKS: usize = 16;

struct HookRegistry {
    hooks: [Option<IdleHook>; MAX_IDLE_HOOKS],
    len: usize,
}

impl HookRegistry {
    const fn new() -> Self {
        Self {
            hooks: [None; MAX_IDLE_HOOKS],
            len: 0,
        }
    }
}

/// Registered hooks. `OnceLock` keeps the registry allocation-free until the
/// first registration and makes initialization race-free.
static HOOKS: OnceLock<Mutex<HookRegistry>> = OnceLock::new();

fn registry() -> &'static Mutex<HookRegistry> {
    HOOKS.get_or_init(|| Mutex::new(HookRegistry::new()))
}

/// Registers `hook` to run on every worker thread right before it blocks for
/// work.
///
/// A registry poisoned by a panicking hook is recovered through its inner
/// value, keeping every prior registration. Registering the same function more
/// than once is allowed and runs it once per registration.
///
/// # Errors
///
/// Returns [`IdleHookRegistrationError::CapacityExhausted`] when all registry
/// slots are occupied. The caller must retain one authoritative registration
/// per provider rather than retrying after this error.
pub fn register_idle_hook(hook: IdleHook) -> Result<(), IdleHookRegistrationError> {
    let mut guard = registry()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if guard.len == MAX_IDLE_HOOKS {
        return Err(IdleHookRegistrationError::CapacityExhausted);
    }
    let slot = guard.len;
    guard.hooks[slot] = Some(hook);
    guard.len += 1;
    Ok(())
}

/// Runs every registered hook on the calling thread.
///
/// Called by worker threads at their quiescent point (spin budget exhausted,
/// no work found, about to block). The registry lock is held only to snapshot
/// the function pointers — never across a hook call — so a hook that calls
/// back into the registry cannot deadlock.
pub fn run_idle_hooks() {
    let snapshot = {
        let guard = registry()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        guard.hooks
    };
    for hook in snapshot.into_iter().flatten() {
        invoke_idle_hook(hook);
    }
}

/// Contains one hook panic so a reclamation integration cannot terminate a
/// worker that still owns runnable executor capacity. The standard panic hook
/// runs before `catch_unwind`, so the failure remains visible to the process's
/// configured diagnostics while later hooks and work continue.
fn invoke_idle_hook(hook: IdleHook) {
    if let Err(payload) = catch_unwind(AssertUnwindSafe(hook)) {
        drop(payload);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::sync::atomic::{AtomicUsize, Ordering};

    static RUNS: AtomicUsize = AtomicUsize::new(0);

    fn counting_hook() {
        RUNS.fetch_add(1, Ordering::Relaxed);
    }

    #[test]
    fn registration_is_accepted_and_counted() {
        register_idle_hook(counting_hook).expect("test registry has capacity");
        assert!(idle_hook_count() >= 1);
    }

    #[test]
    fn run_invokes_every_registered_hook_once_per_call() {
        register_idle_hook(counting_hook).expect("test registry has capacity");
        let before = RUNS.load(Ordering::Relaxed);
        run_idle_hooks();
        assert!(RUNS.load(Ordering::Relaxed) > before);
    }

    #[test]
    fn run_is_safe_with_an_empty_registry() {
        // A fresh test binary's registry may hold prior registrations; the
        // property under test is that `run_idle_hooks` never panics either way.
        run_idle_hooks();
    }

    #[test]
    fn a_panicking_hook_does_not_abort_the_next_hook() {
        fn panicking_hook() {
            panic!("idle hook failure is contained");
        }

        static FOLLOWING_HOOK_RUNS: AtomicUsize = AtomicUsize::new(0);

        fn following_hook() {
            FOLLOWING_HOOK_RUNS.fetch_add(1, Ordering::Relaxed);
        }

        let before = FOLLOWING_HOOK_RUNS.load(Ordering::Relaxed);
        invoke_idle_hook(panicking_hook);
        invoke_idle_hook(following_hook);
        assert_eq!(
            FOLLOWING_HOOK_RUNS.load(Ordering::Relaxed),
            before + 1,
            "a later idle hook must run after an earlier hook panics"
        );
    }

    #[test]
    fn hook_may_register_another_hook_without_deadlock() {
        fn reentrant_hook() {
            register_idle_hook(counting_hook).expect("test registry has capacity");
        }
        register_idle_hook(reentrant_hook).expect("test registry has capacity");
        // The reentrant registration happens while no lock is held, so this
        // must complete rather than deadlock.
        run_idle_hooks();
    }

    fn idle_hook_count() -> usize {
        registry()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len
    }
}

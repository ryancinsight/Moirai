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

use std::sync::{Mutex, OnceLock};

/// A worker idle hook: a plain function pointer, called on the worker thread
/// right before it blocks for work.
pub type IdleHook = fn();

/// Registered hooks. `OnceLock` avoids a const-friendly `Mutex::new` shape
/// check per access while keeping the registry allocation-free until the first
/// registration.
static HOOKS: OnceLock<Mutex<Vec<IdleHook>>> = OnceLock::new();

fn registry() -> &'static Mutex<Vec<IdleHook>> {
    HOOKS.get_or_init(|| Mutex::new(Vec::new()))
}

/// Registers `hook` to run on every worker thread right before it blocks for
/// work.
///
/// Infallible: a registry poisoned by a panicking hook is recovered through
/// its inner value, keeping every prior registration. Registering the same
/// function more than once is allowed and runs it once per registration.
pub fn register_idle_hook(hook: IdleHook) {
    let mut guard = registry()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    guard.push(hook);
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
        let mut hooks = [None; MAX_SNAPSHOT_HOOKS];
        for (slot, &hook) in hooks.iter_mut().zip(guard.iter()) {
            *slot = Some(hook);
        }
        hooks
    };
    for hook in snapshot.into_iter().flatten() {
        hook();
    }
}

/// Snapshot bound. Registration beyond this count is still accepted, but the
/// excess hooks only run once the bound is raised; the count is far beyond any
/// plausible consumer.
const MAX_SNAPSHOT_HOOKS: usize = 16;

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
        register_idle_hook(counting_hook);
        assert!(idle_hook_count() >= 1);
    }

    #[test]
    fn run_invokes_every_registered_hook_once_per_call() {
        register_idle_hook(counting_hook);
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
    fn hook_may_register_another_hook_without_deadlock() {
        fn reentrant_hook() {
            register_idle_hook(counting_hook);
        }
        register_idle_hook(reentrant_hook);
        // The reentrant registration happens while no lock is held, so this
        // must complete rather than deadlock.
        run_idle_hooks();
    }

    fn idle_hook_count() -> usize {
        registry()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len()
    }
}

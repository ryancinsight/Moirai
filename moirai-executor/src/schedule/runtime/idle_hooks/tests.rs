use super::{HookRegistry, IdleHookRegistrationError, MAX_IDLE_HOOKS};
use std::{cell::RefCell, panic::catch_unwind, sync::Arc};

thread_local! {
    static CALLS: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
    static ACTIVE_REGISTRY: RefCell<Option<Arc<HookRegistry>>> = const { RefCell::new(None) };
}

fn first() {
    CALLS.with_borrow_mut(|calls| calls.push(1));
}

fn second() {
    CALLS.with_borrow_mut(|calls| calls.push(2));
}

fn take_calls() -> Vec<usize> {
    CALLS.with_borrow_mut(std::mem::take)
}

#[test]
fn empty_snapshot_calls_nothing() {
    let registry = HookRegistry::new();
    assert_eq!(take_calls(), []);
    registry.run();
    assert_eq!(take_calls(), []);
}

#[test]
fn capacity_rejection_preserves_order_and_duplicates() {
    let registry = HookRegistry::new();
    for slot in 0..MAX_IDLE_HOOKS {
        let hook = if slot % 2 == 0 { first } else { second };
        assert_eq!(registry.register(hook), Ok(()));
    }
    assert_eq!(
        registry.register(second),
        Err(IdleHookRegistrationError::CapacityExhausted)
    );
    let expected: Vec<_> = (0..MAX_IDLE_HOOKS).map(|slot| slot % 2 + 1).collect();
    registry.run();
    assert_eq!(take_calls(), expected);
    registry.run();
    assert_eq!(take_calls(), expected);
}

#[test]
fn reentrant_registration_appears_only_in_later_snapshots() {
    fn register_second() {
        first();
        ACTIVE_REGISTRY.with_borrow(|registry| {
            assert_eq!(
                registry
                    .as_ref()
                    .expect("test registry is installed")
                    .register(second),
                Ok(())
            );
        });
    }

    let registry = Arc::new(HookRegistry::new());
    ACTIVE_REGISTRY.with_borrow_mut(|active| *active = Some(Arc::clone(&registry)));
    assert_eq!(registry.register(register_second), Ok(()));
    registry.run();
    assert_eq!(take_calls(), [1]);
    registry.run();
    assert_eq!(take_calls(), [1, 2]);
    registry.run();
    assert_eq!(take_calls(), [1, 2, 2]);
    ACTIVE_REGISTRY.with_borrow_mut(|active| *active = None);
}

#[test]
fn panic_stops_snapshot_without_poisoning_registration() {
    fn panic_after_first() {
        first();
        panic!("hook failure");
    }

    let registry = HookRegistry::new();
    assert_eq!(registry.register(panic_after_first), Ok(()));
    assert_eq!(registry.register(second), Ok(()));
    let panic = catch_unwind(|| registry.run()).expect_err("callback panic propagates");
    assert_eq!(panic.downcast_ref::<&str>(), Some(&"hook failure"));
    assert_eq!(take_calls(), [1]);
    assert!(!registry.hooks.is_poisoned());
    assert_eq!(registry.register(second), Ok(()));
    assert_eq!(
        registry
            .hooks
            .lock()
            .expect("callback held no registry lock")
            .iter()
            .flatten()
            .count(),
        3
    );
}

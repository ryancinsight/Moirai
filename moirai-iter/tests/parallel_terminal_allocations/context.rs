use super::support::{
    context_for_each_allocation_budget, context_for_each_byte_budget,
    context_large_limit_map_values, context_map_allocation_budget, context_map_byte_budget,
    context_map_values, context_pending_for_each, context_pending_map_values,
    context_pending_map_values_with_limit, warmed_allocation_ledger, CONTEXT_MAP_LEN,
};
use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

fn source() -> Vec<u64> {
    (0..CONTEXT_MAP_LEN as u64)
        .map(|value| value.wrapping_mul(19).wrapping_add(7))
        .collect()
}

fn values_are_exact(mapped: &[u64]) -> bool {
    mapped.iter().enumerate().all(|(index, value)| {
        let input = (index as u64).wrapping_mul(19).wrapping_add(7);
        *value == input.wrapping_mul(5).wrapping_add(3)
    })
}

#[test]
fn parallel_context_async_map_excludes_topology_allocations() {
    let (mapped, allocations, bytes) =
        warmed_allocation_ledger(source(), source(), context_map_values);
    assert!(values_are_exact(&mapped));
    assert!(allocations <= context_map_allocation_budget());
    assert!(bytes <= context_map_byte_budget());
}

#[test]
fn large_context_limit_does_not_reserve_for_short_inputs() {
    const FIXED_ALLOCATION_BUDGET: usize = 20;
    const FIXED_BYTE_BUDGET: usize = 8_192;

    let (empty, empty_allocations, empty_bytes) =
        warmed_allocation_ledger(Vec::new(), Vec::new(), context_large_limit_map_values);
    let (single, single_allocations, single_bytes) =
        warmed_allocation_ledger(vec![5], vec![5], context_large_limit_map_values);

    assert!(empty.is_empty());
    assert_eq!(single, [28]);
    assert!(empty_allocations <= FIXED_ALLOCATION_BUDGET);
    assert!(single_allocations <= FIXED_ALLOCATION_BUDGET);
    assert!(empty_bytes <= FIXED_BYTE_BUDGET);
    assert!(single_bytes <= FIXED_BYTE_BUDGET);
}

#[test]
fn parallel_context_pending_map_records_entry_allocation_ledger() {
    let (mapped, allocations, bytes) =
        warmed_allocation_ledger(source(), source(), context_pending_map_values);
    assert!(values_are_exact(&mapped));
    assert!(allocations <= context_map_allocation_budget());
    assert!(bytes <= context_map_byte_budget());
}

#[test]
#[ignore = "allocation attribution instrument; run explicitly with --nocapture"]
fn retained_wake_allocation_attribution() {
    for max_concurrent in [1, 8, 24] {
        let (mapped, allocations, bytes) = warmed_allocation_ledger(
            (source(), max_concurrent),
            (source(), max_concurrent),
            |(data, limit)| context_pending_map_values_with_limit(data, limit),
        );
        assert!(values_are_exact(&mapped));
        eprintln!(
            "retained wake limit {max_concurrent:>2}: {allocations:>3} allocations, +             {bytes:>6} gross bytes"
        );
    }
}

#[test]
fn parallel_context_pending_for_each_records_entry_allocation_ledger() {
    let visits = || {
        Arc::new(
            (0..CONTEXT_MAP_LEN)
                .map(|_| AtomicUsize::new(0))
                .collect::<Vec<_>>(),
        )
    };
    let input = || (0..CONTEXT_MAP_LEN).collect::<Vec<_>>();
    let measured_visits = visits();
    let ((), allocations, bytes) = warmed_allocation_ledger(
        (input(), visits()),
        (input(), Arc::clone(&measured_visits)),
        |(data, counts)| context_pending_for_each(data, counts),
    );

    assert!(measured_visits
        .iter()
        .all(|count| count.load(Ordering::SeqCst) == 1));
    assert!(allocations <= context_for_each_allocation_budget());
    assert!(bytes <= context_for_each_byte_budget());
}

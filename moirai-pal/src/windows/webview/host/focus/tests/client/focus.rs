//! UIA selects one field; one native parent entry preserves that selection.

mod native;

use super::super::input::Observations;
use serde_json::json;
use std::time::Instant;
use windows::Win32::UI::Accessibility::IUIAutomationElement;

pub(super) fn enter(
    element: &IUIAutomationElement,
    handle: usize,
    deadline: Instant,
    observations: &mut Observations,
) {
    select(element, deadline, observations);
    let entry = native::enter(handle, deadline, observations)
        .expect("one actual native parent focus entry");
    observations.record(json!({"kind":"native_parent_entry", "stage":"entered",
        "parent":format!("{:?}",entry.parent), "previous":format!("{:?}",entry.previous),
        "recipient":format!("{:?}",entry.recipient), "client_thread_id":entry.client_thread,
        "owner_thread_id":entry.owner_thread,"owner_process_id":entry.owner_process}));
    let focused = observations
        .operation("CurrentHasKeyboardFocus_after_parent_entry", || unsafe {
            element.CurrentHasKeyboardFocus()
        })
        .expect("same selected field after native parent entry")
        .as_bool();
    observations.record(
        json!({"kind":"native_parent_entry", "stage":"selected_field",
        "focused":focused}),
    );
    assert!(
        focused,
        "native parent entry preserves the selected UIA field"
    );
    assert!(
        Instant::now() < deadline,
        "native entry retains the original deadline"
    );
}

fn select(element: &IUIAutomationElement, deadline: Instant, observations: &mut Observations) {
    observations
        .operation("SetFocus", || unsafe { element.SetFocus() })
        .expect("actual UIA SetFocus");
    // SetFocus is issued once; only its actual completion property is sampled.
    let started = Instant::now();
    let mut queries = 0_usize;
    let mut previous = None;
    loop {
        queries = queries
            .checked_add(1)
            .expect("bounded UIA focus observations");
        let actual = observations
            .operation("CurrentHasKeyboardFocus", || unsafe {
                element.CurrentHasKeyboardFocus()
            })
            .expect("actual UIA focus")
            .as_bool();
        if previous != Some(actual) {
            observations.record(json!({"kind":"focus_snapshot", "query":queries,"focused":actual}));
            previous = Some(actual);
        }
        if actual {
            observations.record(json!({"kind":"focus_completed", "queries":queries,
                "elapsed_ns":started.elapsed().as_nanos().to_string(),"apartment":"MTA"}));
            break;
        }
        assert!(
            Instant::now() < deadline,
            "actual UIA focus did not reach the field"
        );
        std::thread::yield_now();
    }
}

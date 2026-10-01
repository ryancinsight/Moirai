//! Slot table protocol without the kernel: the tests play the driver by
//! writing the record through the pointers `request` hands out, then dequeue
//! with `complete`. Nothing here calls a foreign function.

use std::sync::atomic::Ordering;
use windows::Win32::Foundation::{
    NTSTATUS, STATUS_CANCELLED, STATUS_INVALID_PARAMETER, STATUS_SUCCESS,
};

use windows::Win32::System::IO::OVERLAPPED;

use super::{Completion, SlotTable, Token};
use crate::Interest;
use crate::windows::afd::abi;

/// `POLL_RECEIVE`, the event bit a readable socket reports.
const RECEIVE: u32 = 0x01;

fn publish(table: &SlotTable, socket: usize) -> Token {
    let index = table.claim().expect("a free slot");
    table.publish(index, socket, abi::poll_info(Interest::READABLE))
}

/// Complete the request as the driver would: fill the record, then dequeue.
fn finish(table: &SlotTable, token: Token, status: NTSTATUS, events: u32) -> Completion {
    let request = table.request(token.index());
    // SAFETY: the pointers address the live record of a published slot and no
    // other thread exists in these tests.
    unsafe {
        (&raw mut (*request.status_block).Anonymous.Status).write(status);
        (&raw mut (*request.info).handles[0].events).write(events);
    }
    table.complete(request.context.cast_mut().cast::<OVERLAPPED>())
}

#[test]
fn a_finished_poll_reports_its_socket_and_releases_the_slot() {
    let table = SlotTable::new(2);
    let token = publish(&table, 7);
    assert_eq!(table.outstanding(), 1);
    let Completion::Finished {
        token: finished,
        readiness,
        ..
    } = finish(&table, token, STATUS_SUCCESS, RECEIVE)
    else {
        panic!("a successful poll must finish");
    };
    assert_eq!(finished, token);
    assert!(readiness.readable && !readiness.writable);
    assert_eq!(readiness.fd as usize, 7);
    assert_eq!(table.outstanding(), 0);
    let reused = publish(&table, 8);
    assert_eq!(reused.index(), token.index());
    assert_ne!(reused, token, "a reused slot issues a new generation");
}

#[test]
fn a_cancelled_status_is_cancelled_and_releases_the_slot() {
    let table = SlotTable::new(1);
    let token = publish(&table, 1);
    assert!(matches!(
        finish(&table, token, STATUS_CANCELLED, 0),
        Completion::Cancelled
    ));
    assert_eq!(table.outstanding(), 0);
}

#[test]
fn a_request_marker_suppresses_a_completion_after_the_canceller_left() {
    let table = SlotTable::new(1);
    let token = publish(&table, 1);
    assert!(table.begin_cancel(token));
    table.end_cancel(token.index());
    assert!(matches!(
        finish(&table, token, STATUS_SUCCESS, RECEIVE),
        Completion::Cancelled
    ));
    assert_eq!(table.outstanding(), 0);
}

#[test]
fn a_completion_inside_a_cancel_leaves_the_release_to_the_canceller() {
    let table = SlotTable::new(1);
    let token = publish(&table, 1);
    assert!(table.begin_cancel(token));
    assert!(matches!(
        finish(&table, token, STATUS_SUCCESS, RECEIVE),
        Completion::Cancelled
    ));
    assert_eq!(table.outstanding(), 1, "the canceller still holds the slot");
    table.end_cancel(token.index());
    assert_eq!(table.outstanding(), 0);
}

#[test]
fn stale_and_repeated_cancels_do_nothing() {
    let table = SlotTable::new(1);
    let stale = publish(&table, 1);
    assert!(matches!(
        finish(&table, stale, STATUS_SUCCESS, RECEIVE),
        Completion::Finished { .. }
    ));
    let live = publish(&table, 2);
    assert!(!table.begin_cancel(stale));
    assert!(table.begin_cancel(live));
    assert!(
        !table.begin_cancel(live),
        "a second cancel finds the marker set"
    );
}

#[test]
fn a_failure_status_is_reported_with_the_socket_and_releases_the_slot() {
    let table = SlotTable::new(1);
    let token = publish(&table, 3);
    let Completion::Finished {
        token: finished,
        status,
        ..
    } = finish(&table, token, STATUS_INVALID_PARAMETER, 0)
    else {
        panic!("a failed poll must finish, not read as cancelled");
    };
    assert_eq!(finished, token);
    assert_eq!(status, STATUS_INVALID_PARAMETER);
    assert_eq!(table.outstanding(), 0);
}

#[test]
fn the_generation_wraps_at_the_top_of_its_range() {
    let table = SlotTable::new(1);
    table.slots[0]
        .word
        .store(u64::from(u32::MAX) << 32, Ordering::Relaxed);
    let last = publish(&table, 1);
    assert_eq!(last.generation, 0, "u32::MAX wraps to 0 without panicking");
    assert!(matches!(
        finish(&table, last, STATUS_SUCCESS, RECEIVE),
        Completion::Finished { .. }
    ));
    let next = publish(&table, 1);
    assert_eq!(next.generation, 1);
    assert_ne!(next, last);
}

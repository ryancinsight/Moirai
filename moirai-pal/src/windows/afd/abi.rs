//! Kernel-facing layout of `IOCTL_AFD_POLL` and the mapping between Moirai
//! interests and AFD event bits.
//!
//! The control code, `AFD_POLL_INFO` layout, and event bits are not documented
//! by Microsoft. They are taken from mio 1.2.3 (`src/sys/windows/afd.rs` and
//! `event.rs`), which ships on them; the layout is pinned by const assertions
//! so a divergence breaks the build instead of corrupting a request.

use std::mem::{align_of, offset_of, size_of};

use windows::Win32::Foundation::{HANDLE, NTSTATUS};

use crate::{Event, Interest, RawFd};

/// Control code of a poll request on an AFD device handle.
pub(super) const IOCTL_AFD_POLL: u32 = 0x0001_2024;

const POLL_RECEIVE: u32 = 0x01;
const POLL_SEND: u32 = 0x04;
const POLL_DISCONNECT: u32 = 0x08;
const POLL_ABORT: u32 = 0x10;
const POLL_LOCAL_CLOSE: u32 = 0x20;
const POLL_ACCEPT: u32 = 0x80;
const POLL_CONNECT_FAIL: u32 = 0x100;

const READABLE: u32 = POLL_RECEIVE | POLL_DISCONNECT | POLL_ACCEPT | POLL_ABORT | POLL_CONNECT_FAIL;
const WRITABLE: u32 = POLL_SEND | POLL_ABORT | POLL_CONNECT_FAIL;

/// One socket entry of an AFD poll request.
#[derive(Clone, Copy)]
#[repr(C)]
pub(super) struct AfdPollHandle {
    pub(super) handle: HANDLE,
    pub(super) events: u32,
    pub(super) status: NTSTATUS,
}

/// Input and output buffer of `IOCTL_AFD_POLL`: a poll over exactly one socket.
#[derive(Clone, Copy)]
#[repr(C)]
pub(super) struct AfdPollInfo {
    pub(super) timeout: i64,
    pub(super) number_of_handles: u32,
    pub(super) exclusive: u32,
    pub(super) handles: [AfdPollHandle; 1],
}

const _: () = {
    assert!(size_of::<AfdPollHandle>() == 2 * size_of::<usize>());
    assert!(size_of::<AfdPollInfo>() == 16 + size_of::<AfdPollHandle>());
    assert!(align_of::<AfdPollInfo>() == align_of::<usize>());
    assert!(offset_of!(AfdPollInfo, number_of_handles) == 8);
    assert!(offset_of!(AfdPollInfo, handles) == 16);
};

impl AfdPollInfo {
    /// A request polling `base_socket` for `interest` until it reports.
    pub(super) fn new(base_socket: HANDLE, interest: Interest) -> Self {
        let mut events = POLL_LOCAL_CLOSE;
        if interest.readable {
            events |= READABLE;
        }
        if interest.writable {
            events |= WRITABLE;
        }
        Self {
            timeout: i64::MAX,
            number_of_handles: 1,
            exclusive: 0,
            handles: [AfdPollHandle {
                handle: base_socket,
                events,
                status: NTSTATUS(0),
            }],
        }
    }

    /// Readiness reported by a completed request for `socket`.
    ///
    /// A local close carries no readiness and is reported as error plus hangup
    /// with neither direction set, the invalidation convention of the reactor.
    pub(super) fn readiness(&self, socket: usize) -> Event {
        let reported = if self.number_of_handles == 0 {
            0
        } else {
            self.handles[0].events
        };
        let closed = reported & POLL_LOCAL_CLOSE != 0;
        Event {
            fd: socket as RawFd,
            readable: !closed && reported & READABLE != 0,
            writable: !closed && reported & WRITABLE != 0,
            error: closed || reported & (POLL_ABORT | POLL_CONNECT_FAIL) != 0,
            hangup: closed || reported & POLL_DISCONNECT != 0,
        }
    }
}

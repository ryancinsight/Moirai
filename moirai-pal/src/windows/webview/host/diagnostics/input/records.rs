//! Copy native callback values now; serialize and trace after the callback.

use crate::{webview_limits::MAX_WEBVIEW_EVENTS, webview_startup::Deferred};
use serde_json::{Value, json};
use std::{
    cell::{Cell, RefCell},
    time::Instant,
};
use windows::Win32::UI::{
    Input::KeyboardAndMouse::{
        GetAsyncKeyState, GetFocus, GetKeyState, VK_CONTROL, VK_MENU, VK_SHIFT,
    },
    WindowsAndMessaging::GetForegroundWindow,
};

#[derive(Clone, Copy, Debug)]
pub(super) enum Event {
    Focus {
        gained: bool,
    },
    Move {
        reason: i32,
        handled: bool,
    },
    Key {
        kind: i32,
        key: u32,
        lparam: i32,
        handled: bool,
    },
    Error {
        operation: &'static str,
        hresult: i32,
    },
}

#[derive(Clone, Copy, Debug)]
struct Snapshot {
    sequence: u64,
    elapsed_ns: u128,
    event: Event,
    focus: usize,
    foreground: usize,
    owner_keys: [i16; 3],
    asynchronous_keys: [i16; 3],
}

pub(super) struct Queue {
    pending: RefCell<Deferred<Snapshot, MAX_WEBVIEW_EVENTS>>,
    sequence: Cell<u64>,
    contended: Cell<u64>,
    pub(super) closed: Cell<bool>,
    started: Instant,
}

impl Queue {
    pub(super) fn new() -> Self {
        Self {
            pending: RefCell::new(Deferred::new()),
            sequence: Cell::new(0),
            contended: Cell::new(0),
            closed: Cell::new(false),
            started: Instant::now(),
        }
    }

    pub(super) fn push(&self, event: Event) {
        if self.closed.get() {
            return;
        }
        let sequence = self.sequence.get().saturating_add(1);
        self.sequence.set(sequence);
        let Ok(mut pending) = self.pending.try_borrow_mut() else {
            self.contended.set(self.contended.get().saturating_add(1));
            return;
        };
        // SAFETY: these immediate user32 reads retain no pointers. GetKeyState
        // belongs to this host callback thread, not the browser's input queue;
        // GetAsyncKeyState is a separate non-atomic system observation.
        let (focus, foreground, owner_keys, asynchronous_keys) = unsafe {
            (
                GetFocus().0 as usize,
                GetForegroundWindow().0 as usize,
                [VK_CONTROL, VK_SHIFT, VK_MENU].map(|key| GetKeyState(i32::from(key.0))),
                [VK_CONTROL, VK_SHIFT, VK_MENU].map(|key| GetAsyncKeyState(i32::from(key.0))),
            )
        };
        pending.push(Snapshot {
            sequence,
            elapsed_ns: self.started.elapsed().as_nanos(),
            event,
            focus,
            foreground,
            owner_keys,
            asynchronous_keys,
        });
    }

    pub(super) fn flush(&self, state: &super::super::state::Shared) {
        // Release callback storage before formatting or invoking a subscriber.
        let (records, omitted) = self.pending.borrow_mut().take();
        let contended = self.contended.replace(0);
        let mut state = state.borrow_mut();
        for record in records.into_iter().flatten() {
            state.failed |= matches!(record.event, Event::Error { .. });
            state.record(&record.value());
        }
        if omitted != 0 || contended != 0 {
            state.failed = true;
            state.record(&json!({"kind":"controller_input_loss",
                "capacity_omitted":omitted.to_string(),"contended":contended.to_string()}));
        }
    }
}

impl Snapshot {
    fn value(self) -> Value {
        let event = match self.event {
            Event::Focus { gained } => json!({"name":if gained {"GotFocus"} else {"LostFocus"}}),
            Event::Move { reason, handled } => {
                json!({"name":"MoveFocusRequested","reason":reason,"handled":handled})
            }
            Event::Key {
                kind,
                key,
                lparam,
                handled,
            } => json!({"name":"AcceleratorKeyPressed","event_kind":kind,
                    "virtual_key":key,"lparam":lparam,"handled":handled}),
            Event::Error { operation, hresult } => {
                json!({"name":"read_error","operation":operation,"hresult":hresult})
            }
        };
        json!({"kind":"controller_input","sequence":self.sequence.to_string(),
            "elapsed_ns":self.elapsed_ns.to_string(),"event":event,
            "focus_hwnd":self.focus.to_string(),"foreground_hwnd":self.foreground.to_string(),
            "owner_keys":self.owner_keys,"asynchronous_keys":self.asynchronous_keys})
    }
}

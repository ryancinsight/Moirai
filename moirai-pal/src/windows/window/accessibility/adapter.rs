//! Windows UI Automation adapter and bounded action delivery.

use accesskit::{Action, ActionData, ActionHandler, ActivationHandler, TreeUpdate};
use accesskit_windows::{HWND, SubclassingAdapter};
use std::collections::VecDeque;
use std::ffi::c_void;
use std::io;
use std::sync::{Arc, Mutex};
use windows::Win32::Foundation::*;
use windows::Win32::UI::WindowsAndMessaging::PostMessageW;

use super::{
    ACCESSIBILITY_WAKE_MESSAGE, AccessibilityAction, AccessibilityTree, MAX_ACCESSIBILITY_ACTIONS,
};

/// One action requested by a screen reader or UI Automation client.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AccessibilityActionRequest {
    /// Stable target node identity.
    pub target_node: u64,
    /// Requested operation.
    pub action: AccessibilityAction,
    /// Replacement value, when the action carries text.
    pub value: Option<String>,
    /// Signed numeric adjustment, when the action increments or decrements.
    pub delta: Option<i8>,
}

/// Windows AccessKit adapter owned by the window thread.
pub struct WindowsAccessibilityAdapter {
    adapter: SubclassingAdapter,
    current: Arc<Mutex<AccessibilityTree>>,
    queue: Arc<Mutex<ActionQueue>>,
}

impl WindowsAccessibilityAdapter {
    /// Installs the UI Automation adapter on a hidden window.
    pub fn new(hwnd: HWND, initial_tree: AccessibilityTree) -> io::Result<Self> {
        initial_tree.validate()?;
        let current = Arc::new(Mutex::new(initial_tree));
        let queue = Arc::new(Mutex::new(ActionQueue::default()));
        let activation = ActivationSource {
            current: Arc::clone(&current),
        };
        let action = ActionSink {
            hwnd_raw: hwnd.0 as isize,
            queue: Arc::clone(&queue),
        };
        let adapter = SubclassingAdapter::new(hwnd, activation, action);
        Ok(Self {
            adapter,
            current,
            queue,
        })
    }

    /// Replaces the current tree and raises the bounded native events.
    pub fn update(&mut self, tree: AccessibilityTree) -> io::Result<()> {
        tree.validate()?;
        let update = tree.to_accesskit();
        let mut current = self
            .current
            .lock()
            .map_err(|_| io::Error::other("accessibility tree lock is poisoned"))?;
        *current = tree;
        drop(current);
        if let Some(events) = self.adapter.update_if_active(|| update) {
            events.raise();
        }
        Ok(())
    }

    pub(crate) fn take_actions(&mut self) -> io::Result<Vec<AccessibilityActionRequest>> {
        let mut queue = self
            .queue
            .lock()
            .map_err(|_| io::Error::other("accessibility action queue lock is poisoned"))?;
        if queue.wake_failed {
            queue.wake_failed = false;
            queue.actions.clear();
            return Err(io::Error::other(
                "accessibility action queue could not wake the window thread",
            ));
        }
        if queue.overflowed {
            queue.overflowed = false;
            queue.actions.clear();
            return Err(io::Error::other(
                "accessibility action queue exceeded its bound",
            ));
        }
        Ok(queue.actions.drain(..).collect())
    }

    pub(crate) fn has_pending_actions(&self) -> bool {
        self.queue
            .lock()
            .map(|queue| !queue.actions.is_empty() || queue.overflowed || queue.wake_failed)
            .unwrap_or(true)
    }
}

struct ActivationSource {
    current: Arc<Mutex<AccessibilityTree>>,
}

impl ActivationHandler for ActivationSource {
    fn request_initial_tree(&mut self) -> Option<TreeUpdate> {
        self.current.lock().ok().map(|tree| tree.to_accesskit())
    }
}

#[derive(Default)]
struct ActionQueue {
    actions: VecDeque<AccessibilityActionRequest>,
    overflowed: bool,
    wake_failed: bool,
}

struct ActionSink {
    // A scalar handle keeps the callback `Send`; the adapter invokes it from
    // its own callback context, while the window thread owns the HWND.
    hwnd_raw: isize,
    queue: Arc<Mutex<ActionQueue>>,
}

impl ActionHandler for ActionSink {
    fn do_action(&mut self, request: accesskit::ActionRequest) {
        let Some(request) = action_request(request) else {
            return;
        };
        let mut queue = match self.queue.lock() {
            Ok(queue) => queue,
            Err(_) => return,
        };
        if queue.actions.len() >= MAX_ACCESSIBILITY_ACTIONS {
            queue.overflowed = true;
        } else {
            queue.actions.push_back(request);
        }
        // SAFETY: the HWND belongs to the owning window thread and the message
        // carries no pointer; Windows copies both scalar parameters before the
        // call returns. The queue is the only cross-thread state.
        if unsafe {
            PostMessageW(
                // SAFETY: `hwnd_raw` came from the live HWND installed with
                // this adapter and is only reconstructed for this message.
                Some(HWND(self.hwnd_raw as *mut c_void)),
                ACCESSIBILITY_WAKE_MESSAGE,
                WPARAM(0),
                LPARAM(0),
            )
        }
        .is_err()
        {
            queue.wake_failed = true;
        }
    }
}

pub(crate) fn action_request(
    request: accesskit::ActionRequest,
) -> Option<AccessibilityActionRequest> {
    let (action, delta) = match request.action {
        Action::Click => (AccessibilityAction::Activate, None),
        Action::Focus => (AccessibilityAction::Focus, None),
        Action::ReplaceSelectedText => (AccessibilityAction::SetValue, None),
        Action::Increment => (AccessibilityAction::AdjustValue, Some(1)),
        Action::Decrement => (AccessibilityAction::AdjustValue, Some(-1)),
        Action::Expand | Action::Collapse => (AccessibilityAction::Open, None),
        _ => return None,
    };
    let value = match request.data {
        Some(ActionData::Value(value)) => Some(value.into()),
        _ => None,
    };
    Some(AccessibilityActionRequest {
        target_node: request.target_node.0,
        action,
        value,
        delta,
    })
}

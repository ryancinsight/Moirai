//! Bounded Windows UI Automation access for a thread-owned native window.

mod adapter;
mod contract;

#[cfg(test)]
mod tests;

use windows::Win32::UI::WindowsAndMessaging::WM_APP;

pub(crate) const ACCESSIBILITY_WAKE_MESSAGE: u32 = WM_APP + 0x36;

pub use adapter::AccessibilityActionRequest;
pub use adapter::WindowsAccessibilityAdapter;
#[cfg(test)]
pub(crate) use adapter::action_request;
pub use contract::{
    AccessibilityAction, AccessibilityNode, AccessibilityRole, AccessibilityTree,
    MAX_ACCESSIBILITY_ACTIONS, MAX_ACCESSIBILITY_NODES, MAX_ACCESSIBILITY_TEXT_BYTES,
};

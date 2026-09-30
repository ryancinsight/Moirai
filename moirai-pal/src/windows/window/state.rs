//! Bounded callback state for a native window.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::ffi::c_void;
use std::io;
use std::mem::ManuallyDrop;
use std::rc::Rc;

use windows::Win32::Foundation::LPARAM;

use super::config::{MAX_WINDOW_EVENTS, allocation_error};
use super::event::{CompositionPhase, ModifierState, WindowEvent};
use super::hotkey::MAX_PENDING_HOTKEY_PRESSES;
use super::input::update_modifier;
use super::menu_bar::MAX_PENDING_MENU_COMMANDS;
use super::tray::{MAX_PENDING_TRAY_EVENTS, TrayEvent, decode_tray_event};

#[derive(Debug)]
pub(super) struct PresentedFrame {
    pub(super) width: u32,
    pub(super) height: u32,
    pub(super) pixels: Vec<u32>,
}

#[derive(Debug)]
pub(super) struct WindowState {
    pub(super) events: VecDeque<WindowEvent>,
    pub(super) hotkey_presses: VecDeque<u16>,
    pub(super) menu_commands: VecDeque<u16>,
    pub(super) tray_events: VecDeque<TrayEvent>,
    pub(super) frame: Option<PresentedFrame>,
    pending_high_surrogate: Option<u16>,
    pub(super) composition_active: bool,
    pub(super) modifiers: ModifierState,
    pub(super) overflowed: bool,
    pub(super) error: Option<io::Error>,
}

impl WindowState {
    pub(super) fn new() -> io::Result<Self> {
        let mut events = VecDeque::new();
        events
            .try_reserve(MAX_WINDOW_EVENTS)
            .map_err(|_| allocation_error())?;
        let mut hotkey_presses = VecDeque::new();
        hotkey_presses
            .try_reserve(MAX_PENDING_HOTKEY_PRESSES)
            .map_err(|_| allocation_error())?;
        let mut menu_commands = VecDeque::new();
        menu_commands
            .try_reserve(MAX_PENDING_MENU_COMMANDS)
            .map_err(|_| allocation_error())?;
        let mut tray_events = VecDeque::new();
        tray_events
            .try_reserve(MAX_PENDING_TRAY_EVENTS)
            .map_err(|_| allocation_error())?;
        Ok(Self {
            events,
            hotkey_presses,
            menu_commands,
            tray_events,
            frame: None,
            pending_high_surrogate: None,
            composition_active: false,
            modifiers: ModifierState::NONE,
            overflowed: false,
            error: None,
        })
    }

    pub(super) fn push(&mut self, event: WindowEvent) {
        if self.events.len() >= MAX_WINDOW_EVENTS {
            self.overflowed = true;
        } else {
            self.events.push_back(event);
        }
    }

    pub(super) fn push_hotkey(&mut self, id: usize) {
        if let Ok(id) = u16::try_from(id)
            && self.hotkey_presses.len() < MAX_PENDING_HOTKEY_PRESSES
        {
            self.hotkey_presses.push_back(id);
        }
    }

    /// Queues a menu-bar command; accelerator and control notifications
    /// carry a non-zero code or a control handle and are ignored.
    pub(super) fn push_menu_command(&mut self, wparam: usize, lparam: isize) {
        let code = (wparam >> 16) & 0xffff;
        if code == 0 && lparam == 0 && self.menu_commands.len() < MAX_PENDING_MENU_COMMANDS {
            self.menu_commands.push_back((wparam & 0xffff) as u16);
        }
    }

    pub(super) fn push_tray(&mut self, wparam: usize, lparam: isize) {
        if let Some(event) = decode_tray_event(wparam, lparam)
            && self.tray_events.len() < MAX_PENDING_TRAY_EVENTS
        {
            self.tray_events.push_back(event);
        }
    }

    pub(super) fn push_composition(&mut self, phase: CompositionPhase, text: String) {
        self.composition_active =
            matches!(phase, CompositionPhase::Started | CompositionPhase::Updated);
        self.push(WindowEvent::TextComposition { phase, text });
    }

    pub(super) fn record_error(&mut self, error: io::Error) {
        if self.error.is_none() {
            self.error = Some(error);
        }
    }

    pub(super) fn update_modifier(&mut self, virtual_key: u32, lparam: LPARAM, pressed: bool) {
        self.modifiers = update_modifier(self.modifiers, virtual_key, lparam, pressed);
    }

    pub(super) fn clear_modifiers(&mut self) {
        self.modifiers = ModifierState::NONE;
    }

    pub(super) fn push_text_unit(&mut self, unit: u16) {
        match (self.pending_high_surrogate.take(), unit) {
            (Some(high), low @ 0xdc00..=0xdfff) => {
                let scalar =
                    0x1_0000 + ((u32::from(high) - 0xd800) << 10) + (u32::from(low) - 0xdc00);
                if let Some(character) = char::from_u32(scalar) {
                    self.push(WindowEvent::TextInput { character });
                } else {
                    self.push(WindowEvent::TextInput {
                        character: '\u{fffd}',
                    });
                }
            }
            (Some(_), _) => {
                self.push(WindowEvent::TextInput {
                    character: '\u{fffd}',
                });
                self.push_text_unit(unit);
            }
            (None, high @ 0xd800..=0xdbff) => {
                self.pending_high_surrogate = Some(high);
            }
            (None, 0xdc00..=0xdfff) => {
                self.push(WindowEvent::TextInput {
                    character: '\u{fffd}',
                });
            }
            (None, unit) => {
                if let Some(character) = char::from_u32(u32::from(unit)) {
                    self.push(WindowEvent::TextInput { character });
                }
            }
        }
    }

    pub(super) fn finish_text(&mut self) {
        if self.pending_high_surrogate.take().is_some() {
            self.push(WindowEvent::TextInput {
                character: '\u{fffd}',
            });
        }
    }
}

/// Callback state shared by a `NativeWindow` and its window procedure.
///
/// Both sides reach the state through this one reference-counted cell, so no
/// `&mut WindowState` is ever derived from a raw pointer and two mutable
/// references cannot coexist: [`with`](Self::with) lends the state for the
/// duration of a closure that performs no native call, so a message that a
/// native call delivers re-entrantly to the window procedure always finds the
/// state unborrowed. The window holds its own strong count from `WM_NCCREATE`
/// to `WM_NCDESTROY`, so the state outlives every message the window can
/// receive, including after a failed `DestroyWindow`.
#[derive(Clone, Debug)]
pub(super) struct SharedWindowState(Rc<RefCell<WindowState>>);

impl SharedWindowState {
    pub(super) fn new(state: WindowState) -> Self {
        Self(Rc::new(RefCell::new(state)))
    }

    /// Lends the state mutably to `operation`, which must not call into the
    /// native window system.
    pub(super) fn with<R>(&self, operation: impl FnOnce(&mut WindowState) -> R) -> R {
        operation(&mut self.0.borrow_mut())
    }

    /// The `lpCreateParams` value that lets the window procedure adopt this
    /// state during `WM_NCCREATE`. Borrows: no count is transferred.
    pub(super) fn create_param(&self) -> *const c_void {
        Rc::as_ptr(&self.0).cast()
    }

    /// Gives the window its own strong count and returns the value to store
    /// in `GWLP_USERDATA`.
    ///
    /// # Safety
    /// `create_param` must come from [`create_param`](Self::create_param) of a
    /// state that is alive for this call, and the caller must store the result
    /// as the window's user data and later release it exactly once through
    /// [`release_userdata`](Self::release_userdata).
    pub(super) unsafe fn adopt_create_param(create_param: *const c_void) -> isize {
        let cell = create_param.cast::<RefCell<WindowState>>();
        // SAFETY: the caller guarantees `cell` is the live `Rc` allocation of
        // a `SharedWindowState`, so a strong count may be added to it.
        unsafe { Rc::increment_strong_count(cell) };
        cell as isize
    }

    /// Views the state a window adopted, without changing its count.
    ///
    /// # Safety
    /// `userdata` must be zero or a value returned by
    /// [`adopt_create_param`](Self::adopt_create_param) that has not been
    /// released, and the result must not be used after its release.
    pub(super) unsafe fn borrow_userdata(userdata: isize) -> Option<ManuallyDrop<Self>> {
        let cell = userdata as *const RefCell<WindowState>;
        if cell.is_null() {
            return None;
        }
        // SAFETY: the adopted count keeps the allocation alive; `ManuallyDrop`
        // stops the reconstructed `Rc` from consuming that count.
        Some(ManuallyDrop::new(Self(unsafe { Rc::from_raw(cell) })))
    }

    /// Releases the count adopted by the window.
    pub(super) fn release_userdata(state: ManuallyDrop<Self>) {
        drop(ManuallyDrop::into_inner(state));
    }

    #[cfg(test)]
    pub(super) fn holders(&self) -> usize {
        Rc::strong_count(&self.0)
    }
}

pub(super) fn decode_composition(units: &[u16]) -> io::Result<String> {
    if units.len() > super::config::MAX_COMPOSITION_UNITS {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "native IME composition exceeds the bounded UTF-16 limit",
        ));
    }
    String::from_utf16(units).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "native IME composition contains invalid UTF-16",
        )
    })
}

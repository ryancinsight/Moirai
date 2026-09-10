//! Native input and lifecycle values.

pub(super) const CONTROL_BITS: u8 = 0b0000_0011;
pub(super) const CONTROL_LEFT: u8 = 0b0000_0001;
pub(super) const CONTROL_RIGHT: u8 = 0b0000_0010;
pub(super) const SHIFT_BITS: u8 = 0b0000_1100;
pub(super) const SHIFT_LEFT: u8 = 0b0000_0100;
pub(super) const SHIFT_RIGHT: u8 = 0b0000_1000;
pub(super) const ALT_BITS: u8 = 0b0011_0000;
pub(super) const ALT_LEFT: u8 = 0b0001_0000;
pub(super) const ALT_RIGHT: u8 = 0b0010_0000;
pub(super) const META_BITS: u8 = 0b1100_0000;
pub(super) const META_LEFT: u8 = 0b0100_0000;
pub(super) const META_RIGHT: u8 = 0b1000_0000;
const WHEEL_FLAG_CONTROL: usize = 0x0008;
const WHEEL_FLAG_SHIFT: usize = 0x0004;

/// Modifier-key state captured with one native input event.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ModifierState {
    bits: u8,
}

impl ModifierState {
    /// No modifier keys are pressed.
    pub const NONE: Self = Self { bits: 0 };

    /// Returns whether Control was held for the event.
    #[must_use]
    pub const fn ctrl(self) -> bool {
        self.bits & CONTROL_BITS != 0
    }

    /// Returns whether Shift was held for the event.
    #[must_use]
    pub const fn shift(self) -> bool {
        self.bits & SHIFT_BITS != 0
    }

    /// Returns whether Alt was held for the event.
    #[must_use]
    pub const fn alt(self) -> bool {
        self.bits & ALT_BITS != 0
    }

    /// Returns whether the Windows key was held for the event.
    #[must_use]
    pub const fn meta(self) -> bool {
        self.bits & META_BITS != 0
    }

    pub(super) const fn set_bits(self, mask: u8, pressed: bool) -> Self {
        let bits = if pressed {
            self.bits | mask
        } else {
            self.bits & !mask
        };
        Self { bits }
    }

    pub(super) const fn with_wheel_message_flags(self, wparam: usize) -> Self {
        let flags = wparam & 0xffff;
        let mut state = self.set_bits(CONTROL_BITS | SHIFT_BITS, false);
        if flags & WHEEL_FLAG_CONTROL != 0 {
            state = state.set_bits(CONTROL_BITS, true);
        }
        if flags & WHEEL_FLAG_SHIFT != 0 {
            state = state.set_bits(SHIFT_BITS, true);
        }
        state
    }
}

/// Mouse button reported by a native pointer message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MouseButton {
    /// Left mouse button.
    Left,
    /// Right mouse button.
    Right,
    /// Middle mouse button.
    Middle,
    /// First extended mouse button.
    X1,
    /// Second extended mouse button.
    X2,
}

/// Phase of a native text composition transaction.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompositionPhase {
    /// An IME started composing text.
    Started,
    /// An IME changed the uncommitted preedit text.
    Updated,
    /// An IME committed text into the control.
    Committed,
    /// An IME canceled its uncommitted preedit text.
    Canceled,
}

/// Value event translated from one native window message.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WindowEvent {
    /// The user requested that the window close.
    CloseRequested,
    /// The HWND completed destruction.
    Destroyed,
    /// The window received keyboard focus.
    FocusGained,
    /// The window lost keyboard focus.
    FocusLost,
    /// The pointer moved in client coordinates.
    PointerMove {
        /// Horizontal client coordinate.
        x: i32,
        /// Vertical client coordinate.
        y: i32,
    },
    /// A mouse button was pressed in client coordinates.
    PointerDown {
        /// Horizontal client coordinate.
        x: i32,
        /// Vertical client coordinate.
        y: i32,
        /// Pressed button.
        button: MouseButton,
    },
    /// A mouse button was released in client coordinates.
    PointerUp {
        /// Horizontal client coordinate.
        x: i32,
        /// Vertical client coordinate.
        y: i32,
        /// Released button.
        button: MouseButton,
    },
    /// A wheel rotated at client coordinates.
    ///
    /// Win32 reports wheel movement in signed multiples of `WHEEL_DELTA`;
    /// horizontal and vertical messages populate only their corresponding
    /// component. The modifier snapshot combines the wheel message flags with
    /// modifier transitions observed by this window.
    PointerWheel {
        /// Horizontal client coordinate.
        x: i32,
        /// Vertical client coordinate.
        y: i32,
        /// Signed horizontal wheel delta in Win32 units.
        delta_x: i16,
        /// Signed vertical wheel delta in Win32 units.
        delta_y: i16,
        /// Modifier keys held when the wheel message was received.
        modifiers: ModifierState,
    },
    /// A virtual key was pressed.
    KeyDown {
        /// Windows virtual-key value.
        virtual_key: u32,
        /// The message is an auto-repeat.
        repeated: bool,
    },
    /// A virtual key was released.
    KeyUp {
        /// Windows virtual-key value.
        virtual_key: u32,
    },
    /// Unicode text produced by the native message queue.
    TextInput {
        /// One scalar or replacement character; never an unmatched surrogate.
        character: char,
    },
    /// A bounded native IME composition update.
    TextComposition {
        /// Composition lifecycle phase.
        phase: CompositionPhase,
        /// Preedit or committed UTF-8 text; start and cancel carry an empty value.
        text: String,
    },
    /// The client size changed, including a minimized zero extent.
    Resized {
        /// Horizontal client extent.
        width: u32,
        /// Vertical client extent.
        height: u32,
    },
    /// The window crossed a display scale boundary.
    DpiChanged {
        /// Effective horizontal DPI reported by Windows.
        dpi: u32,
    },
}

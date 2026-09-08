//! Native input and lifecycle values.

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

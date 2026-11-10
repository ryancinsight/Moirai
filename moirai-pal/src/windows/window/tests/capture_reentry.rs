//! Capture ownership between windows of one thread, and re-entry from the
//! window that loses capture.
//!
//! `SetCapture` runs the losing window's procedure before it returns, and that
//! procedure can send messages back to the window being captured. These tests
//! build that situation with real windows on the test thread.

use super::super::event::{MouseButton, WindowEvent};
use super::super::native::NativeWindow;
use super::super::state::WindowState;
use super::pointer_capture::{button_wparam, capture_owner, hidden_window, pointer_lparam, send};
use std::cell::{Cell, RefCell};
use std::rc::Weak;
use std::time::Duration;
use windows::Win32::Foundation::{
    ERROR_CLASS_ALREADY_EXISTS, HINSTANCE, HWND, LPARAM, LRESULT, WPARAM,
};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::SystemServices::{MK_LBUTTON, MK_RBUTTON};
use windows::Win32::UI::Input::KeyboardAndMouse::{ReleaseCapture, SetCapture};
use windows::Win32::UI::Shell::{DefSubclassProc, RemoveWindowSubclass, SetWindowSubclass};
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DestroyWindow, RegisterClassW, SendMessageW, WINDOW_EX_STYLE,
    WM_CAPTURECHANGED, WM_LBUTTONDOWN, WM_LBUTTONUP, WM_MOUSEMOVE, WM_RBUTTONDOWN, WM_RBUTTONUP,
    WNDCLASSW, WS_OVERLAPPED,
};
use windows::core::w;

thread_local! {
    /// Window the peer procedure sends its pointer move to.
    static OBSERVED: Cell<HWND> = const { Cell::new(HWND(std::ptr::null_mut())) };
    /// Set by the test; the peer reacts to exactly one loss of capture.
    static RECAPTURE_ARMED: Cell<bool> = const { Cell::new(false) };
    /// Set by the test; the peer drops [`OWNER`] on exactly one loss of capture.
    static DESTROY_ARMED: Cell<bool> = const { Cell::new(false) };
    /// Owning handle of the window under test, moved here so the peer can drop it.
    static OWNER: RefCell<Option<NativeWindow>> = const { RefCell::new(None) };
    /// Non-owning handle to the state of that window.
    static STATE_PROBE: RefCell<Weak<RefCell<WindowState>>> = const { RefCell::new(Weak::new()) };
    /// `WM_CAPTURECHANGED` messages the subclassed window received.
    static CAPTURE_CHANGES: Cell<u32> = const { Cell::new(0) };
    /// Counts on the state the peer saw right after dropping the owner.
    static COUNTS_AFTER_OWNER_DROP: Cell<Option<usize>> = const { Cell::new(None) };
}

/// Peer window procedure: on its first loss of capture it sends a pointer move
/// to [`OBSERVED`] and takes the capture back.
unsafe extern "system" fn peer_proc(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
) -> LRESULT {
    if message == WM_CAPTURECHANGED && DESTROY_ARMED.replace(false) {
        // Dropping the owner destroys the window under test, whose
        // `WM_NCDESTROY` returns the window's count, and then releases the
        // owner's count. Only a count held by the interrupted button handler
        // can remain.
        drop(OWNER.take());
        COUNTS_AFTER_OWNER_DROP.set(Some(STATE_PROBE.with_borrow(Weak::strong_count)));
        return LRESULT(0);
    }
    if message == WM_CAPTURECHANGED && RECAPTURE_ARMED.replace(false) {
        // SAFETY: both calls run on the thread that owns the two live windows.
        // The send re-enters the observed window's procedure, which the test
        // requires to hold no borrow of its state while `SetCapture` runs.
        unsafe {
            SendMessageW(
                OBSERVED.get(),
                WM_MOUSEMOVE,
                Some(WPARAM(0)),
                Some(pointer_lparam(7, 9)),
            );
            SetCapture(hwnd);
        }
        return LRESULT(0);
    }
    // SAFETY: forwards the message this procedure received, unchanged.
    unsafe { DefWindowProcW(hwnd, message, wparam, lparam) }
}

/// Hidden window of the test thread that runs [`peer_proc`].
struct PeerWindow {
    hwnd: HWND,
}

impl PeerWindow {
    fn new() -> Self {
        // SAFETY: a null module name returns the handle of this process image.
        let instance = HINSTANCE::from(
            unsafe { GetModuleHandleW(None) }.expect("module handle of the test executable"),
        );
        let class = WNDCLASSW {
            lpfnWndProc: Some(peer_proc),
            hInstance: instance,
            lpszClassName: w!("MoiraiCapturePeer"),
            ..Default::default()
        };
        // SAFETY: `class` holds a static class name and a procedure with the
        // system callback ABI, valid for this synchronous call.
        let atom = unsafe { RegisterClassW(&class) };
        assert!(
            atom != 0
                || std::io::Error::last_os_error().raw_os_error()
                    == Some(i32::try_from(ERROR_CLASS_ALREADY_EXISTS.0).expect("a Win32 code")),
            "peer class registers once per process"
        );
        // SAFETY: the class and title are static; the window is created hidden
        // on this thread and destroyed by `Drop`.
        let hwnd = unsafe {
            CreateWindowExW(
                WINDOW_EX_STYLE::default(),
                w!("MoiraiCapturePeer"),
                w!("Moirai capture peer"),
                WS_OVERLAPPED,
                0,
                0,
                16,
                16,
                None,
                None,
                Some(instance),
                None,
            )
        }
        .expect("peer window");
        Self { hwnd }
    }
}

impl Drop for PeerWindow {
    fn drop(&mut self) {
        // SAFETY: the window was created on this thread and is destroyed once.
        let destroyed = unsafe { DestroyWindow(self.hwnd) };
        assert!(
            destroyed.is_ok() || std::thread::panicking(),
            "peer window destroys"
        );
    }
}

#[test]
fn a_release_on_one_window_leaves_the_capture_of_another() {
    let mut first = hidden_window("Moirai capture owner");
    let mut second = hidden_window("Moirai capture bystander");

    send(
        &first,
        WM_LBUTTONDOWN,
        button_wparam(MK_LBUTTON.0, 0),
        10,
        20,
    );
    assert_eq!(capture_owner(), first.hwnd, "the press captures");

    send(&second, WM_LBUTTONUP, button_wparam(0, 0), 3, 4);
    assert_eq!(
        capture_owner(),
        first.hwnd,
        "a window that does not hold the capture must not release it"
    );

    send(&first, WM_LBUTTONUP, button_wparam(0, 0), 10, 20);
    assert!(capture_owner().is_invalid(), "the owner's release ends it");

    assert_eq!(
        first.poll_events().expect("owner events"),
        vec![
            WindowEvent::PointerDown {
                x: 10,
                y: 20,
                button: MouseButton::Left
            },
            WindowEvent::PointerUp {
                x: 10,
                y: 20,
                button: MouseButton::Left
            },
        ]
    );
    assert_eq!(
        second.poll_events().expect("bystander events"),
        vec![WindowEvent::PointerUp {
            x: 3,
            y: 4,
            button: MouseButton::Left
        }]
    );
    first.close().expect("destroy");
    second.close().expect("destroy");
}

/// Presses a button on `window` while `peer` holds the capture and takes it
/// back from inside the loss notification, so the capture call fails.
fn press_while_a_peer_retakes_the_capture(window: &NativeWindow, peer: &PeerWindow) {
    OBSERVED.set(window.hwnd);
    // SAFETY: the peer is a live window of this thread.
    unsafe { SetCapture(peer.hwnd) };
    assert_eq!(capture_owner(), peer.hwnd, "the peer holds the capture");

    RECAPTURE_ARMED.set(true);
    send(
        window,
        WM_LBUTTONDOWN,
        button_wparam(MK_LBUTTON.0, 0),
        10,
        20,
    );
    assert!(
        !RECAPTURE_ARMED.get(),
        "taking the capture reached the peer procedure"
    );
    assert_eq!(
        capture_owner(),
        peer.hwnd,
        "the peer took the capture back, so this window does not own it"
    );
}

#[test]
fn wait_events_reports_a_failed_capture_like_poll_events() {
    let mut window = hidden_window("Moirai capture wait");
    let peer = PeerWindow::new();
    press_while_a_peer_retakes_the_capture(&window, &peer);

    let error = window
        .wait_events(Duration::ZERO)
        .expect_err("the lost capture is reported");
    assert_eq!(error.kind(), std::io::ErrorKind::Other);
    assert_eq!(
        error.to_string(),
        "native window could not capture the mouse"
    );
    assert_eq!(
        window
            .wait_events(Duration::ZERO)
            .expect("queued events survive the error"),
        vec![
            WindowEvent::PointerDown {
                x: 10,
                y: 20,
                button: MouseButton::Left
            },
            WindowEvent::PointerMove { x: 7, y: 9 },
        ]
    );

    // SAFETY: the calling thread owns the capture being released.
    unsafe { ReleaseCapture() }.expect("release the peer's capture");
    window.close().expect("destroy");
}

#[test]
fn a_peer_reentering_during_capture_loss_fails_the_capture_without_losing_events() {
    let mut window = hidden_window("Moirai capture reentry");
    let peer = PeerWindow::new();
    press_while_a_peer_retakes_the_capture(&window, &peer);

    let error = window
        .poll_events()
        .expect_err("the lost capture is reported");
    assert_eq!(error.kind(), std::io::ErrorKind::Other);
    assert_eq!(
        error.to_string(),
        "native window could not capture the mouse"
    );
    assert_eq!(
        window
            .poll_events()
            .expect("queued events survive the error"),
        vec![
            WindowEvent::PointerDown {
                x: 10,
                y: 20,
                button: MouseButton::Left
            },
            WindowEvent::PointerMove { x: 7, y: 9 },
        ],
        "the pointer move sent during the capture change follows the press that caused it"
    );

    // SAFETY: the calling thread owns the capture being released.
    unsafe { ReleaseCapture() }.expect("release the peer's capture");
    window.close().expect("destroy");
}

#[test]
fn a_peer_destroying_the_window_during_capture_loss_leaves_the_state_alive_for_the_error() {
    let window = hidden_window("Moirai capture destroyed");
    let hwnd = window.hwnd;
    let peer = PeerWindow::new();
    // SAFETY: the peer is a live window of this thread.
    unsafe { SetCapture(peer.hwnd) };
    assert_eq!(capture_owner(), peer.hwnd, "the peer holds the capture");

    STATE_PROBE.replace(window.state.downgrade());
    OWNER.set(Some(window));
    DESTROY_ARMED.set(true);
    // SAFETY: the target is the live window moved into `OWNER`; the peer
    // procedure drops that owner on this thread while the message is being
    // handled, which is the situation under test.
    unsafe {
        SendMessageW(
            hwnd,
            WM_LBUTTONDOWN,
            Some(button_wparam(MK_LBUTTON.0, 0)),
            Some(pointer_lparam(10, 20)),
        )
    };

    assert!(
        !DESTROY_ARMED.get(),
        "taking the capture reached the peer procedure"
    );
    assert!(
        OWNER.with_borrow(Option::is_none),
        "the peer dropped the owner"
    );
    assert_eq!(
        COUNTS_AFTER_OWNER_DROP.get(),
        Some(1),
        "after the window and its owner released their counts, the button handler's own count keeps the state alive"
    );
    assert_eq!(
        STATE_PROBE.with_borrow(Weak::strong_count),
        0,
        "the handler releases that count when it returns"
    );
}

/// Subclass procedure that counts the capture-loss notifications a window gets.
unsafe extern "system" fn count_capture_changes(
    hwnd: HWND,
    message: u32,
    wparam: WPARAM,
    lparam: LPARAM,
    _subclass_id: usize,
    _reference: usize,
) -> LRESULT {
    if message == WM_CAPTURECHANGED {
        CAPTURE_CHANGES.set(CAPTURE_CHANGES.get() + 1);
    }
    // SAFETY: forwards the message this procedure received, unchanged.
    unsafe { DefSubclassProc(hwnd, message, wparam, lparam) }
}

#[test]
fn a_second_press_does_not_notify_the_window_that_already_owns_the_capture() {
    const SUBCLASS_ID: usize = 1;
    let mut window = hidden_window("Moirai capture owner count");
    // SAFETY: the window is live on this thread and the procedure has the
    // subclass ABI; it is removed before the window is destroyed.
    assert!(
        unsafe { SetWindowSubclass(window.hwnd, Some(count_capture_changes), SUBCLASS_ID, 0) }
            .as_bool()
    );
    CAPTURE_CHANGES.set(0);

    send(
        &window,
        WM_LBUTTONDOWN,
        button_wparam(MK_LBUTTON.0, 0),
        10,
        20,
    );
    assert_eq!(capture_owner(), window.hwnd, "the first press captures");
    send(
        &window,
        WM_RBUTTONDOWN,
        button_wparam(MK_LBUTTON.0 | MK_RBUTTON.0, 0),
        10,
        20,
    );
    assert_eq!(capture_owner(), window.hwnd, "the capture persists");
    assert_eq!(
        CAPTURE_CHANGES.get(),
        0,
        "re-taking a capture the window owns must not notify the window"
    );

    send(
        &window,
        WM_RBUTTONUP,
        button_wparam(MK_LBUTTON.0, 0),
        10,
        20,
    );
    send(&window, WM_LBUTTONUP, button_wparam(0, 0), 10, 20);
    assert!(capture_owner().is_invalid(), "the last release ends it");
    assert_eq!(
        CAPTURE_CHANGES.get(),
        1,
        "ending the capture notifies the window once"
    );

    // SAFETY: removes the subclass installed above from the same live window.
    assert!(
        unsafe { RemoveWindowSubclass(window.hwnd, Some(count_capture_changes), SUBCLASS_ID) }
            .as_bool()
    );
    window.close().expect("destroy");
}

//! Restored-rectangle and maximized-state round trips through a live HWND.

use super::super::native::NativeWindow;
use super::super::{
    MAX_FRAME_DIMENSION, MAX_PLACEMENT_COORDINATE, WindowConfig, WindowPlacement, WindowVisibility,
};
use windows::Win32::UI::WindowsAndMessaging::IsWindowVisible;

#[test]
fn placement_values_are_bounded() {
    assert!(WindowPlacement::new(-40, 30, 800, 600, true).is_ok());
    assert!(WindowPlacement::new(MAX_PLACEMENT_COORDINATE + 1, 0, 800, 600, false).is_err());
    assert!(WindowPlacement::new(0, -MAX_PLACEMENT_COORDINATE - 1, 800, 600, false).is_err());
    assert!(WindowPlacement::new(0, 0, 0, 600, false).is_err());
    assert!(WindowPlacement::new(0, 0, MAX_FRAME_DIMENSION + 1, 600, false).is_err());
}

#[test]
fn hidden_window_placement_round_trips_without_showing() {
    let config =
        WindowConfig::with_visibility("Moirai placement", 320, 240, WindowVisibility::Hidden)
            .expect("config");
    let mut window = NativeWindow::new(&config).expect("native window");

    let restored = WindowPlacement::new(96, 128, 480, 360, false).expect("placement");
    window.set_placement(restored).expect("set placement");
    assert_eq!(window.placement().expect("placement"), restored);

    let maximized = WindowPlacement::new(64, 72, 500, 400, true).expect("placement");
    window
        .set_placement(maximized)
        .expect("set maximized placement");
    assert_eq!(window.placement().expect("placement"), maximized);
    // SAFETY: the handle is the live HWND owned by this test's thread.
    assert!(!unsafe { IsWindowVisible(window.hwnd) }.as_bool());

    window.close().expect("close");
    assert!(window.placement().is_err());
    assert!(window.set_placement(restored).is_err());
}

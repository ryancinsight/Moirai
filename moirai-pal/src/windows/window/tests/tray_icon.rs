//! Tray icon values, callback decoding and the live shell round trip.

use super::super::native::NativeWindow;
use super::super::tray::{TRAY_CALLBACK_MESSAGE, decode_tray_event};
use super::super::{
    MAX_NOTIFICATION_BODY_UNITS, MAX_TRAY_TOOLTIP_UNITS, TrayEvent, TrayIconImage, WindowConfig,
    WindowVisibility,
};
use windows::Win32::Foundation::{LPARAM, WPARAM};
use windows::Win32::UI::Shell::{NIN_BALLOONUSERCLICK, NIN_SELECT, NINF_KEY};
use windows::Win32::UI::WindowsAndMessaging::{PostMessageW, WM_CONTEXTMENU, WM_MOUSEMOVE};

fn anchor(x: i16, y: i16) -> usize {
    usize::from(x as u16) | (usize::from(y as u16) << 16)
}

#[test]
fn tray_images_are_square_and_sized() {
    assert!(TrayIconImage::new(16, &[0xff00_78d4; 256]).is_ok());
    assert!(TrayIconImage::new(32, &[0; 1024]).is_ok());
    assert!(TrayIconImage::new(24, &[0; 576]).is_err());
    assert!(TrayIconImage::new(16, &[0; 255]).is_err());
}

#[test]
fn version_four_callbacks_decode() {
    assert_eq!(
        decode_tray_event(0, NIN_SELECT as isize),
        Some(TrayEvent::Activated)
    );
    assert_eq!(
        decode_tray_event(0, (NIN_SELECT | NINF_KEY) as isize),
        Some(TrayEvent::Activated)
    );
    assert_eq!(
        decode_tray_event(anchor(-40, 900), WM_CONTEXTMENU as isize),
        Some(TrayEvent::ContextRequested { x: -40, y: 900 })
    );
    assert_eq!(
        decode_tray_event(0, NIN_BALLOONUSERCLICK as isize),
        Some(TrayEvent::NotificationClicked)
    );
    assert_eq!(decode_tray_event(0, WM_MOUSEMOVE as isize), None);
    // The icon identifier travels in the high word and is ignored.
    assert_eq!(
        decode_tray_event(0, ((1 << 16) | NIN_SELECT) as isize),
        Some(TrayEvent::Activated)
    );
}

#[test]
fn tray_icon_shows_notifies_and_is_removed() {
    let config = WindowConfig::with_visibility("Moirai tray", 320, 240, WindowVisibility::Hidden)
        .expect("config");
    let mut window = NativeWindow::new(&config).expect("native window");
    assert!(
        window.show_notification("Title", "Body").is_err(),
        "no icon yet"
    );
    let image = TrayIconImage::new(16, &[0xff00_78d4; 256]).expect("image");
    let long_tip = "t".repeat(MAX_TRAY_TOOLTIP_UNITS + 1);
    assert!(window.show_tray_icon(&image, &long_tip).is_err());

    window
        .show_tray_icon(&image, "Moirai tray test")
        .expect("add icon");
    window
        .show_tray_icon(&image, "Moirai tray test, updated")
        .expect("modify icon");
    assert!(window.show_notification("Title", "").is_err());
    let long_body = "b".repeat(MAX_NOTIFICATION_BODY_UNITS + 1);
    assert!(window.show_notification("Title", &long_body).is_err());
    window
        .show_notification("Moirai", "Tray notification test")
        .expect("notify");

    // SAFETY: the message targets the live HWND owned by this test and
    // carries only scalar parameters.
    unsafe {
        PostMessageW(
            Some(window.hwnd),
            TRAY_CALLBACK_MESSAGE,
            WPARAM(anchor(12, 34)),
            LPARAM(WM_CONTEXTMENU as isize),
        )
    }
    .expect("post tray callback");
    window.poll_events().expect("pump");
    assert_eq!(
        window.take_tray_events(),
        vec![TrayEvent::ContextRequested { x: 12, y: 34 }]
    );

    assert!(window.remove_tray_icon().expect("remove"));
    assert!(!window.remove_tray_icon().expect("second remove"));
    window
        .show_tray_icon(&image, "Moirai tray test")
        .expect("add again");
    window.close().expect("close removes the icon");
    assert!(window.show_tray_icon(&image, "closed").is_err());
}

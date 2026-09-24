//! Popup menu validation and a dismissed live menu.

use super::super::native::NativeWindow;
use super::super::{
    MAX_POPUP_MENU_ITEMS, MAX_POPUP_MENU_LABEL_UNITS, PopupMenu, PopupMenuItem, WindowConfig,
    WindowVisibility,
};
use windows::Win32::Foundation::HWND;
use windows::Win32::UI::WindowsAndMessaging::{EndMenu, KillTimer, SetTimer};

#[test]
fn menus_are_validated() {
    assert!(PopupMenuItem::action("Open", true).is_ok());
    assert!(PopupMenuItem::action("", true).is_err());
    assert!(PopupMenuItem::action("a\0b", true).is_err());
    let long = "x".repeat(MAX_POPUP_MENU_LABEL_UNITS + 1);
    assert!(PopupMenuItem::action(&long, true).is_err());
    assert!(PopupMenu::new(vec![PopupMenuItem::separator()]).is_err());
    assert!(PopupMenu::new(Vec::new()).is_err());
    let open = PopupMenuItem::action("Open", true).expect("item");
    assert!(PopupMenu::new(vec![open.clone(); MAX_POPUP_MENU_ITEMS + 1]).is_err());
    let menu = PopupMenu::new(vec![open, PopupMenuItem::separator()]).expect("menu");
    assert_eq!(menu.len(), 2);
}

unsafe extern "system" fn dismiss(_: HWND, _: u32, _: usize, _: u32) {
    // SAFETY: EndMenu only asks the active menu loop on this thread to end.
    let _ = unsafe { EndMenu() };
}

#[test]
fn a_dismissed_menu_returns_no_choice() {
    let config = WindowConfig::with_visibility("Moirai menu", 320, 240, WindowVisibility::Hidden)
        .expect("config");
    let mut window = NativeWindow::new(&config).expect("native window");
    let menu = PopupMenu::new(vec![
        PopupMenuItem::action("Show", true).expect("item"),
        PopupMenuItem::separator(),
        PopupMenuItem::action("Quit", false).expect("item"),
    ])
    .expect("menu");
    // SAFETY: the timer targets this test's live window and its callback
    // only ends the menu loop the next call starts.
    let timer = unsafe { SetTimer(Some(window.hwnd), 7, 50, Some(dismiss)) };
    assert_ne!(timer, 0, "dismissal timer");
    assert_eq!(
        window.show_popup_menu(&menu, 40, 40).expect("menu loop"),
        None
    );
    // SAFETY: the timer was created above for this live window.
    unsafe { KillTimer(Some(window.hwnd), 7) }.expect("kill timer");
    window.close().expect("close");
    assert!(window.show_popup_menu(&menu, 0, 0).is_err());
}

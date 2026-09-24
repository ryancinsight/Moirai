//! Menu bar validation, command decoding and live attachment.

use super::super::menu_bar::decode_command;
use super::super::native::NativeWindow;
use super::super::{
    MAX_MENU_BAR_MENUS, MenuBar, MenuCommand, PopupMenu, PopupMenuItem, WindowConfig,
    WindowVisibility,
};
use windows::Win32::Foundation::{LPARAM, WPARAM};
use windows::Win32::UI::WindowsAndMessaging::{GetMenu, PostMessageW, WM_COMMAND};

fn menu(labels: &[&str]) -> PopupMenu {
    PopupMenu::new(
        labels
            .iter()
            .map(|label| {
                if label.is_empty() {
                    PopupMenuItem::separator()
                } else {
                    PopupMenuItem::action(label, true).expect("item")
                }
            })
            .collect(),
    )
    .expect("menu")
}

#[test]
fn menu_bars_are_validated() {
    assert!(MenuBar::new(Vec::<(&str, PopupMenu)>::new()).is_err());
    assert!(MenuBar::new(vec![("", menu(&["Open"]))]).is_err());
    let many: Vec<_> = (0..=MAX_MENU_BAR_MENUS)
        .map(|_| ("&File", menu(&["Open"])))
        .collect();
    assert!(MenuBar::new(many).is_err());
    assert!(MenuBar::new(vec![("&File", menu(&["Open", "", "Exit"]))]).is_ok());
}

#[test]
fn command_identifiers_decode_against_the_bar_shape() {
    let shape = [3, 1];
    assert_eq!(
        decode_command(256, &shape),
        Some(MenuCommand { menu: 0, item: 0 })
    );
    assert_eq!(
        decode_command(258, &shape),
        Some(MenuCommand { menu: 0, item: 2 })
    );
    assert_eq!(
        decode_command(512, &shape),
        Some(MenuCommand { menu: 1, item: 0 })
    );
    for stale in [0, 1, 255, 259, 513, 768] {
        assert_eq!(decode_command(stale, &shape), None, "{stale}");
    }
}

#[test]
fn attached_bars_report_commands_until_replaced() {
    let config =
        WindowConfig::with_visibility("Moirai menu bar", 320, 240, WindowVisibility::Hidden)
            .expect("config");
    let mut window = NativeWindow::new(&config).expect("native window");
    let bar = MenuBar::new(vec![
        ("&File", menu(&["&Open", "", "E&xit"])),
        ("&Help", menu(&["&About"])),
    ])
    .expect("bar");
    window.set_menu_bar(Some(&bar)).expect("attach");
    // SAFETY: reads the live window's menu handle.
    assert!(!unsafe { GetMenu(window.hwnd) }.is_invalid());

    let post = |window: &NativeWindow, wparam: usize, lparam: isize| {
        // SAFETY: the message targets the live HWND owned by this test and
        // carries only scalar parameters.
        unsafe {
            PostMessageW(
                Some(window.hwnd),
                WM_COMMAND,
                WPARAM(wparam),
                LPARAM(lparam),
            )
        }
        .expect("post command");
    };
    post(&window, 258, 0);
    post(&window, 512, 0);
    post(&window, (1 << 16) | 256, 0); // an accelerator, not a menu
    post(&window, 257, 0); // the separator's slot never fires, but decodes in range
    window.poll_events().expect("pump");
    assert_eq!(
        window.take_menu_commands(),
        vec![
            MenuCommand { menu: 0, item: 2 },
            MenuCommand { menu: 1, item: 0 },
            MenuCommand { menu: 0, item: 1 },
        ]
    );

    let smaller = MenuBar::new(vec![("&File", menu(&["&Open"]))]).expect("bar");
    window.set_menu_bar(Some(&smaller)).expect("replace");
    post(&window, 512, 0);
    window.poll_events().expect("pump");
    assert!(
        window.take_menu_commands().is_empty(),
        "stale menus are dropped"
    );

    window.set_menu_bar(None).expect("remove");
    // SAFETY: reads the live window's menu handle.
    assert!(unsafe { GetMenu(window.hwnd) }.is_invalid());
    window.close().expect("close");
    assert!(window.set_menu_bar(Some(&bar)).is_err());
}

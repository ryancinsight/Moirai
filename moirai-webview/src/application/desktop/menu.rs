//! AppKit owns application and window commands, including keyboard dispatch.
//!
//! A decorated window alone does not install a main menu: without a Quit item,
//! Command-Q reaches the WebView and never asks NSApplication to terminate.
//! Application commands target the retained application singleton. Editing and
//! window commands use AppKit's responder chain so they apply to the focused
//! view/key window, including while the page has keyboard focus. No JavaScript
//! callback receives these commands or authority over another native window.

use objc2::{MainThreadMarker, MainThreadOnly, rc::Retained, runtime::Sel, sel};
use objc2_app_kit::{NSApplication, NSEventModifierFlags, NSMenu, NSMenuItem};
use objc2_foundation::NSString;

pub(super) fn install(app: &NSApplication, title: &str, mtm: MainThreadMarker) {
    let main = menu("", mtm);
    let application = menu(title, mtm);
    let hide = item(&format!("Hide {title}"), sel!(hide:), "h", mtm);
    let hide_others = item("Hide Others", sel!(hideOtherApplications:), "h", mtm);
    hide_others
        .setKeyEquivalentModifierMask(NSEventModifierFlags::Command | NSEventModifierFlags::Option);
    let show_all = item("Show All", sel!(unhideAllApplications:), "", mtm);
    let quit = item(&format!("Quit {title}"), sel!(terminate:), "q", mtm);
    for command in [&hide, &hide_others, &show_all, &quit] {
        // SAFETY: Every selector above is an NSApplication sender-taking
        // action; the singleton outlives its weak menu-item target.
        unsafe { command.setTarget(Some(app)) };
    }
    for command in [&hide, &hide_others, &show_all] {
        application.addItem(command);
    }
    application.addItem(&NSMenuItem::separatorItem(mtm));
    application.addItem(&quit);
    submenu(&main, title, &application, mtm);

    let edit = menu("Edit", mtm);
    edit.addItem(&item("Undo", sel!(undo:), "z", mtm));
    let redo = item("Redo", sel!(redo:), "z", mtm);
    redo.setKeyEquivalentModifierMask(NSEventModifierFlags::Command | NSEventModifierFlags::Shift);
    edit.addItem(&redo);
    edit.addItem(&NSMenuItem::separatorItem(mtm));
    for (label, selector, key) in [
        ("Cut", sel!(cut:), "x"),
        ("Copy", sel!(copy:), "c"),
        ("Paste", sel!(paste:), "v"),
        ("Select All", sel!(selectAll:), "a"),
    ] {
        edit.addItem(&item(label, selector, key, mtm));
    }
    submenu(&main, "Edit", &edit, mtm);

    let windows = menu("Window", mtm);
    for (label, selector, key) in [
        ("Close", sel!(performClose:), "w"),
        ("Minimize", sel!(performMiniaturize:), "m"),
        ("Zoom", sel!(performZoom:), ""),
    ] {
        windows.addItem(&item(label, selector, key, mtm));
    }
    let fullscreen = item("Toggle Full Screen", sel!(toggleFullScreen:), "f", mtm);
    fullscreen.setKeyEquivalentModifierMask(
        NSEventModifierFlags::Command | NSEventModifierFlags::Control,
    );
    windows.addItem(&fullscreen);
    submenu(&main, "Window", &windows, mtm);
    app.setMainMenu(Some(&main));
    app.setWindowsMenu(Some(&windows));
}

fn menu(title: &str, mtm: MainThreadMarker) -> Retained<NSMenu> {
    NSMenu::initWithTitle(NSMenu::alloc(mtm), &NSString::from_str(title))
}

fn submenu(main: &NSMenu, title: &str, contents: &NSMenu, mtm: MainThreadMarker) {
    let parent = NSMenuItem::new(mtm);
    parent.setTitle(&NSString::from_str(title));
    parent.setSubmenu(Some(contents));
    main.addItem(&parent);
}

fn item(title: &str, selector: Sel, key: &str, mtm: MainThreadMarker) -> Retained<NSMenuItem> {
    // SAFETY: Callers supply AppKit sender-taking action selectors. The
    // responder chain validates action availability for its current target;
    // NSApplication commands receive their explicit singleton target above.
    let item = unsafe {
        NSMenuItem::initWithTitle_action_keyEquivalent(
            NSMenuItem::alloc(mtm),
            &NSString::from_str(title),
            Some(selector),
            &NSString::from_str(key),
        )
    };
    item.setKeyEquivalentModifierMask(NSEventModifierFlags::Command);
    item
}

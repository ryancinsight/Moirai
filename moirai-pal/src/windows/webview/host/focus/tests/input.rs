//! Separate physical select-all and exact UTF-16 packet batches.

use windows::Win32::UI::{
    Input::KeyboardAndMouse::{
        GetKeyboardLayout, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBD_EVENT_FLAGS, KEYBDINPUT,
        KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP, KEYEVENTF_SCANCODE, KEYEVENTF_UNICODE,
        MAPVK_VK_TO_VSC_EX, MapVirtualKeyExW, SendInput,
    },
    WindowsAndMessaging::{GUITHREADINFO, GetGUIThreadInfo, GetWindowThreadProcessId},
};

pub(super) fn select_all(handle: usize) {
    let focus = recipient(handle);
    let thread = unsafe { GetWindowThreadProcessId(focus, None) };
    assert_ne!(
        thread,
        0,
        "actual recipient thread lookup failed: {}",
        std::io::Error::last_os_error()
    );
    let layout = unsafe { GetKeyboardLayout(thread) };
    assert!(!layout.is_invalid(), "actual recipient HKL unavailable");
    let mut records = Vec::new();
    for (key, released) in [(0x11, false), (0x41, false), (0x41, true), (0x11, true)] {
        let mapped = unsafe { MapVirtualKeyExW(key, MAPVK_VK_TO_VSC_EX, Some(layout)) };
        assert_ne!(mapped & 0xff, 0);
        assert!(matches!(mapped >> 8, 0 | 0xe0));
        records.push(record(
            u16::try_from(mapped & 0xff).expect("scan byte"),
            KEYEVENTF_SCANCODE
                | if mapped >> 8 == 0xe0 {
                    KEYEVENTF_EXTENDEDKEY
                } else {
                    KEYBD_EVENT_FLAGS::default()
                }
                | if released {
                    KEYEVENTF_KEYUP
                } else {
                    KEYBD_EVENT_FLAGS::default()
                },
        ));
    }
    deliver(&records, &[(0, 3), (1, 2)]);
}

fn recipient(handle: usize) -> windows::Win32::Foundation::HWND {
    let mut info = GUITHREADINFO {
        cbSize: size_of::<GUITHREADINFO>() as u32,
        ..Default::default()
    };
    unsafe { GetGUIThreadInfo(0, &mut info) }.expect("actual foreground GUI queue");
    assert!(!info.hwndFocus.is_invalid(), "actual foreground recipient");
    let host = windows::Win32::Foundation::HWND(handle as *mut _);
    assert_eq!(
        unsafe { windows::Win32::UI::WindowsAndMessaging::GetForegroundWindow() },
        host
    );
    assert!(
        info.hwndFocus == host
            || unsafe { windows::Win32::UI::WindowsAndMessaging::IsChild(host, info.hwndFocus) }
                .as_bool(),
        "actual recipient belongs to the owned host"
    );
    info.hwndFocus
}

pub(super) fn insert_unicode(handle: usize, value: &str) {
    let _ = recipient(handle);
    let mut records = Vec::new();
    for unit in value.encode_utf16() {
        records.push(record(unit, KEYEVENTF_UNICODE));
        records.push(record(unit, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP));
    }
    let pairs: Vec<_> = (0..records.len()).step_by(2).map(|i| (i, i + 1)).collect();
    deliver(&records, &pairs);
}

fn deliver(records: &[INPUT], pairs: &[(usize, usize)]) {
    let sent = unsafe { SendInput(records, size_of::<INPUT>() as i32) } as usize;
    if sent != records.len() {
        let primary = std::io::Error::last_os_error();
        for &(down, up) in pairs {
            if sent > down && sent <= up {
                let released = unsafe { SendInput(&records[up..=up], size_of::<INPUT>() as i32) };
                if released != 1 {
                    let error = std::io::Error::last_os_error();
                    tracing::error!(down, up, %error, "native unmatched-key release failed");
                }
            }
        }
        panic!(
            "SendInput admitted {sent}/{}; raw native error: {primary}",
            records.len()
        );
    }
    assert_eq!(sent, records.len(), "actual OS input insertion count");
}

fn record(scan: u16, flags: KEYBD_EVENT_FLAGS) -> INPUT {
    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wScan: scan,
                dwFlags: flags,
                ..Default::default()
            },
        },
    }
}

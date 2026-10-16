//! Exact physical chords, with immediately retained partial-insertion errors.

mod law;

use serde::Serialize;
use std::io;
use windows::Win32::{
    Foundation::HWND,
    UI::{
        Input::KeyboardAndMouse::{
            GetFocus, GetKeyboardLayout, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBD_EVENT_FLAGS,
            KEYBDINPUT, KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP, KEYEVENTF_SCANCODE,
            MAPVK_VK_TO_VSC_EX, MapVirtualKeyExW, SendInput,
        },
        WindowsAndMessaging::{GetForegroundWindow, GetWindowThreadProcessId, IsChild},
    },
};

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub(super) struct Stroke {
    pub(super) key: u32,
    pub(super) scan: u16,
    pub(super) flags: u32,
}

#[derive(Clone, Debug, Serialize)]
pub(super) struct Chord {
    pub(super) letter: u32,
    pub(super) focus: String,
    pub(super) foreground: String,
    pub(super) thread: u32,
    pub(super) process: u32,
    pub(super) layout: String,
    pub(super) strokes: Vec<Stroke>,
    pub(super) admitted: Option<u32>,
    pub(super) insertion_error: Option<i32>,
    pub(super) releases: Vec<(usize, u32, Option<i32>)>,
}

pub(super) fn insert(parent: HWND, letter: u32, output: &mut Vec<Chord>) -> io::Result<()> {
    // SAFETY: only immediate current-owner window/layout reads precede input.
    let (focus, foreground, owned, thread, process, layout) = unsafe {
        let focus = GetFocus();
        let foreground = GetForegroundWindow();
        let owned = focus != parent && IsChild(parent, focus).as_bool();
        let mut process = 0;
        let thread = GetWindowThreadProcessId(focus, Some(&mut process));
        (
            focus,
            foreground,
            owned,
            thread,
            process,
            GetKeyboardLayout(thread),
        )
    };
    if focus.is_invalid() || foreground != parent || !owned || thread == 0 || layout.is_invalid() {
        return Err(io::Error::other(
            "physical fixture lacks an owned foreground focus/layout",
        ));
    }
    let mut strokes = Vec::with_capacity(6);
    let mut records = Vec::with_capacity(6);
    let keys = [0x12, 0x10, letter];
    let mut mapped = [0; 3];
    for (index, key) in keys.into_iter().enumerate() {
        // SAFETY: map each held key once; its release reuses this exact mapping.
        mapped[index] = unsafe { MapVirtualKeyExW(key, MAPVK_VK_TO_VSC_EX, Some(layout)) };
        if mapped[index].trailing_zeros() >= 8 || !matches!(mapped[index] >> 8, 0 | 0xe0) {
            return Err(io::Error::other(
                "physical fixture has an unsupported actual key mapping",
            ));
        }
    }
    for (index, released) in [
        (0, false),
        (1, false),
        (2, false),
        (2, true),
        (1, true),
        (0, true),
    ] {
        let scan = u16::try_from(mapped[index] & 0xff).expect("mapped scan byte");
        let flags = KEYEVENTF_SCANCODE
            | if mapped[index] >> 8 == 0xe0 {
                KEYEVENTF_EXTENDEDKEY
            } else {
                KEYBD_EVENT_FLAGS::default()
            }
            | if released {
                KEYEVENTF_KEYUP
            } else {
                KEYBD_EVENT_FLAGS::default()
            };
        strokes.push(Stroke {
            key: keys[index],
            scan,
            flags: flags.0,
        });
        records.push(INPUT {
            r#type: INPUT_KEYBOARD,
            Anonymous: INPUT_0 {
                ki: KEYBDINPUT {
                    wScan: scan,
                    dwFlags: flags,
                    ..Default::default()
                },
            },
        });
    }
    output.push(Chord {
        letter,
        focus: format!("{focus:?}"),
        foreground: format!("{foreground:?}"),
        thread,
        process,
        layout: format!("{layout:?}"),
        strokes,
        admitted: None,
        insertion_error: None,
        releases: Vec::new(),
    });
    let chord = output.last_mut().expect("actual admitted chord owner");
    // SAFETY: records hold six real INPUT values, including every matching release.
    let sent = unsafe {
        SendInput(
            &records,
            i32::try_from(size_of::<INPUT>()).expect("INPUT size"),
        )
    };
    let primary = (sent != 6).then(io::Error::last_os_error);
    chord.admitted = Some(sent);
    chord.insertion_error = primary.as_ref().and_then(io::Error::raw_os_error);
    if let Some(primary) = primary {
        for up in law::releases(sent) {
            // SAFETY: release only a down known admitted without its matching up.
            let released = unsafe {
                SendInput(
                    &records[up..=up],
                    i32::try_from(size_of::<INPUT>()).expect("INPUT size"),
                )
            };
            let error = (released != 1).then(io::Error::last_os_error);
            chord.releases.push((
                up,
                released,
                error.as_ref().and_then(io::Error::raw_os_error),
            ));
        }
        return Err(primary);
    }
    Ok(())
}
